"""Define the base Vector and Transfer classes."""
from copy import deepcopy
import os
import weakref
from contextlib import contextmanager

import numpy as np

from openmdao.utils.name_maps import prom_name2abs_name, rel_name2abs_name
# from openmdao.utils.array_utils import Iadd2IsubArray


_full_slice = slice(None)
_type_map = {
    'input': 'input',
    'output': 'output',
    'residual': 'output'
}


class Vector(object):
    """
    Base Vector class.

    This class is instantiated for inputs, outputs, and residuals.
    It provides a dictionary interface and an arithmetic operations interface.
    Implementations:

    - <DefaultVector>
    - <PETScVector>

    Attributes
    ----------
    _name : str
        The name of the vector: 'nonlinear', 'linear', or right-hand side name.
    _typ : str
        Type: 'input' for input vectors; 'output' for output/residual vectors.
    _kind : str
        Specific kind of vector, either 'input', 'output', or 'residual'.
    _system : System
        Pointer to the owning system.
    _iproc : int
        Global processor index.
    _length : int
        Length of flattened vector.
    _views : dict
        Dictionary mapping absolute variable names to the ndarray views.
    _views_flat : dict
        Dictionary mapping absolute variable names to the flattened ndarray views.
    _names : set([str, ...])
        Set of variables that are relevant in the current context.
    _root_vector : Vector
        Pointer to the vector owned by the root system.
    _alloc_complex : Bool
        If True, then space for the complex vector is also allocated.
    _data : ndarray
        Actual allocated data.
    _slices : dict
        Mapping of var name to slice.
    _cplx_data : ndarray
        Actual allocated data under complex step.
    _cplx_views : dict
        Dictionary mapping absolute variable names to the ndarray views under complex step.
    _cplx_views_flat : dict
        Dictionary mapping absolute variable names to the flattened ndarray views under complex
        step.
    _under_complex_step : bool
        When True, this vector is under complex step, and data is swapped with the complex data.
    _ncol : int
        Number of columns for multi-vectors.
    _icol : int or None
        If not None, specifies the 'active' column of a multivector when interfaceing with
        a component that does not support multivectors.
    _do_scaling : bool
        True if this vector performs scaling.
    _scaling : dict
        Contains scale factors to convert data arrays.
    read_only : bool
        When True, values in the vector cannot be changed via the user __setitem__ API.
    _under_complex_step : bool
        When True, self._data is replaced with self._cplx_data.
    _len : int
        Total length of data vector (including shared memory parts).
    _data_len : int
        Total length of only the internal _data array (does not include shared memory parts).
    _neg : bool
        Currently does nothing.
    """

    # Listing of relevant citations that should be referenced when
    cite = ""

    def __init__(self, name, kind, system, root_vector=None, alloc_complex=False, ncol=1):
        """
        Initialize all attributes.

        Parameters
        ----------
        name : str
            The name of the vector: 'nonlinear', 'linear', or right-hand side name.
        kind : str
            The kind of vector, 'input', 'output', or 'residual'.
        system : <System>
            Pointer to the owning system.
        root_vector : <Vector>
            Pointer to the vector owned by the root system.
        alloc_complex : bool
            Whether to allocate any imaginary storage to perform complex step. Default is False.
        ncol : int
            Number of columns for multi-vectors.
        """
        self._name = name
        self._typ = _type_map[kind]
        self._kind = kind
        self._ncol = ncol
        self._icol = None
        self._len = 0
        self._data_len = 0
        self._neg = False

        self._system = weakref.ref(system)

        self._views = {}
        self._views_flat = {}

        # self._names will either contain the same names as self._views or to the
        # set of variables relevant to the current matvec product.
        self._names = self._views

        self._root_vector = None
        self._data = None
        self._slices = None

        # Support for Complex Step
        self._alloc_complex = alloc_complex
        self._cplx_data = None
        self._cplx_views = {}
        self._cplx_views_flat = {}
        self._under_complex_step = False

        self._do_scaling = ((kind == 'input' and system._has_input_scaling) or
                            (kind == 'output' and system._has_output_scaling) or
                            (kind == 'residual' and system._has_resid_scaling))

        self._scaling = {}

        if root_vector is None:
            self._root_vector = self
        else:
            self._root_vector = root_vector

        self._initialize_data(root_vector)
        self._initialize_views()

        if self._len != self._data_len:
            self._init_nocopy()

        self.read_only = False

    def __str__(self):
        """
        Return a string representation of the Vector object.

        Returns
        -------
        str
            String rep of this object.
        """
        return str(self.asarray())

    def __repr__(self):
        """
        Return a simple string representation.

        Returns
        -------
        str
            A string representation.
        """
        return f"{self.__class__.__name__}({self.asarray()})"

    def __len__(self):
        """
        Return the length of the array that would be returned from self.asarray().

        Note that this may be a larger array than this Vector's actual internal _data
        array if this is an input vector that shares some of its memory with its connected
        outputs.

        Returns
        -------
        int
            Total flattened length of this vector, including shared memory parts.
        """
        return self._len

    def data_len(self):
        """
        Return the flattened length of this Vector's data array.

        Note that for input vectors, this data array may be smaller than the array returned
        from self.asarray() since some input entries will share memory with their connected outputs.

        Returns
        -------
        int
            The length of the internal _data array.
        """
        return self._data_len

    def _get_arr(self, arr):
        """
        Return an ndarray or a Iadd2IsubArray depending on the value of self._neg.

        Parameters
        ----------
        arr : ndarray
            Array containing data for return value.

        Returns
        -------
        ndarray or Iadd2IsubArray
            The current type of array based on self._neg.
        """
        # if self._neg:
        #     return arr.view(Iadd2IsubArray)  # inverts __iadd__ and __isub__
        return arr

    def _negative_mode(self, neg=True):
        self._neg = neg
        # if neg:
        #     self._data = self._data.view(Iadd2IsubArray)
        # else:
        #     self._data = self._data.view(np.ndarray)

    def _init_nocopy(self):
        """
        Switch over some methods to nocopy versions in order to avoid overhead for other versions.
        """
        # self.get_nocopy = self._get_nl_input_nocopy
        self.set_val = self._nocopy_set_val
        self.asarray = self._nocopy_asarray
        self.iadd = self._nocopy_iadd
        self.isub = self._nocopy_isub
        self.imul = self._nocopy_imul

    def _copy_views(self):
        """
        Return a dictionary containing just the views.

        Returns
        -------
        dict
            Dictionary containing the _views.
        """
        return deepcopy(self._views)

    def keys(self):
        """
        Return variable names of variables contained in this vector (relative names).

        Returns
        -------
        listiterator (Python 3.x) or list (Python 2.x)
            the variable names.
        """
        return self.__iter__()

    def values(self):
        """
        Return values of variables contained in this vector.

        Returns
        -------
        list
            the variable values.
        """
        return [v for n, v in self._views.items() if n in self._names]

    def _name2abs_name(self, name):
        """
        Map the given promoted or relative name to the absolute name.

        This is only valid when the name is unique; otherwise, a KeyError is thrown.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        str or None
            Absolute variable name if unique abs_name found or None otherwise.
        """
        system = self._system()

        # try relative name first
        abs_name = '.'.join((system.pathname, name)) if system.pathname else name
        if abs_name in self._names:
            return abs_name

        abs_name = prom_name2abs_name(system, name, self._typ)
        if abs_name in self._names:
            return abs_name

    def __iter__(self):
        """
        Yield an iterator over variables involved in the current mat-vec product (relative names).

        Returns
        -------
        listiterator
            iterator over the variable names.
        """
        system = self._system()
        idx = len(system.pathname) + 1 if system.pathname else 0
        names = self._names
        return (n[idx:] for n in system._var_abs2meta[self._typ] if n in names)

    def _abs_item_iter(self, flat=True):
        """
        Iterate over the items in the vector, using absolute names.

        Parameters
        ----------
        flat : bool
            If True, return the flattened values.
        """
        arrs = self._views_flat if flat else self._views
        yield from arrs.items()

    def _abs_iter(self):
        """
        Iterate over the absolute names in the vector.
        """
        yield from self._views

    def __contains__(self, name):
        """
        Check if the variable is found in this vector.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        boolean
            True or False.
        """
        return self._name2abs_name(name) is not None

    def _contains_abs(self, name):
        """
        Check if the variable is found in this vector.

        Parameters
        ----------
        name : str
            Absolute variable name.

        Returns
        -------
        boolean
            True or False.
        """
        return name in self._names

    def __getitem__(self, name):
        """
        Get the variable value.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        float or ndarray
            variable value.
        """
        abs_name = self._name2abs_name(name)
        if abs_name is not None:
            if self._icol is None:
                return self._get_arr(self._views[abs_name])
            else:
                return self._get_arr(self._views[abs_name][:, self._icol])
            return val
        else:
            raise KeyError(f"{self._system().msginfo}: Variable name '{name}' not found.")

    def _abs_get_val(self, name, flat=True):
        """
        Get the variable value using the absolute name.

        No error checking is performed on the name.

        Parameters
        ----------
        name : str
            Absolute name in the owning system's namespace.
        flat : bool
            If True, return the flat value.

        Returns
        -------
        float or ndarray
            variable value.
        """
        return self._get_arr(self._views_flat[name] if flat else self._views[name])

    def __setitem__(self, name, value):
        """
        Set the variable value.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.
        value : float or list or tuple or ndarray
            variable value to set
        """
        self.set_var(name, value)

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

        Must be implemented by the subclass.

        Parameters
        ----------
        root_vector : <Vector> or None
            the root's vector instance or None, if we are at the root.
        """
        raise NotImplementedError('_initialize_data not defined for vector type %s' %
                                  type(self).__name__)

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Must be implemented by the subclass.
        """
        raise NotImplementedError('_initialize_views not defined for vector type %s' %
                                  type(self).__name__)

    def __iadd__(self, vec):
        """
        Perform in-place vector addition.

        Parameters
        ----------
        vec : <Vector> or ndarray
            vector or array to add to self.

        Returns
        -------
        <Vector>
            self + vec
        """
        if isinstance(vec, Vector):
            self.iadd(vec.asarray())
        else:
            self.iadd(vec)
        return self

    def __isub__(self, vec):
        """
        Perform in-place vector substraction.

        Parameters
        ----------
        vec : <Vector>
            vector to subtract from self.

        Returns
        -------
        <Vector>
            self - vec
        """
        if isinstance(vec, Vector):
            self.isub(vec.asarray())
        else:
            self.isub(vec)
        return self

    def __imul__(self, vec):
        """
        Perform in-place multiplication.

        Parameters
        ----------
        vec : Vector, int, float or ndarray
            Value to multiply self.

        Returns
        -------
        <Vector>
            self * vec
        """
        if isinstance(vec, Vector):
            self.imul(vec.asarray())
        else:
            self.imul(vec)
        return self

    def iadd(self, val, idxs=_full_slice):
        """
        Perform in-place vector addition.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        raise NotImplementedError('iadd not defined for vector type %s' %
                                  type(self).__name__)

    def isub(self, val, idxs=_full_slice):
        """
        Subtract the value from the data array at the specified indices or slice(s).

        Must be implemented by the subclass.

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        raise NotImplementedError('isub not defined for vector type %s' %
                                  type(self).__name__)

    def imul(self, val, idxs=_full_slice):
        """
        Perform in-place scalar multiplication.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        raise NotImplementedError('imul not defined for vector type %s' %
                                  type(self).__name__)

    def scale(self, scale_to):
        """
        Scale this vector to normalized or physical form.

        Parameters
        ----------
        scale_to : str
            Values are "phys" or "norm" to scale to physical or normalized.
        """
        raise NotImplementedError('scale not defined for vector type %s' %
                                  type(self).__name__)

    def asarray(self, copy=False):
        """
        Return a flat array representation of this vector.

        If copy is True, return a copy.  Otherwise, try to avoid it.

        Parameters
        ----------
        copy : bool
            If True, return a copy of the array.

        Returns
        -------
        ndarray
            Array representation of this vector.
        """
        raise NotImplementedError('asarray not defined for vector type %s' %
                                  type(self).__name__)
        return None  # silence lint warning

    def iscomplex(self):
        """
        Return True if this vector contains complex values.

        This checks the type of the values, not whether they have a nonzero imaginary part.

        Returns
        -------
        bool
            True if this vector contains complex values.
        """
        raise NotImplementedError('iscomplex not defined for vector type %s' %
                                  type(self).__name__)
        return False  # silence lint warning

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        raise NotImplementedError('set_vec not defined for vector type %s' %
                                  type(self).__name__)

    def set_val(self, val, idxs=_full_slice):
        """
        Set the data array of this vector to a scalar or array value, with optional indices.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : float or ndarray
            scalar or array to set data array to.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        raise NotImplementedError('set_arr not defined for vector type %s' %
                                  type(self).__name__)

    def set_var(self, name, val, idxs=_full_slice):
        """
        Set the array view corresponding to the named variable, with optional indexing.

        Parameters
        ----------
        name : str
            The name of the variable.
        val : float or ndarray
            Scalar or array to set data array to.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        abs_name = self._name2abs_name(name)
        if abs_name is None:
            raise KeyError(f"{self._system().msginfo}: Variable name '{name}' not found.")

        if self.read_only:
            raise ValueError(f"{self._system().msginfo}: Attempt to set value of '{name}' in "
                             f"{self._kind} vector when it is read only.")

        if self._icol is not None:
            idxs = (idxs, self._icol)

        value = np.asarray(val)

        try:
            self._views[abs_name][idxs] = value
        except Exception as err:
            try:
                value = value.reshape(self._views[abs_name][idxs].shape)
            except Exception:
                raise ValueError(f"{self._system().msginfo}: Failed to set value of "
                                 f"'{name}': {str(err)}.")
            self._views[abs_name][idxs] = value

    def dot(self, vec):
        """
        Compute the dot product of the current vec and the incoming vec.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            The incoming vector being dotted with self.
        """
        raise NotImplementedError('dot not defined for vector type %s' %
                                  type(self).__name__)

    def get_norm(self):
        """
        Return the norm of this vector.

        Must be implemented by the subclass.

        Returns
        -------
        float
            norm of this vector.
        """
        raise NotImplementedError('get_norm not defined for vector type %s' %
                                  type(self).__name__)
        return None  # silence lint warning about missing return value.

    def _in_matvec_context(self):
        """
        Return True if this vector is inside of a matvec_context.
        """
        raise NotImplementedError('_in_matvec_context not defined for vector type %s' %
                                  type(self).__name__)

    def set_complex_step_mode(self, active, keep_real=False):
        """
        Turn on or off complex stepping mode.

        When turned on, the default real ndarray is replaced with a complex ndarray and all
        pointers are updated to point to it.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.

        keep_real : bool
            When this flag is True, keep the real value when turning off complex step. You only
            need to do this when temporarily disabling complex step for guess_nonlinear.
        """
        if active:
            arr = self.asarray()
        elif keep_real:
            arr = self.asarray().real
        else:
            arr = None

        self._data, self._cplx_data = self._cplx_data, self._data
        self._views, self._cplx_views = self._cplx_views, self._views
        self._views_flat, self._cplx_views_flat = self._cplx_views_flat, self._views_flat
        self._under_complex_step = active

        if arr is not None and arr.size > 0:
            self.set_val(arr)
