"""Define the base Vector and Transfer classes."""
from copy import deepcopy
from contextlib import contextmanager
import os
import weakref

import numpy as np

from openmdao.utils.name_maps import prom_name2abs_name, rel_name2abs_name


_full_slice = slice(None)
_type_map = {
    'input': 'input',
    'output': 'output',
    'residual': 'output'
}

# This is the dtype we use for index arrays.  Petsc by default uses 32 bit ints
if os.environ.get('OPENMDAO_USE_BIG_INTS'):
    INT_DTYPE = np.dtype(np.int64)
else:
    INT_DTYPE = np.dtype(np.int32)


class Vector(object):
    """
    Base Vector class.

    This class is instantiated for inputs, outputs, and residuals.
    It provides a dictionary interface and an arithmetic operations interface.
    Implementations:

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

        self._system = weakref.ref(system)

        self._iproc = system.comm.rank
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

        self.read_only = False

    def __str__(self):
        """
        Return a string representation of the Vector object.

        Returns
        -------
        str
            String rep of this object.
        """
        raise NotImplementedError('__str__ not defined for vector type %s' %
                                  type(self).__name__)
        return None  # silence lint warning

    def __len__(self):
        """
        Return the flattened length of this Vector.

        Returns
        -------
        int
            Total flattened length of this vector.
        """
        raise NotImplementedError('__len__ not defined for vector type %s' %
                                  type(self).__name__)
        return None  # silence lint warning

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
        raise NotImplementedError('values not defined for vector type %s' %
                                  type(self).__name__)
        return None  # silence lint warning

    def name2abs_name(self, name):
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
        abs_name = name if system.pathname == '' else '.'.join((system.pathname, name))
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
        path = system.pathname
        idx = len(path) + 1 if path else 0

        return (n[idx:] for n in system._var_abs_names[self._typ] if n in self._names)

    def _abs_val_iter(self, flat=True):
        """
        Iterate over the items in the vector, using absolute names.

        Parameters
        ----------
        flat : bool
            If True, return the flattened values.
        """
        raise NotImplementedError('_abs_val_iter not defined for vector type %s' %
                                  type(self).__name__)

    def _abs_iter(self):
        """
        Iterate over the absolute names in the vector.
        """
        raise NotImplementedError('_abs_iter not defined for vector type %s' %
                                  type(self).__name__)

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
        return self.name2abs_name(name) is not None

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
        raise NotImplementedError('__getitem__ not defined for vector type %s' %
                                  type(self).__name__)
        return None  # silence lint warning

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
        raise NotImplementedError('_abs_get_val not defined for vector type %s' %
                                  type(self).__name__)
        return None  # silence lint warning

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
        raise NotImplementedError('__setitem__ not defined for vector type %s' %
                                  type(self).__name__)

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

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            vector to add to self.
        """
        raise NotImplementedError('__iadd__ not defined for vector type %s' %
                                  type(self).__name__)

    def __isub__(self, vec):
        """
        Perform in-place vector substraction.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            vector to subtract from self.
        """
        raise NotImplementedError('__isub__ not defined for vector type %s' %
                                  type(self).__name__)

    def __imul__(self, val):
        """
        Perform in-place scalar multiplication.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            scalar to multiply self.
        """
        raise NotImplementedError('__imul__ not defined for vector type %s' %
                                  type(self).__name__)

    def add_scal_vec(self, val, vec):
        """
        Perform in-place addition of a vector times a scalar.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            scalar.
        vec : <Vector>
            this vector times val is added to self.
        """
        raise NotImplementedError('add_scale_vec not defined for vector type %s' %
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
        Return an array representation of this vector.

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
        raise NotImplementedError('set_var not defined for vector type %s' %
                                  type(self).__name__)

    def dot(self, vec):
        """
        Compute the dot product of the current vec and the incoming vec.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            The incoming vector being dotted with self.

        Returns
        -------
        float or ndarray
            dot product of this vector and vec
        """
        raise NotImplementedError('dot not defined for vector type %s' %
                                  type(self).__name__)
        return None  # silence lint warning.

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
        return None  # silence lint warning.

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
        raise NotImplementedError('set_complex_step_mode not defined for vector type %s' %
                                  type(self).__name__)


@contextmanager
def _matvec_context(d_inputs, d_outputs, d_residuals, scope_out, scope_in, mode, clear=True):
    """
    Context manager for vectors.

    For the given vec_name, return vectors that use a set of
    internal variables that are relevant to the current matrix-vector
    product.  This is called only from _apply_linear.

    Parameters
    ----------
    d_inputs : <Vector>
        Input derivatives vector.
    d_outputs : <Vector>
        Output derivatives vector.
    d_residuals : <Vector>
        Residual derivatives vector.
    scope_out : frozenset or None
        Set of absolute output names in the scope of this mat-vec product.
        If None, all are in the scope.
    scope_in : frozenset or None
        Set of absolute input names in the scope of this mat-vec product.
        If None, all are in the scope.
    mode : str
        Key for specifying derivative direction. Values are 'fwd'
        or 'rev'.
    clear : bool(True)
        If True, zero out residuals (in fwd mode) or inputs and outputs
        (in rev mode).

    Yields
    ------
    (d_inputs, d_outputs, d_residuals) : tuple of Vectors
        Yields the three Vectors configured internally to deal only
        with variables relevant to the current matrix vector product.

    """
    if clear:
        if mode == 'fwd':
            d_residuals.set_val(0.0)
        else:  # rev
            d_inputs.set_val(0.0)
            d_outputs.set_val(0.0)

    if scope_out is None and scope_in is None:
        yield d_inputs, d_outputs, d_residuals
    else:
        old_ins = d_inputs._names
        old_outs = d_outputs._names

        if scope_out is not None:
            d_outputs._names = scope_out.intersection(d_outputs._abs_iter())
        if scope_in is not None:
            d_inputs._names = scope_in.intersection(d_inputs._abs_iter())

        yield d_inputs, d_outputs, d_residuals

        # reset _names so users will see full vector contents
        d_inputs._names = old_ins
        d_outputs._names = old_outs
