"""Define the base Vector and Transfer classes."""
from copy import deepcopy
import os
import weakref
from collections import OrderedDict

import numpy as np

from openmdao.utils.name_maps import prom_name2abs_name, rel_name2abs_name
from openmdao.utils.array_utils import yield_slice_info

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
    _names : set([str, ...])
        Set of variables that are relevant in the current context.
    _full_names : set([str, ...])
        Full set of variables contained in this vector.
    _root_vector : Vector
        Pointer to the vector owned by the root system.
    _alloc_complex : Bool
        If True, then space for the complex vector is also allocated.
    _data : ndarray
        Actual allocated data.
    _cplx_data : ndarray
        Actual allocated data under complex step.
    _under_complex_step : bool
        When True, this vector is under complex step, and data is swapped with the complex data.
    _ncol : int
        Number of columns for multi-vectors.
    _icol : int or None
        If not None, specifies the 'active' column of a multivector when interfaceing with
        a component that does not support multivectors.
    _relevant : dict
        Mapping of a VOI to a tuple containing dependent inputs, dependent outputs,
        and dependent systems.
    _do_scaling : bool
        True if this vector performs scaling.
    _scaling : dict
        Contains scale factors to convert data arrays.
    read_only : bool
        When True, values in the vector cannot be changed via the user __setitem__ API.
    _under_complex_step : bool
        When True, self._data is replaced with self._cplx_data.
    _age : int
        Used to determine if this vector needs to update its cached values from the root vector.
    _start : int
        Starting index for this vector within the root vector.
    _stop : int
        Ending index + 1 for this vector within the root vector.
    _offset_view : OffsetDict
        Dict of range and shape for each variable into this vector.
    """

    # Listing of relevant citations that should be referenced when
    cite = ""

    def __init__(self, name, kind, system, root_vector=None, resize=False, alloc_complex=False,
                 ncol=1, relevant=None):
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
        resize : bool
            If true, resize the root vector.
        alloc_complex : bool
            Whether to allocate any imaginary storage to perform complex step. Default is False.
        ncol : int
            Number of columns for multi-vectors.
        relevant : dict
            Mapping of a VOI to a tuple containing dependent inputs, dependent outputs,
            and dependent systems.
        """
        self._name = name
        self._typ = _type_map[kind]
        self._kind = kind
        self._ncol = ncol
        self._icol = None
        self._relevant = relevant
        self._age = -1
        self._offset_view = None

        self._system = weakref.ref(system)

        self._iproc = system.comm.rank

        # self._names will be the set of variables relevant to the current matvec product.
        self._names = frozenset(system._var_relevant_names[self._name][self._typ])
        self._full_names = self._names

        self._root_vector = None
        self._data = None
        self._start = None
        self._stop = None

        # Support for Complex Step
        self._alloc_complex = alloc_complex
        self._cplx_data = None
        self._under_complex_step = False

        if root_vector is None:
            self._root_vector = self
        else:
            self._root_vector = root_vector

        if resize:
            if root_vector is None:
                raise RuntimeError(
                    'Cannot resize the vector because the root vector has not yet '
                    'been created in system %s' % system.pathname)
            self._update_root_data()

        self._do_scaling = ((kind == 'input' and system._has_input_scaling) or
                            (kind == 'output' and system._has_output_scaling) or
                            (kind == 'residual' and system._has_resid_scaling))

        self._initialize_data(root_vector)
        # self._init_scaling()

        self.read_only = False

    def __str__(self):
        """
        Return a string representation of the Vector object.

        Returns
        -------
        str
            String rep of this object.
        """
        try:
            return str(self._data)
        except Exception as err:
            return "<error during call to Vector.__str__>: %s" % err

    def __len__(self):
        """
        Return the flattened length of this Vector.

        Returns
        -------
        int
            Total flattened length of this vector.
        """
        return self._data.size

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
        return [v for n, v in self.abs_item_iter()]

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
        abs_name = prom_name2abs_name(system, name, self._typ)
        if abs_name in self._names:
            return abs_name

        abs_name = rel_name2abs_name(system, name)
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

    def abs_iter(self):
        """
        Yield an iterator over variables involved in the current mat-vec product (absolute names).

        Returns
        -------
        listiterator
            iterator over the variable names.
        """
        return (n for n in self._system()._var_abs_names[self._typ] if n in self._names)

    def abs_item_iter(self):
        """
        Yield an iterator over names and values involved in the current mat-vec product.

        Names are absolute names.

        Yields
        ------
        tuple (str, ndarray)
            Name and value of each variable.
        """
        vmap, _, _ = self._get_offset_view()
        data = self._data
        for n in vmap:
            if n in self._names:
                start, stop, shape = vmap[n]
                val = data[start:stop]
                val.shape = shape
                yield (n, val)

    def __contains__(self, name):
        """
        Check if the variable is involved in the current mat-vec product.

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

    def contains_abs(self, abs_name):
        """
        Check if the variable is involved in the current mat-vec product.

        Parameters
        ----------
        abs_name : str
            Absolute variable name.

        Returns
        -------
        boolean
            True or False.
        """
        return abs_name in self._names

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
        abs_name = self.name2abs_name(name)
        if abs_name is not None:
            return self.get_view(abs_name)
        else:
            raise KeyError('Variable name "{}" not found.'.format(name))

    def get_view(self, abs_name):
        """
        Return the view associated with the given absolute name.

        Parameters
        ----------
        abs_name : str
            The absolute variable name.

        Returns
        -------
        ndarray
            Array view corresponding to the given variable name.
        """
        if abs_name not in self._names:
            raise KeyError('Variable name "{}" not found.'.format(abs_name))

        start, stop, shape = self._get_offset_view()[0][abs_name]

        if self._ncol == 1:
            val = self._data[start:stop]
        else:
            start = start // self._ncol
            stop = stop // self._ncol
            if self._icol is None:
                shape = tuple(list(shape) + [self._ncol])
                val = self._data[start:stop]
            else:
                val = self._data[start:stop, self._icol]

        val.shape = shape
        return val

    def get_flat_view(self, abs_name):
        """
        Get the variable value in flattened form.

        Parameters
        ----------
        abs_name : str
            Absolute variable name.

        Returns
        -------
        float or ndarray
            flattened variable value.
        """
        if abs_name not in self._names:
            raise KeyError('Variable name "{}" not found.'.format(abs_name))

        vmap, _, _ = self._get_offset_view()
        start, stop, _ = vmap[abs_name]

        if self._ncol > 1:
            start = start // self._ncol
            stop = stop // self._ncol
            if self._icol is not None:
                return self._data[start:stop][:, self._icol]

        return self._data[start:stop]

    def get_var_to_var_root_range(self, var1, var2):
        """
        Return the slice that contains variables var1 to var2.

        The variables are assumed to be in ascending order by index.

        Parameters
        ----------
        var1 : str
            Name of starting variable.
        var2 : str
            Name of ending variable.

        Returns
        -------
        int
            The offset of this array within the root array.
        int
            The end index + 1 of this array within the root array.
        """
        vmap, _, _ = self._root_vector._get_offset_view()

        # get the extent of the slice this entire vec occupies in the root vec
        start = vmap[var1][0]
        stop = vmap[var2][1]

        if self._ncol > 1:
            start = start // self._ncol
            stop = stop // self._ncol

        return start, stop

    def get_root_range(self):
        """
        Return the slice that this vector occupies in the root vector.

        Returns
        -------
        int
            The offset of this array within the root array.
        int
            The end index + 1 of this array within the root array.
        """
        if self is self._root_vector:
            return 0, self._data.size

        abs_names = self._system()._var_relevant_names[self._name][self._typ]

        if abs_names:
            return self.get_var_to_var_root_range(abs_names[0], abs_names[-1])

        return 0, 0

    def _get_offset_view(self):
        if self._age != self._root_vector._age or self._age == -1:
            self._offset_view, self._start, self._stop = self._create_offset_view()
            self._data = self._root_vector._data[self._start:self._stop]
            if self._alloc_complex:
                self._cplx_data = self._root_vector._cplx_data[self._start:self._stop]
                if self._under_complex_step and self._data.dtype != np.complex:
                    self._data, self._cplx_data = self._cplx_data, self._data

            if self._age == -1:  # only will happen for root
                self._age = 0

        return self._offset_view, self._start, self._stop

    def _create_offset_view(self):
        vmap, _ = self._root_vector._system().get_var_range_info(self._name, self._typ, self._ncol)
        start, stop = self.get_root_range()
        self._age = self._root_vector._age
        if self is self._root_vector:
            dct = vmap
        else:
            strt = start if self._ncol == 1 else start * self._ncol
            dct = _OffsetDict(vmap, strt,
                              self._system()._var_relevant_names[self._name][self._typ])
        return dct, start, stop

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
        abs_name = self.name2abs_name(name)
        if abs_name is not None:
            if self.read_only:
                msg = "Attempt to set value of '{}' in {} vector when it is read only."
                raise ValueError(msg.format(name, self._kind))

            self.set_val_abs(abs_name, value)
        else:
            msg = 'Variable name "{}" not found.'
            raise KeyError(msg.format(name))

    def set_val_abs(self, abs_name, value):
        """
        Set the variable value.

        Parameters
        ----------
        abs_name : str
            Absolute variable name.
        value : float or list or tuple or ndarray
            variable value to set
        """
        oldval = self.get_view(abs_name)
        value = np.asarray(value)
        if value.shape != () and value.shape != (1,) and oldval.shape != value.shape:
            raise ValueError("Incompatible shape for '%s': Expected %s but got %s." %
                             (abs_name, oldval.shape, value.shape))

        oldval[:] = value

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

        Must be implemented by the subclass.

        Sets the following attributes:

        - _data

        Parameters
        ----------
        root_vector : <Vector> or None
            the root's vector instance or None, if we are at the root.
        """
        raise NotImplementedError('_initialize_data not defined for vector type %s' %
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
        _, start, stop = self._get_offset_view()
        adder, scaler = self._root_vector._scaling[scale_to]
        if self._ncol == 1:
            self._data *= scaler[start:stop]
            if adder is not None:  # nonlinear only
                self._data += adder[start:stop]
        else:
            self._data *= scaler[start:stop, np.newaxis]
            if adder is not None:  # nonlinear only
                self._data += adder[start:stop, np.newaxis]

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

    def set_val(self, val):
        """
        Set the value of this vector to a value.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            scalar to set self to.
        """
        raise NotImplementedError('set_val not defined for vector type %s' %
                                  type(self).__name__)

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

    def _enforce_bounds_vector(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds, backtracking the entire vector together.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        raise NotImplementedError('_enforce_bounds_vector not defined for vector type %s' %
                                  type(self).__name__)

    def _enforce_bounds_scalar(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds on each scalar separately, then backtrack as a vector.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        raise NotImplementedError('_enforce_bounds_scalar not defined for vector type %s' %
                                  type(self).__name__)

    def _enforce_bounds_wall(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds on each scalar separately, then backtrack along the wall.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        raise NotImplementedError('_enforce_bounds_wall not defined for vector type %s' %
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
            self._cplx_data[:] = self._data

        elif keep_real:
            self._cplx_data[:] = self._data.real

        self._data, self._cplx_data = self._cplx_data, self._data
        self._under_complex_step = active


class _OffsetDict(object):
    """
    Fake dictionary that returns offset values for variable ranges.

    Attributes
    ----------
    _root_dct : OrderedDict
        Mapping of var name to start, stop, shape in root vector.
    _offset : int
        The offset of these ranges from the root vector ranges.
    _vnames : list of str
        List of names of variables in this fake dict.
    """

    __slots__ = ['_root_dct', '_offset', '_vnames']

    def __init__(self, root_dct, offset, vnames):
        self._root_dct = root_dct
        self._offset = offset
        self._vnames = vnames

    def __getitem__(self, abs_name):
        start, stop, shape = self._root_dct[abs_name]
        return start - self._offset, stop - self._offset, shape

    def __iter__(self):
        return iter(self._vnames)

    def keys(self):
        return iter(self._vnames)

    def values(self):
        rdct = self._root_dct
        off = self._offset
        for n in self._vnames:
            start, stop, shape = rdct[n]
            yield (start - off, stop - off, shape)

    def items(self):
        rdct = self._root_dct
        off = self._offset
        for n in self._vnames:
            start, stop, shape = rdct[n]
            yield (n, (start - off, stop - off, shape))
