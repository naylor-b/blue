"""Define the default Vector class."""
from copy import deepcopy
import numbers

import numpy as np

from openmdao.vectors.vector import Vector, INT_DTYPE, _full_slice
from openmdao.vectors.default_transfer import DefaultTransfer
from openmdao.utils.mpi import MPI, multi_proc_exception_check


class DefaultVector(Vector):
    """
    Default NumPy vector.

    Attributes
    ----------
    _len : int
        Total length of data vector (including shared memory parts).
    """

    TRANSFER = DefaultTransfer

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
        self._len = 0
        super(DefaultVector, self).__init__(name, kind, system, root_vector, alloc_complex, ncol)

    def __len__(self):
        """
        Return the flattened length of this Vector.

        Returns
        -------
        int
            Total flattened length of this vector.
        """
        return self._len

    def __str__(self):
        """
        Return a string representation of the Vector object.

        Returns
        -------
        str
            String rep of this object.
        """
        return str(self.asarray())

    def _abs_val_iter(self, flat=True):
        """
        Iterate over the items in the vector, using absolute names.

        Parameters
        ----------
        flat : bool
            If True, return the flattened values.
        """
        arrs = self._views_flat if flat else self._views

        names = self._names
        if len(names) == len(self._views):
            yield from arrs.items()
        else:
            for name, val in arrs.items():
                if name in names:
                    yield name, val

    def _abs_iter(self):
        """
        Iterate over the absolute names in the vector.
        """
        names = self._names
        for name in self._views:
            if name in names:
                yield name

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
            if self._icol is None:
                return self._views[abs_name]
            else:
                return self._views[abs_name][:, self._icol]
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
        if flat:
            return self._views_flat[name]
        return self._views[name]

    def values(self):
        """
        Return values of variables contained in this vector.

        Returns
        -------
        list
            the variable values.
        """
        return [v for n, v in self._views.items() if n in self._names]

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
        abs_name = self.name2abs_name(name)
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
            arr = self._data
        elif keep_real:
            arr = self._data.real
        else:
            arr = None

        self._data, self._cplx_data = self._cplx_data, self._data
        self._views, self._cplx_views = self._cplx_views, self._views
        self._views_flat, self._cplx_views_flat = self._cplx_views_flat, self._views_flat
        self._under_complex_step = active

        if arr is not None:
            self.set_val(arr)

    def _create_data(self):
        """
        Allocate data array.

        This happens only in the top level system.  Child systems use views of the array
        we allocate here.

        Returns
        -------
        ndarray
            zeros array of correct size to hold all of this vector's variables.
        """
        system = self._system()
        ncol = self._ncol
        size = np.sum(system._var_sizes[self._name][self._typ][system.comm.rank, :])
        return np.zeros(size) if ncol == 1 else np.zeros((size, ncol))

    def _extract_root_data(self):
        """
        Extract views of arrays from root_vector.

        Returns
        -------
        ndarray
            zeros array of correct size.
        """
        system = self._system()
        type_ = self._typ
        ncol = self._ncol
        root_vec = self._root_vector

        slices = root_vec.get_slice_dict()

        mynames = system._var_relevant_names[self._name][type_]
        if mynames:
            myslice = slice(slices[mynames[0]].start // ncol, slices[mynames[-1]].stop // ncol)
        else:
            myslice = slice(0, 0)

        data = root_vec._data[myslice]

        # Extract view for complex storage too.
        cplx_data = root_vec._cplx_data[myslice] if self._alloc_complex else None

        scaling = {}
        if self._do_scaling:
            scaling['phys'] = {}
            scaling['norm'] = {}

            for typ in ('phys', 'norm'):
                root_scale = root_vec._scaling[typ]
                rs0 = root_scale[0]
                if rs0 is None:
                    scaling[typ] = (rs0, root_scale[1][myslice])
                else:
                    scaling[typ] = (rs0[myslice], root_scale[1][myslice])

        return data, cplx_data, scaling

    def _initialize_data(self, root_vector):
        """
        Internally allocate data array.

        Parameters
        ----------
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        if root_vector is None:  # we're the root
            self._data = self._create_data()

            if self._do_scaling:
                self._scaling = {}
                data = self._data
                if self._name == 'nonlinear':
                    self._scaling['phys'] = (np.zeros(data.size), np.ones(data.size))
                    self._scaling['norm'] = (np.zeros(data.size), np.ones(data.size))
                elif self._name == 'linear':
                    # reuse the nonlinear scaling vecs since they're the same as ours
                    nlvec = self._system()._root_vecs[self._kind]['nonlinear']
                    self._scaling['phys'] = (None, nlvec._scaling['phys'][1])
                    self._scaling['norm'] = (None, nlvec._scaling['norm'][1])
                else:
                    self._scaling['phys'] = (None, np.ones(data.size))
                    self._scaling['norm'] = (None, np.ones(data.size))

            # Allocate imaginary for complex step
            if self._alloc_complex:
                self._cplx_data = np.zeros(self._data.shape, dtype=np.complex)

        else:
            self._data, self._cplx_data, self._scaling = self._extract_root_data()

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Sets the following attributes:
        _views
        _views_flat
        """
        system = self._system()
        type_ = self._typ
        kind = self._kind
        ncol = self._ncol

        do_scaling = self._do_scaling
        if do_scaling:
            factors = system._scale_factors
            scaling = self._scaling

        self._views = views = {}
        self._views_flat = views_flat = {}

        alloc_complex = self._alloc_complex
        self._cplx_views = cplx_views = {}
        self._cplx_views_flat = cplx_views_flat = {}

        abs2meta = system._var_abs2meta
        start = end = 0
        for abs_name in system._var_relevant_names[self._name][type_]:
            meta = abs2meta[abs_name]
            end = start + meta['size']
            shape = meta['shape']
            if ncol > 1:
                if not isinstance(shape, tuple):
                    shape = (shape,)
                shape = tuple(list(shape) + [ncol])

            views_flat[abs_name] = v = self._data[start:end]
            if shape != v.shape:
                v = v.view()
                v.shape = shape
            views[abs_name] = v

            if alloc_complex:
                cplx_views_flat[abs_name] = v = self._cplx_data[start:end]
                if shape != v.shape:
                    v = v.view()
                    v.shape = shape
                cplx_views[abs_name] = v

            if do_scaling:
                for scaleto in ('phys', 'norm'):
                    scale0, scale1 = factors[abs_name][kind, scaleto]
                    vec = scaling[scaleto]
                    if vec[0] is not None:
                        vec[0][start:end] = scale0
                    vec[1][start:end] = scale1

            start = end

        self._names = frozenset(views)
        self._len = end

    def _in_matvec_context(self):
        """
        Return True if this vector is inside of a matvec_context.

        Returns
        -------
        bool
            Whether or not this vector is in a matvec_context.
        """
        return len(self._names) != len(self._views)

    def __iadd__(self, vec):
        """
        Perform in-place vector addition.

        Parameters
        ----------
        vec : <Vector>
            vector to add to self.

        Returns
        -------
        <Vector>
            self + vec
        """
        if isinstance(vec, Vector):
            self.iadd(vec._data)
        else:
            self._data += vec
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
            self.isub(vec._data)
        else:
            self._data -= vec
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
            self.imul(vec._data)
        else:
            self._data *= vec
        return self

    def add_scal_vec(self, val, vec):
        """
        Perform in-place addition of a vector times a scalar.

        Parameters
        ----------
        val : int or float
            scalar.
        vec : <Vector>
            this vector times val is added to self.
        """
        self._data += (val * vec._data)

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        self._data[:] = vec._data

    def set_val(self, val, idxs=_full_slice):
        """
        Set the data array of this vector to a value, with optional indexing.

        Parameters
        ----------
        val : float or ndarray
            scalar or array to set data array to.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        self._data[idxs] = val

    def scale(self, scale_to):
        """
        Scale this vector to normalized or physical form.

        Parameters
        ----------
        scale_to : str
            Values are "phys" or "norm" to scale to physical or normalized.
        """
        adder, scaler = self._scaling[scale_to]
        if self._ncol == 1:
            self._data *= scaler
            if adder is not None:  # nonlinear only
                self._data += adder
        else:
            self._data *= scaler[:, np.newaxis]
            if adder is not None:  # nonlinear only
                self._data += adder

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
        if copy:
            return self._data.copy()

        return self._data

    def iadd(self, val, idxs=_full_slice):
        """
        Add the value to the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        self._data[idxs] += val

    def isub(self, val, idxs=_full_slice):
        """
        Subtract the value from the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        self._data[idxs] -= val

    def imul(self, val, idxs=_full_slice):
        """
        Multiply the value to the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        self._data[idxs] *= val

    def dot(self, vec):
        """
        Compute the dot product of the real parts of the current vec and the incoming vec.

        Parameters
        ----------
        vec : <Vector>
            The incoming vector being dotted with self.

        Returns
        -------
        float
            The computed dot product value.
        """
        return np.dot(self._data, vec._data)

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        return np.linalg.norm(self._data)

    def get_slice_dict(self):
        """
        Return a dict of var names mapped to their slice in the local data array.

        Returns
        -------
        dict
            Mapping of var name to slice.
        """
        if self._slices is None:
            slices = {}
            start = end = 0
            for name in self._system()._var_relevant_names[self._name][self._typ]:
                end += self._views_flat[name].size
                slices[name] = slice(start, end)
                start = end
            self._slices = slices

        return self._slices

    def __getstate__(self):
        """
        Return state as a dict.

        For pickling vectors in case recording, we want to get rid of
        the system contained within Vectors, because MPI Comm objects cannot
        be pickled using Python3's pickle module.

        Returns
        -------
        dict
            state minus system member.
        """
        state = self.__dict__.copy()
        del state['_system']
        return state
