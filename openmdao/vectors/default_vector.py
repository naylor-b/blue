"""Define the default Vector class."""
from copy import deepcopy
import numbers

import numpy as np

from openmdao.core.constants import INT_DTYPE
from openmdao.vectors.vector import Vector, _full_slice
from openmdao.vectors.default_transfer import DefaultTransfer
from openmdao.utils.mpi import MPI, multi_proc_exception_check
from openmdao.utils.general_utils import _slice_indices


class DefaultVector(Vector):
    """
    Default NumPy vector.
    """

    TRANSFER = DefaultTransfer

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
        io = self._typ
        root_vec = self._root_vector
        system = root_vec._system()
        mynames = self._system()._var_relevant_names[self._name][io]

        if mynames:
            abs2idx = system._var_allprocs_abs2idx[self._name]
            sizes = system._var_sizes[self._name][io][system.comm.rank]
            istart = abs2idx[mynames[0]]
            start = np.sum(sizes[:istart])
            stop = start + np.sum(sizes[istart:abs2idx[mynames[-1]] + 1])
            myslice = slice(start, stop)
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
        io = self._typ
        kind = self._kind
        ncol = self._ncol
        isinp = io == 'input'
        if isinp:
            root_sys = self._root_vector._system()
            root_out_vec = root_sys._vectors['output'][self._name]
            conns = root_sys._conn_global_abs_in2out

        do_scaling = self._do_scaling
        if do_scaling:
            factors = system._scale_factors
            scaling = self._scaling

        self._views = views = {}
        self._views_flat = views_flat = {}

        alloc_complex = self._alloc_complex
        self._cplx_views = cplx_views = {}
        self._cplx_views_flat = cplx_views_flat = {}

        abs2meta = system._var_abs2meta[io]
        start = end = 0
        for abs_name in system._var_relevant_names[self._name][io]:
            meta = abs2meta[abs_name]
            shape = meta['shape']

            if ncol > 1:
                if not isinstance(shape, tuple):
                    shape = (shape,)
                shape = tuple(list(shape) + [ncol])

            if isinp and meta['shared']:
                src = conns[abs_name]
                views_flat[abs_name] = v = root_out_vec._views_flat[src]
                views[abs_name] = root_out_vec._views[src].view()
                views[abs_name].shape = shape
                if alloc_complex:
                    cplx_views_flat[abs_name] = root_out_vec._cplx_views_flat[src]
                    cplx_views[abs_name] = root_out_vec._cplx_views[src]
                self._len += v.size
                continue

            end = start + meta['size']

            views_flat[abs_name] = v = self._data[start:end]
            self._len += v.size
            self._data_len += v.size
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

    def _in_matvec_context(self):
        """
        Return True if this vector is inside of a matvec_context.

        Returns
        -------
        bool
            Whether or not this vector is in a matvec_context.
        """
        return len(self._names) != len(self._views)

    def _sub_arr_iter(self, idxs):
        start = end = 0
        if idxs is _full_slice:
            for arr in self._views_flat.values():
                end += arr.size
                yield self._get_arr(arr), start, end, idxs
                start = end
            return

        if isinstance(idxs, slice):
            # TODO: more efficiently support other slices
            sz = len(self)
            idxs = _slice_indices(idxs, sz, (sz,))
        else:
            idxs = np.asarray(idxs)

        for arr in self._views_flat.values():
            end += arr.size
            in_arr = np.logical_and(start <= idxs, idxs < end)
            inds = idxs[in_arr]
            if inds.size > 0:
                yield self._get_arr(arr), start, end, inds - start
            start = end

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        self.set_val(vec.asarray())

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

    def _nocopy_set_val(self, val, idxs=_full_slice):
        """
        Fill the data array with the value at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        if np.isscalar(val):
            for arr, start, end, in_arr in self._sub_arr_iter(idxs):
                arr[in_arr] = val
        else:
            for arr, start, end, in_arr in self._sub_arr_iter(idxs):
                arr[in_arr] = val[start:end]

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

    def asarray(self, copy=False, idxs=_full_slice):
        """
        Return a flat array representation of this vector, including shared memory parts.

        If copy is True, return a copy.  Otherwise, try to avoid it.

        Parameters
        ----------
        copy : bool
            If True, return a copy of the array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations to pull from the data array.

        Returns
        -------
        ndarray
            Array representation of this vector.
        """
        if copy and (idxs is _full_slice or isinstance(idxs, slice)):
            return self._data[idxs].copy()

        return self._data[idxs]  # idxs is not a slice, so we'll get a copy regardless

    def _nocopy_asarray(self, idxs=_full_slice, copy=False):
        """
        Return a flat array representation of this vector, including shared memory parts.

        If copy is True, return a copy.  Otherwise, try to avoid it.

        Parameters
        ----------
        idxs : int or slice or tuple of ints and/or slices.
            The locations to pull from the data array.
        copy : bool
            If True, return a copy of the array.

        Returns
        -------
        ndarray
            Array representation of this vector.
        """
        # we have shared memory parts, so must assemble a new array
        arr = self._get_arr(np.empty(self._len, dtype=self._data.dtype))
        start = end = 0
        for dat in self._views_flat.values():
            end += dat.size
            arr[start:end] = dat.flat
            start = end
        return arr[idxs]

    def iscomplex(self):
        """
        Return True if this vector contains complex values.

        This checks the type of the values, not whether they have a nonzero imaginary part.

        Returns
        -------
        bool
            True if this vector contains complex values.
        """
        return np.iscomplexobj(self._data)

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

    def _nocopy_iadd(self, val, idxs=_full_slice):
        """
        Add the value to the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        if np.isscalar(val):
            for arr, start, end, in_arr in self._sub_arr_iter(idxs):
                arr[in_arr] += val
        else:
            for arr, start, end, in_arr in self._sub_arr_iter(idxs):
                arr[in_arr] += val[start:end]

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

    def _nocopy_isub(self, val, idxs=_full_slice):
        """
        Subtract the value from the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        if np.isscalar(val):
            for arr, start, end, in_arr in self._sub_arr_iter(idxs):
                arr[in_arr] -= val
        else:
            for arr, start, end, in_arr in self._sub_arr_iter(idxs):
                arr[in_arr] -= val[start:end]

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

    def _nocopy_imul(self, val, idxs=_full_slice):
        """
        Multiply the value to the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        if np.isscalar(val):
            for arr, start, end, in_arr in self._sub_arr_iter(idxs):
                arr[in_arr] *= val
        else:
            for arr, start, end, in_arr in self._sub_arr_iter(idxs):
                arr[in_arr] *= val[start:end]

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
        return np.dot(self.asarray(), vec.asarray())

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        return np.linalg.norm(self.asarray())

    def get_slice_dict(self):
        """
        Return a dict of var names mapped to their slice in the local array.

        The slice indices reflect the position within the array returned from self.asarray(),
        which may include input variables that share memory with their connected outputs.

        Returns
        -------
        dict
            Mapping of var name to slice.
        """
        if self._slices is None:
            slices = {}
            start = end = 0
            for name, arr in self._views_flat.items():
                end += arr.size
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
