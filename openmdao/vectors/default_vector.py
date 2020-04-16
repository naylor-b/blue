"""Define the default Vector class."""
from collections import OrderedDict
import numbers

import numpy as np
from numpy import logical_and

from openmdao.vectors.vector import Vector, INT_DTYPE, _full_slice
from openmdao.vectors.default_transfer import DefaultTransfer
from openmdao.utils.mpi import MPI, multi_proc_exception_check


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
        ncol = self._ncol
        size = np.sum(self._system()._var_sizes[self._name][self._typ][self._iproc, :])
        return np.zeros(size) if ncol == 1 else np.zeros((size, ncol))

    def _update_root_data(self, outvec):
        """
        Resize the root data if necesary (i.e., due to reconfiguration).

        Parameters
        ----------
        outvec : Vector or None
            If not None, the output vector used to share memory for nocopy inputs.
        """
        system = self._system()
        type_ = self._typ
        vec_name = self._name
        root_vec = self._root_vector

        sys_offset, size_after_sys = system._ext_sizes[vec_name][type_]
        sys_size = np.sum(system._var_sizes[vec_name][type_][self._iproc, :])
        old_sizes_total = root_vec._data.size

        root_vec._data = np.concatenate([
            root_vec._data[:sys_offset],
            np.zeros(sys_size),
            root_vec._data[old_sizes_total - size_after_sys:],
        ])

        if self._alloc_complex and root_vec._cplx_data.size != root_vec._data.size:
            root_vec._cplx_data = np.zeros(root_vec._data.size, dtype=complex)

        root_vec._initialize_views(outvec)

    def _extract_root_data(self):
        """
        Extract views of arrays from root_vector.

        Returns
        -------
        ndarray
            zeros array of correct size.
        """
        system = self._system()
        root_vec = self._root_vector

        cplx_data = None
        scaling = {}
        if self._do_scaling:
            scaling['phys'] = {}
            scaling['norm'] = {}

        sizes = system._var_sizes[self._name][self._typ]
        ind1 = system._ext_sizes[self._name][self._typ][0]
        ind2 = ind1 + np.sum(sizes[self._iproc, :])

        data = root_vec._data[ind1:ind2]

        # Extract view for complex storage too.
        if self._alloc_complex:
            cplx_data = root_vec._cplx_data[ind1:ind2]

        if self._do_scaling:
            for typ in ('phys', 'norm'):
                root_scale = root_vec._scaling[typ]
                rs0 = root_scale[0]
                if rs0 is None:
                    scaling[typ] = (rs0, root_scale[1][ind1:ind2])
                else:
                    scaling[typ] = (rs0[ind1:ind2], root_scale[1][ind1:ind2])

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
                elif self._name == 'linear' and self._typ == 'output':
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

    def _initialize_views(self, outvec):
        """
        Internally assemble views onto the vectors.

        Sets the following attributes:
        _views
        _views_flat

        Parameters
        ----------
        outvec : Vector or None
            If not None, the output vector used to share memory for nocopy inputs.
        """
        system = self._system()
        type_ = self._typ
        kind = self._kind
        iproc = self._iproc
        ncol = self._ncol
        nocopy = self.get_nocopy()

        do_scaling = self._do_scaling
        if do_scaling:
            factors = system._scale_factors
            scaling = self._scaling

        self._views = views = OrderedDict()
        self._views_flat = views_flat = OrderedDict()

        alloc_complex = self._alloc_complex
        self._cplx_views = cplx_views = OrderedDict()
        self._cplx_views_flat = cplx_views_flat = OrderedDict()

        allprocs_abs2idx_t = system._var_allprocs_abs2idx[self._name]
        sizes_t = system._var_sizes[self._name][type_]
        offs = system._get_var_offsets()[self._name][type_]
        if offs.size > 0:
            offs = offs[iproc].copy()
            # turn global offset into local offset
            start = offs[0]
            offs -= start
        else:
            offs = offs[0].copy()
        offsets_t = offs

        length = 0

        abs2meta = system._var_abs2meta
        for abs_name in system._var_relevant_names[self._name][type_]:
            idx = allprocs_abs2idx_t[abs_name]

            shape = abs2meta[abs_name]['shape']
            if ncol > 1:
                if not isinstance(shape, tuple):
                    shape = (shape,)
                shape = tuple(list(shape) + [ncol])

            ind1 = offsets_t[idx]
            ind2 = ind1 + sizes_t[iproc, idx]
            if abs_name in nocopy:  # nocopy input
                views_flat[abs_name] = v = outvec._views_flat[nocopy[abs_name]]
            else:
                views_flat[abs_name] = v = self._data[ind1:ind2]

            length += v.size

            if shape != v.shape:
                v = v.view()
                v.shape = shape
            views[abs_name] = v

            if alloc_complex:
                if abs_name in nocopy:  # nocopy input
                    cplx_views_flat[abs_name] = v = outvec._cplx_views_flat[nocopy[abs_name]]
                else:
                    cplx_views_flat[abs_name] = v = self._cplx_data[ind1:ind2]
                if shape != v.shape:
                    v = v.view()
                    v.shape = shape
                cplx_views[abs_name] = v

            if do_scaling:
                for scaleto in ('phys', 'norm'):
                    scale0, scale1 = factors[abs_name][kind, scaleto]
                    vec = scaling[scaleto]
                    if vec[0] is not None:
                        vec[0][ind1:ind2] = scale0
                    vec[1][ind1:ind2] = scale1

        self._names = frozenset(views)
        self._len = length

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

    def _sub_arr_iter(self, idxs):
        start = end = 0
        if idxs is _full_slice:
            for arr in self._views_flat.values():
                end += arr.size
                yield arr, start, end, idxs
                start = end
        # TODO: allow other slices besides _full_slice
        else:
            idxs = np.asarray(idxs)
            for arr in self._views_flat.values():
                end += arr.size
                in_arr = logical_and(start <= idxs, idxs < end)
                yield arr, start, end, idxs[in_arr] - start
                start = end

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
        self.iadd(val * vec.asarray())

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
        Fill the data array with the value at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
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

    def asarray(self, idxs=_full_slice):
        """
        Return parts of the data array at the specified indices or slice(s).

        Parameters
        ----------
        idxs : int or slice or tuple of ints and/or slices.
            The locations to pull from the data array.

        Returns
        -------
        ndarray
            Array of values.
        """
        return self._data[idxs]

    def _nocopy_asarray(self, idxs=_full_slice):
        """
        Return parts of the data array at the specified indices or slice(s).

        Parameters
        ----------
        idxs : int or slice or tuple of ints and/or slices.
            The locations to pull from the data array.

        Returns
        -------
        ndarray
            Array of values.
        """
        lst = [v for v in self._views_flat.values() if v.size > 0]
        if lst:
            return np.concatenate(lst)
        else:
            return np.zeros(0)

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
        u = self

        # The assumption is that alpha * du has been added to self (i.e., u)
        # just prior to this method being called. We are currently in the
        # initialization of a line search, and we're trying to ensure that
        # the u does not violate bounds in the first iteration. If it does,
        # we modify the du vector directly.

        # This is the required change in step size, relative to the du vector.
        d_alpha = 0

        # Find the largest amount a bound is violated
        # where positive means a bound is violated - i.e. the required d_alpha.
        mask = du._data != 0
        if mask.any():
            abs_du_mask = np.abs(du._data[mask])
            u_mask = u._data[mask]

            # Check lower bound
            max_d_alpha = np.amax((lower_bounds._data[mask] - u_mask) / abs_du_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

            # Check upper bound
            max_d_alpha = np.amax((u_mask - upper_bounds._data[mask]) / abs_du_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

        if d_alpha > 0:
            # d_alpha will not be negative because it was initialized to be 0
            # and we've only done max operations.
            # d_alpha will not be greater than alpha because the assumption is that
            # the original point was valid - i.e., no bounds were violated.
            # Therefore 0 <= d_alpha <= alpha.

            # We first update u to reflect the required change to du.
            u.add_scal_vec(-d_alpha, du)

            # At this point, we normalize d_alpha by alpha to figure out the relative
            # amount that the du vector has to be reduced, then apply the reduction.
            du *= 1 - d_alpha / alpha

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
        u = self

        # The assumption is that alpha * step has been added to this vector
        # just prior to this method being called. We are currently in the
        # initialization of a line search, and we're trying to ensure that
        # the initial step does not violate bounds. If it does, we modify
        # the step vector directly.

        # enforce bounds on step in-place.
        u_data = u._data

        # If u > lower, we're just adding zero. Otherwise, we're adding
        # the step required to get up to the lower bound.
        # For du, we normalize by alpha since du eventually gets
        # multiplied by alpha.
        change_lower = np.maximum(u_data, lower_bounds._data) - u_data

        # If u < upper, we're just adding zero. Otherwise, we're adding
        # the step required to get down to the upper bound, but normalized
        # by alpha since du eventually gets multiplied by alpha.
        change_upper = np.minimum(u_data, upper_bounds._data) - u_data

        change = change_lower + change_upper

        u_data += change
        du._data += change / alpha

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
        u = self

        # The assumption is that alpha * step has been added to this vector
        # just prior to this method being called. We are currently in the
        # initialization of a line search, and we're trying to ensure that
        # the initial step does not violate bounds. If it does, we modify
        # the step vector directly.

        # enforce bounds on step in-place.
        u_data = u._data
        du_data = du._data

        # If u > lower, we're just adding zero. Otherwise, we're adding
        # the step required to get up to the lower bound.
        # For du, we normalize by alpha since du eventually gets
        # multiplied by alpha.
        change_lower = np.maximum(u_data, lower_bounds._data) - u_data

        # If u < upper, we're just adding zero. Otherwise, we're adding
        # the step required to get down to the upper bound, but normalized
        # by alpha since du eventually gets multiplied by alpha.
        change_upper = np.minimum(u_data, upper_bounds._data) - u_data

        change = change_lower + change_upper

        u_data += change
        du_data += change / alpha

        # Now we ensure that we will backtrack along the wall during the
        # line search by setting the entries of du at the bounds to zero.
        changed_either = change.astype(bool)
        du_data[changed_either] = 0.

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
