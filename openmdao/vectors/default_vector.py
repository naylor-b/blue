"""Define the default Vector class."""
from copy import deepcopy
import numbers

import numpy as np

from openmdao.vectors.vector import Vector, INT_DTYPE
from openmdao.vectors.default_transfer import DefaultTransfer
from openmdao.utils.mpi import MPI, multi_proc_exception_check


class DefaultVector(Vector):
    """
    Default NumPy vector.
    """

    TRANSFER = DefaultTransfer

    def _update_root_data(self):
        """
        Resize the root data if necesary (i.e., due to reconfiguration).
        """
        type_ = self._typ
        root_vec = self._root_vector
        abs_names = self._system()._var_relevant_names[self._name][type_]

        if abs_names:
            start, stop = self.get_root_range()
            sys_size = stop - start
            size_after_sys = root_vec._data.size - stop
        else:
            start = size_after_sys = sys_size = 0

        old_sizes_total = root_vec._data.size

        root_vec._data = np.concatenate([
            root_vec._data[:start],
            np.zeros(sys_size),
            root_vec._data[old_sizes_total - size_after_sys:],
        ])

        if self._alloc_complex and root_vec._cplx_data.size != root_vec._data.size:
            root_vec._cplx_data = np.zeros(root_vec._data.size, dtype=complex)

        root_vec._init_scaling()

    def _initialize_data(self, root_vector):
        """
        Internally allocate data array.

        Parameters
        ----------
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        if root_vector is None:  # we're the root
            system = self._system()
            abs_names = system._var_relevant_names[self._name][self._typ]
            if abs_names:
                vmap, size = system.get_var_range_info(self._name, self._typ, self._ncol)
                self._data = np.zeros(size)
            else:
                self._data = np.zeros(0)
                size = 0

            if self._ncol > 1:
                self._data = self._data.reshape((size // self._ncol, self._ncol))

            # Allocate imaginary for complex step
            if self._alloc_complex:
                self._cplx_data = np.zeros(self._data.shape, dtype=np.complex)

            if self._do_scaling:
                self._scaling = {}
                data = self._data
                if self._name == 'nonlinear':
                    self._scaling['phys'] = (np.zeros(size), np.ones(data.size))
                    self._scaling['norm'] = (np.zeros(size), np.ones(data.size))
                elif self._name == 'linear':
                    # reuse the nonlinear scaling vecs since they're the same as ours
                    nlvec = system._root_vecs[self._kind]['nonlinear']
                    self._scaling['phys'] = (None, nlvec._scaling['phys'][1])
                    self._scaling['norm'] = (None, nlvec._scaling['norm'][1])
                else:
                    self._scaling['phys'] = (None, np.ones(size))
                    self._scaling['norm'] = (None, np.ones(size))

                self._init_scaling()
            else:
                self._scaling = None

        else:
            self._get_offset_view()

    def _init_scaling(self):
        """
        Internally assemble views onto the vectors.
        """
        if self._do_scaling:
            kind = self._kind
            factors = self._system()._scale_factors
            scaling = self._scaling
            slcdict, _, _ = self._get_offset_view()

            for abs_name, (start, stop, _) in slcdict.items():
                for scaleto in ('phys', 'norm'):
                    scale0, scale1 = factors[abs_name][kind, scaleto]
                    vec0, vec1 = scaling[scaleto]
                    if vec0 is not None:
                        vec0[start:stop] = scale0
                    vec1[start:stop] = scale1

    def add_at_indices(self, idxs, value):
        """
        Add the given value to the vector at the specified indices.

        Parameters
        ----------
        idxs : ndarray
            Index array.
        value : float or ndarray
            The value being added.
        """
        self._data[idxs] += value

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
            self._data += vec._data
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
            self._data -= vec._data
        else:
            self._data -= vec
        return self

    def __imul__(self, val):
        """
        Perform in-place scalar multiplication.

        Parameters
        ----------
        val : int or float
            scalar to multiply self.

        Returns
        -------
        <Vector>
            self * val
        """
        self._data *= val
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
        self._data += val * vec._data

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        self._data[:] = vec._data

    def set_val(self, val):
        """
        Set the value of this vector to a value.

        Parameters
        ----------
        val : int or float
            scalar to set self to.
        """
        self._data[:] = val

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
