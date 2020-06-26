"""
An unordered collection of variables.  Used before Vectors (ordered collections of variables) exist.
"""

import numpy as np

from openmdao.vectors.vector import _full_slice
from openmdao.utils.options_dictionary import _undefined

# What is a vector?
#  ORDER INDEPENDENT
#   - provides a mapping from var name to var value (shaped)
#
#  ORDER DEPENDENT
#   - provides a mapping from var name to a flat array view/slice/range
#   - provides access to a flat array representing all local variables found in a given system
#   - provides access to a virtual array representing all variables (local and remote) found in a
#     given system
#        - can obtain this using offset/size of first var and last var since vars are ordered.


# Where are vector vars accessed by var name vs. by full array?
#  - in components - by (relative=promoted) var name
#  - at top level (from problem get/set) by abs var name
#  - problem.compute_jacvec_product by prom var name AND full array
#  - problem.check_partials
#  - matvec_context sets internal set of allowed names
#  - system get_val - by prom name
#  - system._abs_get_val
#  - retrieve_data_of_kind
#  - solvers access by full array


_DEFAULT_META_ = {
    'size': 0,
    'shape': None,
    'value': _undefined,
    'discrete': False,
    'distributed': False,
}


class UnorderedVarCollection(object):
    """
    A class to represent the input and output nonlinear vectors prior to vector setup.

    Variables can be set/retrieved by name, but there is no full array available since
    the variables are not ordered.

    Attributes
    ----------
    pathname : str
        The pathname of the system containing the variable collection.
    _meta : dict
        Metadata dictionary.
    _root : UnorderedVarCollection or None
        THe var collection owned by the model.
    """

    def __init__(self):
        """
        Initialize this collection.
        """
        self.pathname = ''
        self._meta = {}
        self._prom = {}
        self._root = None

    def __contains__(self, name):
        """
        Check if the variable is found in this collection.

        Parameters
        ----------
        name : str
            Relative variable name in the owning system's namespace.

        Returns
        -------
        boolean
            True or False.
        """
        return name in self._meta

    def __getitem__(self, name):
        """
        Get the variable value.

        Parameters
        ----------
        name : str
            Relative variable name in the owning system's namespace.

        Returns
        -------
        float or ndarray
            variable value.
        """
        try:
            return self._meta[name]['value']
        except KeyError:
            raise KeyError(f"{self.msginfo}: Variable '{name} not found.")

    def __setitem__(self, name, value):
        """
        Set the variable value.

        Parameters
        ----------
        name : str
            Relative variable name in the owning system's namespace.
        value : float or list or tuple or ndarray
            variable value to set
        """
        self._meta[name]['value'] = value

    def set_pathname(self, pathname):
        self.pathname = pathname

    def _abs_get_val(self, abs_name, flat=True):
        try:
            sname, vname = abs_name.rsplit('.', 1)
            if sname == self.pathname:
                if flat:
                    meta = self._meta[vname]
                    val = meta['value']
                    if meta['discrete'] or np.isscalar(val):
                        return val
                    else:
                        return val.ravel()
                else:
                    return self._meta[vname]['value']
        except (KeyError, ValueError, IndexError) as err:
            raise type(err)(f"{self.msginfo}: Variable '{abs_name} not found.")

    def _abs_iter(self):
        """
        Iterate over the absolute names in the vector.
        """
        if self.pathname:
            path = self.pathname + '.'
            for name in self._meta:
                yield path + name
        else:
            for name in self._meta:
                yield name

    def size_shape_iter(self):
        """
        Return tuples of the form (name, size, shape).

        This will be used to build Vector instances.
        """
        pass

    def get_meta(self, name, meta_name=None):
        """
        Return misc metadata for the named variable.

        Parameters
        ----------
        name : str
            Relative name of the variable.
        meta_name : str
            Name of the metadata entry.

        Returns
        -------
        object or dict
            Either a variable's metadata dict or a specific metadata value.
        """
        if meta_name is None:
            return self._meta[name]
        return self._meta[name][meta_name]

    def add_var(self, name, discrete=False, **kwargs):
        """
        Add a variable to this collection.

        Parameters
        ----------
        name : str
            Relative name of the variable.
        discrete : bool
            If True, this variable is discrete.
        **kwargs : dict
            Metadata for the variable.
        """
        # must add to _var2meta since var name is all we have early on
        self._meta[name] = _DEFAULT_META_.copy()
        self._meta[name].update(kwargs)
        self._meta[name]['discrete'] = discrete

    # def reshape_var(self, name, shape):
    #     meta = self._meta[name]
    #     meta['shape'] = shape
    #     val = meta['value']
    #     if val is None:
    #         meta['value'] = np.ones(shape)
    #     else:
    #         val = np.asarray(val)
    #         if val.shape != shape:
    #             if val.size == np.prod(shape):
    #                 meta['value'] = val.reshape(shape)
    #             else:
    #                 meta['value'] = np.ones(shape)

    def append(self, vec):
        """
        Add the variables from the given UnorderedVarCollection to this UnorderedVarCollection.

        Parameters
        ----------
        vec : UnorderedVarCollection
            UnorderedVarCollection being added.
        """
        pass

    def set_var(self, name, val, idxs=_full_slice):
        """
        Set the value corresponding to the named variable, with optional indexing.

        Parameters
        ----------
        name : str
            The name of the variable.
        val : float or ndarray
            Scalar or array to set data array to.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        pass

    @property
    def msginfo(self):
        """
        Our instance pathname, if available, or our class name.  For use in error messages.

        Returns
        -------
        str
            Either our instance pathname or class name.
        """
        if self.pathname == '':
            return f"(<model>)"
        if self.pathname is not None:
            return f"{self.pathname}"
        return f"{type(self).__name__}"

    def asarray(self):
        """
        Raise error indicating that ordering of variables has not yet taken place.
        """
        raise RuntimeError(f"{self.msginfo}: asarray is not allowed yet because variables are "
                           "still unordered. Variables will be ordered later in the setup process.")
