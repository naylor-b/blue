"""
An unordered collection of variables.  Used before Vectors (ordered collections of variables) exist.
"""

from collections import OrderedDict
import numpy as np
import weakref

from openmdao.vectors.vector import _full_slice
from openmdao.utils.options_dictionary import _undefined
from openmdao.utils.name_maps import name2abs_names, abs_name2rel_name

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


class UnorderedVarCollection(object):
    """
    A class to represent the input and output nonlinear vectors prior to vector setup.

    Variables can be set/retrieved by name, but there is no full array available since
    the variables are not ordered.

    Attributes
    ----------
    _root : UnorderedVarCollection or None
        The var collection owned by the model.
    _iotype : str
        'input' or 'output'
    _system : System
        Weak ref to the owning system.
    """

    def __init__(self, system, iotype):
        """
        Initialize this collection.

        Parameters
        ----------
        system : <System>
            System that owns this variable collection.
        iotype : str
            IO type indicator.  Value is either 'input' or 'output'.
        """
        self._root = None
        self._iotype = iotype
        self._system = weakref.ref(system)

    def name2abs_name(self, name):
        """
        Map the given name to the absolute name.

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
        if system._has_var_data():
            abs_names = system._var_allprocs_prom2abs_list[self._iotype][name]
            if len(abs_names) > 1:
                raise RuntimeError(f"The promoted name '{name}'' is invalid because it refers to "
                                   f"multiple inputs: {abs_names}.")
            return abs_names[0] if abs_names else None
        else:  # setup hasn't happened yet
            try:
                if name in system._var_rel2meta:
                    return name  # use relative name since no abs name data structures exist yet
                else:
                    return None
            except AttributeError:  # system most likely a Group
                raise RuntimeError(f"{system.msginfo}: Can't access variable '{name}' before "
                                   "setup.")

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
        return self.name2abs_name(name) is not None

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
        if self._system()._has_var_data():
            abs_name = self.name2abs_name(name)
            if abs_name is None:
                raise KeyError(f"{self.msginfo}: Variable '{name}' not found.")
            return self._abs_get_val(abs_name, flat=False)
        else:
            # pre-setup, assume relative name, since absolute name keyed data
            # structures don't exist yet and promotions haven't been processed.
            try:
                return self._system()._var_rel2meta[name]['value']
            except KeyError:
                raise KeyError(f"{self.msginfo}: Variable '{name}' not found.")
            except AttributeError:
                raise ValueError(f"{self.msginfo:}: Can't access variable '{name}' before setup.")

    # def __setitem__(self, name, value):
    #     """
    #     Set the variable value.

    #     Parameters
    #     ----------
    #     name : str
    #         Relative variable name in the owning system's namespace.
    #     value : float or list or tuple or ndarray
    #         variable value to set
    #     """
    #     self._meta[name]['value'] = value

    def _abs_get_val(self, abs_name, flat=True):
        system = self._system()
        try:
            val = system._var_abs2meta[abs_name]['value']
        except KeyError:
            # could be discrete
            # TODO: change discrete to use abs names like others (or others to use rel)
            try:
                rel = abs_name2rel_name(system, abs_name)
                if self._iotype == 'output':
                    val = system._var_discrete['output'][rel]['value']
                else:
                    val = system._var_discrete['input'][rel]['value']
            except KeyError:
                raise KeyError(f"{self.msginfo}: Variable '{abs_name}' not found.")
        if flat:
            return val.ravel()
        return val

    def _abs_iter(self):
        """
        Iterate over the absolute names in the vector.
        """
        yield from self._system()._var_abs2prom[self.iotype]

    # def size_shape_iter(self):
    #     """
    #     Return tuples of the form (name, size, shape).

    #     This will be used to build Vector instances.
    #     """
    #     pass

    # def add_var(self, name, discrete=False, meta=None):
    #     """
    #     Add a variable to this collection.

    #     Parameters
    #     ----------
    #     name : str
    #         Relative name of the variable.
    #     discrete : bool
    #         If True, this variable is discrete.
    #     meta : dict
    #         Metadata for the variable.
    #     """
    #     # must add to _var2meta since var name is all we have early on
    #     self._meta[name] = meta
    #     self._meta[name]['discrete'] = discrete

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

    # def append(self, vec):
    #     """
    #     Add the variables from the given UnorderedVarCollection to this UnorderedVarCollection.

    #     Parameters
    #     ----------
    #     vec : UnorderedVarCollection
    #         UnorderedVarCollection being added.
    #     """
    #     pass

    # def set_var(self, name, val, idxs=_full_slice):
    #     """
    #     Set the value corresponding to the named variable, with optional indexing.

    #     Parameters
    #     ----------
    #     name : str
    #         The name of the variable.
    #     val : float or ndarray
    #         Scalar or array to set data array to.
    #     idxs : int or slice or tuple of ints and/or slices.
    #         The locations where the data array should be updated.
    #     """
    #     pass

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
