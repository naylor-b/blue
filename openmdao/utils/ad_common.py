
import sys
import os
import traceback

import numpy as np

from openmdao.utils.ad_autograd import _get_autograd_ad_func, _get_autograd_ad_jac
from openmdao.utils.ad_tangent import _get_tangent_ad_func, _get_tangent_ad_jac
from numpy.testing import assert_almost_equal
from openmdao.core.problem import Problem
from openmdao.core.explicitcomponent import Component, ExplicitComponent
import openmdao.utils.mod_wrapper as mod_wrapper


def _ad_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao ad' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs='?', help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Output file name. By default, output goes to stdout.')
    parser.add_argument('-m', '--method', default='autograd', action='store', dest='ad_method',
                        help='AD method (autograd, tangent).')
    parser.add_argument('-c', '--class', action='append', dest='classes', default=[],
                        help='Specify component class(es) to run AD on.')


def _ad_exec(options):
    """
    Process command line args and perform postprocessing on the specified memory dump file.
    """
    if options.file:
        from openmdao.utils.om import _post_setup_exec
        options.file = [options.file]
        _post_setup_exec(options)
    else:
        _ad(None, options)


def _get_class(classpath):
    modpath, cname = classpath.rsplit('.', 1)
    import_module(modpath)
    mod = sys.modules[modpath]
    return getattr(mod, cname)


def _comp_iter(classes, prob):
    """
    """
    if all(['.' in cpath for cpath in classes]) and classes:
        insts = [_get_class(cpath)() for cpath in classes]
        for obj in insts:
            prob.model.add_subsystem(obj.__class__.__name__.lower() + '_', obj)

        prob.setup()
        prob.run_model()
        invec = prob.model._inputs._data
        invec[:] = np.random.random(invec.size)

        print("\nChecking partials:")
        check_dct = prob.check_partials(out_stream=None)
        prob.run_model()

        for obj in insts:
            print("\nClass:", obj.__class__.__name__)
            yield obj, check_dct


    else:  # find an instance of each Component class in the model
        prob.run_model()
        print("\nChecking partials:")
        check_dct = prob.check_partials(out_stream=None)
        prob.run_model()
        seen = set(('IndepVarComp', 'ExecComp'))
        for s in prob.model.system_iter(recurse=True, include_self=True, typ=Component):
            cname = s.__class__.__name__
            if cname not in seen and (cname in classes or not classes):
                seen.add(cname)
                print("\nClass:", cname)
                print("Instance:", s.pathname)
                yield s, check_dct

            if classes and (len(seen) == len(classes) + 2):
                break

        not_found = classes - seen
        if not_found:
            raise RuntimeError("Couldn't find an instance of the following classes: %s." %
                                not_found)


def _ad(prob, options):
    """
    Compute the fwd and rev AD for the compute or apply_nonlinear method of the given class.
    """
    if options.outfile is None:
        out = sys.stdout
    else:
        out = open(options.outfile, 'w')

    classes = set(options.classes)

    Problem._post_setup_func = None  # prevent infinite recursion

    if prob is None:
        prob = Problem()

    summary = {}

    for s, check_dct in _comp_iter(classes, prob):

        summary[s.__class__.__name__] = summ = {}

        rel_offset = len(s.pathname) + 1 if s.pathname else 0

        type_ = 'Explicit' if isinstance(s, ExplicitComponent) else 'Implicit'
        summ['type'] = type_
        print("Type:", type_)

        if options.ad_method == 'autograd':
            import autograd.numpy as agnp
            mod_wrapper.np = mod_wrapper.numpy = agnp

        for mode in ('fwd', 'rev'):
            summ[mode] = {}
            try:
                J = {}
                if options.ad_method == 'autograd':
                    func = _get_autograd_ad_func(s, mode)
                    _get_autograd_ad_jac(s, mode, func, J)
                elif options.ad_method == 'tangent':
                    func, deriv_mod = _get_tangent_ad_func(s, mode, verbose=1)
                    _get_tangent_ad_jac(s, mode, func, J)

                    del sys.modules[deriv_mod.__name__]
                    os.remove(deriv_mod.__file__)

                mx_diff = 0.0
                print("\n%s J:" % mode.upper())
                for key in sorted(J):
                    o, i = key
                    rel_o = o[rel_offset:]
                    rel_i = i[rel_offset:]
                    if np.any(J[key]) or (rel_o, rel_i) in check_dct[s.pathname]:
                        if (rel_o, rel_i) not in check_dct[s.pathname]:
                            check_dct[s.pathname][rel_o, rel_i] = d = {}
                            d['J_fwd'] = np.zeros(J[key].shape)
                        print("(%s, %s)" % (rel_o, rel_i), end='')
                        try:
                            assert_almost_equal(J[key], check_dct[s.pathname][rel_o, rel_i]['J_fwd'], decimal=5)
                        except:
                            max_diff = np.max(np.abs(J[key] - check_dct[s.pathname][rel_o, rel_i]['J_fwd']))
                            if max_diff > mx_diff:
                                mx_diff = max_diff
                            print("  MAX DIFF:", max_diff)
                        else:
                            print(" ok")
                summ[mode]['diff'] = mx_diff
                summ[mode]['ran'] = True
                print()
            except:
                traceback.print_exc(file=sys.stdout)
                summ[mode]['ran'] = False
                summ[mode]['diff'] = float('nan')
                print("\n")
            finally:
                if options.ad_method == 'autograd':
                    mod_wrapper.np = mod_wrapper.numpy = np

    max_cname = max(len(s) for s in summary) + 2
    max_diff = 16
    bothgood = []
    fwdgood = []
    revgood = []
    bad = []

    toptemplate = "{cname:<{cwidth}}{typ:<10}{fdiff:<{dwidth}}{rdiff:<{dwidth}}"
    template = "{cname:<{cwidth}}{typ:<10}{fdiff:<{dwidth}.4}{rdiff:<{dwidth}.4}"
    print(toptemplate.format(cname='Class', typ='Type', fdiff='Max Diff (fwd)', rdiff='Max Diff (rev)', cwidth=max_cname, dwidth=max_diff))
    print('--------- both derivs ok ------------')
    for cname in sorted(summary):
        s = summary[cname]
        typ = s['type']
        fwdran = s['fwd']['ran']
        fwdmax = s['fwd']['diff']
        revran = s['rev']['ran']
        revmax = s['rev']['diff']
        line = template.format(cname=cname, typ=typ, fdiff=fwdmax, rdiff=revmax, cwidth=max_cname, dwidth=max_diff)
        if fwdran and revran and fwdmax == 0.0 and revmax == 0.0:
            bothgood.append(line)
            print(line)
        elif fwdran and fwdmax == 0.0:
            fwdgood.append(line)
        elif revran and revmax == 0.0:
            revgood.append(line)
        else:
            bad.append(line)

    if fwdgood:
        print('--------- fwd derivs ok ------------')
        for b in fwdgood:
            print(b)

    if revgood:
        print('--------- rev derivs ok ------------')
        for b in revgood:
            print(b)

    if bad:
        print('--------- both derivs bad ------------')
        for b in bad:
            print(b)

    print('\nSummary:  %d both good,  %d fwd good,  %d rev good' % (len(bothgood), len(fwdgood),
                                                                    len(revgood)))
    exit()


def _ad_cmd(options):
    """
    Return the post_setup hook function for 'openmdao ad'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    return lambda prob: _ad(prob, options)
