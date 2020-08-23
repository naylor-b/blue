import inspect
import ast
import textwrap

from collections import defaultdict


class Debugger(object):
    def __init__(self):
        self._breakpoints = defaultdict(list)

    def setup_debugging(self, problem):
        seen = set([object])
        for oname, obj in self.debuggable_obj_iter(problem):
            if obj.__class__ in seen:
                continue
            seen.add(obj.__class__)
            for fname, func in inspect.getmembers(obj, inspect.ismethod):
                ident = (func.__code__.co_filename, func.__code__.co_firstlineno)
                if ident in self._breakpoints:
                    self._breakpoints[ident].append((id(obj), oname, obj.__class__.__name__,
                                                     None, None, None))
                    continue

                srcfile, func_start = ident
                first, rets, last = self.get_func_info(func)

                self._breakpoints[ident].append((id(obj), oname, obj.__class__.__name__, 
                                                 first, rets, last))

    def get_func_info(self, func):
        visitor = _BreakpointFinder(func)
        src = textwrap.dedent(inspect.getsource(func))
        visitor.visit(ast.parse(src, mode='exec'))
        start = func.__code__.co_firstlineno
        return visitor.first_line, [r.lineno + start - 1 for r in visitor.returns], visitor.last_line

    def debuggable_obj_iter(self, problem):
        if problem._name is None:
            yield 'problem', problem
            yield 'problem.driver', problem.driver
        else:
            yield problem._name, problem
            yield f'{problem._name}.driver', problem.driver

        for s in problem.model.system_iter(include_self=True, recurse=True):
            yield s.pathname, s
            if s.nonlinear_solver is not None:
                yield f'{s.pathname}.nonlinear_solver', s.nonlinear_solver
            if s.linear_solver is not None:
                yield f'{s.pathname}.linear_solver', s.linear_solver


class _BreakpointFinder(ast.NodeVisitor):
    """
    An ast.NodeVisitor that finds method start and return lines.
    """

    def __init__(self, func):
        super(_BreakpointFinder, self).__init__()
        self.start_line = func.__code__.co_firstlineno
        self.first_line = None
        self.last_line = None
        self.returns = []
        # TODO: need stack to ensure we exclude stuff from nested functs

    def visit_Return(self, node):
        # expr will be an ast node.  we can evaluate that later without changing
        # it, but we may want to convert it back to a string for display to the user.
        self.returns.append(node)

    def visit_FunctionDef(self, node):
        # for d in node.decorator_list:
        #     self.cache[d.lineno] = qual

        self.first_line = node.body[0].lineno + self.start_line
        for bnode in node.body:
            self.visit(bnode)
        if node.body[0] is not node.body[-1] and node.body[-1] not in self.returns:
            self.last_line = node.body[-1].lineno + self.start_line


if __name__ == '__main__':
    import openmdao.api as om
    p = om.Problem()
    p.model.add_subsystem('C1', om.ExecComp('y=2*x'), promotes=['x'])
    p.model.add_subsystem('C2', om.ExecComp('y=3*x'), promotes=['x'])

    p.setup()
    p.run_model()

    d = Debugger()
    d.setup_debugging(p)
    import pprint
    pprint.pprint(d._breakpoints)

    print('-' * 50)
    for ident, lst in d._breakpoints.items():
        if len(lst) > 1:
            print(ident, len(lst))

    print("*******")

    print(d._breakpoints[('/home/bret/dev/OpenMDAO/openmdao/core/system.py', 580)])