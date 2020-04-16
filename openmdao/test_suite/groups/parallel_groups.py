"""Define `Group`s with parallel topologies for testing"""

import openmdao.api as om
import numpy as np

class FanOut(om.Group):
    """
    Topology where one comp broadcasts an output to two target
    components.
    """

    def __init__(self, size=1, **kwargs):
        super(FanOut, self).__init__(**kwargs)

        self.add_subsystem('p', om.IndepVarComp('x', np.ones(size)))
        self.add_subsystem('comp1', om.ExecComp(['y=3.0*x'], x=np.ones(size), y=np.ones(size)))
        self.add_subsystem('comp2', om.ExecComp(['y=-2.0*x'], x=np.ones(size), y=np.ones(size)))
        self.add_subsystem('comp3', om.ExecComp(['y=5.0*x'], x=np.ones(size), y=np.ones(size)))

        self.connect("p.x", "comp1.x")
        self.connect("comp1.y", "comp2.x")
        self.connect("comp1.y", "comp3.x")


class FanOutGrouped(om.Group):
    """
    Topology where one component broadcasts an output to two target
    components.
    """

    def __init__(self, size=1, **kwargs):
        super(FanOutGrouped, self).__init__(**kwargs)

        self.add_subsystem('iv', om.IndepVarComp('x', np.ones(size)))
        self.add_subsystem('c1', om.ExecComp(['y=3.0*x'], x=np.ones(size), y=np.ones(size)))

        self.sub = self.add_subsystem('sub', om.ParallelGroup())
        self.sub.add_subsystem('c2', om.ExecComp(['y=-2.0*x'], x=np.ones(size), y=np.ones(size)))
        self.sub.add_subsystem('c3', om.ExecComp(['y=5.0*x'], x=np.ones(size), y=np.ones(size)))

        self.add_subsystem('c2', om.ExecComp(['y=x'], x=np.ones(size), y=np.ones(size)))
        self.add_subsystem('c3', om.ExecComp(['y=x'], x=np.ones(size), y=np.ones(size)))

        self.connect('iv.x', 'c1.x')

        self.connect('c1.y', 'sub.c2.x')
        self.connect('c1.y', 'sub.c3.x')

        self.connect('sub.c2.y', 'c2.x')
        self.connect('sub.c3.y', 'c3.x')


class FanIn(om.Group):
    """
    Topology where two comps feed a single comp.
    """

    def __init__(self, size=1, **kwargs):
        super(FanIn, self).__init__(**kwargs)

        self.add_subsystem('p1', om.IndepVarComp('x1', np.ones(size)))
        self.add_subsystem('p2', om.IndepVarComp('x2', np.ones(size)))
        self.add_subsystem('comp1', om.ExecComp(['y=-2.0*x'], x=np.ones(size), y=np.ones(size)))
        self.add_subsystem('comp2', om.ExecComp(['y=5.0*x'], x=np.ones(size), y=np.ones(size)))
        self.add_subsystem('comp3', om.ExecComp(['y=3.0*x1+7.0*x2'], x1=np.ones(size),
                                                x2=np.ones(size), y=np.ones(size)))

        self.connect("comp1.y", "comp3.x1")
        self.connect("comp2.y", "comp3.x2")
        self.connect("p1.x1", "comp1.x")
        self.connect("p2.x2", "comp2.x")


class FanInGrouped(om.Group):
    """
    Topology where two components in a Group feed a single component
    outside of that Group.
    """

    def __init__(self, size=1, **kwargs):
        super(FanInGrouped, self).__init__(**kwargs)

        iv = self.add_subsystem('iv', om.IndepVarComp())
        iv.add_output('x1', np.ones(size))
        iv.add_output('x2', np.ones(size))
        iv.add_output('x3', np.ones(size))

        self.sub = self.add_subsystem('sub', om.ParallelGroup())
        self.sub.add_subsystem('c1', om.ExecComp(['y=-2.0*x'], x=np.ones(size), y=np.ones(size)))
        self.sub.add_subsystem('c2', om.ExecComp(['y=5.0*x'], x=np.ones(size), y=np.ones(size)))

        self.add_subsystem('c3', om.ExecComp(['y=3.0*x1+7.0*x2'], x1=np.ones(size),
                                             x2=np.ones(size), y=np.ones(size)))

        self.connect("sub.c1.y", "c3.x1")
        self.connect("sub.c2.y", "c3.x2")

        self.connect("iv.x1", "sub.c1.x")
        self.connect("iv.x2", "sub.c2.x")


class FanInGrouped2(om.Group):
    """
    Topology where two components in a Group feed a single component
    outside of that Group. This is slightly different than FanInGrouped
    in that it has two different IndepVarComps.  This configuration
    is used to test a reverse indexing MPI bug that does not appear
    when using FanInGrouped.
    """

    def __init__(self, size=1, **kwargs):
        super(FanInGrouped2, self).__init__(**kwargs)

        p1 = self.add_subsystem('p1', om.IndepVarComp('x', np.ones(size)))
        p2 = self.add_subsystem('p2', om.IndepVarComp('x', np.ones(size)))

        self.sub = self.add_subsystem('sub', om.ParallelGroup())
        self.sub.add_subsystem('c1', om.ExecComp(['y=-2.0*x'], x=np.ones(size), y=np.ones(size)))
        self.sub.add_subsystem('c2', om.ExecComp(['y=5.0*x'], x=np.ones(size), y=np.ones(size)))

        self.add_subsystem('c3', om.ExecComp(['y=3.0*x1+7.0*x2'], x1=np.ones(size),
                                             x2=np.ones(size), y=np.ones(size)))

        self.connect("sub.c1.y", "c3.x1")
        self.connect("sub.c2.y", "c3.x2")

        self.connect("p1.x", "sub.c1.x")
        self.connect("p2.x", "sub.c2.x")


class DiamondFlat(om.Group):
    """
    Topology: one - two - one.

    This one is flat."""

    def __init__(self, size=1, **kwargs):
        super(DiamondFlat, self).__init__(**kwargs)

        self.add_subsystem('iv', om.IndepVarComp('x', np.ones(size) * 2.0))

        self.add_subsystem('c1', om.ExecComp(['y1 = 2.0*x1**2',
                                              'y2 = 3.0*x1'
                                              ], x1=np.ones(size),
                                                 y1=np.ones(size),
                                                 y2=np.ones(size)))

        self.add_subsystem('c2', om.ExecComp('y1 = 0.5*x1', x1=np.ones(size), y1=np.ones(size)))
        self.add_subsystem('c3', om.ExecComp('y1 = 3.5*x1', x1=np.ones(size), y1=np.ones(size)))

        self.add_subsystem('c4', om.ExecComp(['y1 = x1 + 2.0*x2',
                                              'y2 = 3.0*x1 - 5.0*x2'
                                              ], x1=np.ones(size),
                                                 x2=np.ones(size),
                                                 y1=np.ones(size),
                                                 y2=np.ones(size)))

        # make connections
        self.connect('iv.x', 'c1.x1')
        self.connect('c1.y1', 'c2.x1')
        self.connect('c1.y2', 'c3.x1')
        self.connect('c2.y1', 'c4.x1')
        self.connect('c3.y1', 'c4.x2')


class Diamond(om.Group):
    """
    Topology: one - two - one.
    """

    def __init__(self, size=1, **kwargs):
        super(Diamond, self).__init__(**kwargs)

        self.add_subsystem('iv', om.IndepVarComp('x', np.ones(size) * 2.0))

        self.add_subsystem('c1', om.ExecComp(['y1 = 2.0*x1**2',
                                              'y2 = 3.0*x1'
                                              ], x1=np.ones(size),
                                                 y1=np.ones(size),
                                                 y2=np.ones(size)))

        sub = self.add_subsystem('sub', om.ParallelGroup())
        sub.add_subsystem('c2', om.ExecComp('y1 = 0.5*x1', x1=np.ones(size), y1=np.ones(size)))
        sub.add_subsystem('c3', om.ExecComp('y1 = 3.5*x1', x1=np.ones(size), y1=np.ones(size)))

        self.add_subsystem('c4', om.ExecComp(['y1 = x1 + 2.0*x2',
                                              'y2 = 3.0*x1 - 5.0*x2'
                                              ], x1=np.ones(size),
                                                 x2=np.ones(size),
                                                 y1=np.ones(size),
                                                 y2=np.ones(size)))

        # make connections
        self.connect('iv.x', 'c1.x1')

        self.connect('c1.y1', 'sub.c2.x1')
        self.connect('c1.y2', 'sub.c3.x1')

        self.connect('sub.c2.y1', 'c4.x1')
        self.connect('sub.c3.y1', 'c4.x2')


class ConvergeDivergeFlat(om.Group):
    """
    Topology one - two - one - two - one. This model was critical in
    testing parallel reverse scatters. This version is perfectly flat.
    """

    def __init__(self, size=1, **kwargs):
        super(ConvergeDivergeFlat, self).__init__(**kwargs)

        self.add_subsystem('iv', om.IndepVarComp('x', np.ones(size) * 2.0))

        self.add_subsystem('c1', om.ExecComp(['y1 = 2.0*x1**2',
                                              'y2 = 3.0*x1'
                                              ], x1=np.ones(size),
                                                 y1=np.ones(size),
                                                 y2=np.ones(size)))

        self.add_subsystem('c2', om.ExecComp('y1 = 0.5*x1', x1=np.ones(size), y1=np.ones(size)))
        self.add_subsystem('c3', om.ExecComp('y1 = 3.5*x1', x1=np.ones(size), y1=np.ones(size)))

        self.add_subsystem('c4', om.ExecComp(['y1 = x1 + 2.0*x2',
                                              'y2 = 3.0*x1 - 5.0*x2'
                                              ], x1=np.ones(size),
                                                 x2=np.ones(size),
                                                 y1=np.ones(size),
                                                 y2=np.ones(size)))

        self.add_subsystem('c5', om.ExecComp('y1 = 0.8*x1', x1=np.ones(size), y1=np.ones(size)))
        self.add_subsystem('c6', om.ExecComp('y1 = 0.5*x1', x1=np.ones(size), y1=np.ones(size)))
        self.add_subsystem('c7', om.ExecComp('y1 = x1 + 3.0*x2', x1=np.ones(size), x2=np.ones(size),
                                             y1=np.ones(size)))

        # make connections
        self.connect('iv.x', 'c1.x1')

        self.connect('c1.y1', 'c2.x1')
        self.connect('c1.y2', 'c3.x1')

        self.connect('c2.y1', 'c4.x1')
        self.connect('c3.y1', 'c4.x2')

        self.connect('c4.y1', 'c5.x1')
        self.connect('c4.y2', 'c6.x1')

        self.connect('c5.y1', 'c7.x1')
        self.connect('c6.y1', 'c7.x2')


class ConvergeDiverge(om.Group):
    """
    Topology: one - two - one - two - one.

    Used for testing parallel reverse scatters.
    """

    def __init__(self, size=1, **kwargs):
        super(ConvergeDiverge, self).__init__(**kwargs)

        self.add_subsystem('iv', om.IndepVarComp('x', np.ones(size) * 2.0))

        self.add_subsystem('c1', om.ExecComp(['y1 = 2.0*x1**2',
                                              'y2 = 3.0*x1'
                                              ], x1=np.ones(size),
                                                 y1=np.ones(size),
                                                 y2=np.ones(size)))

        g1 = self.add_subsystem('g1', om.ParallelGroup())
        g1.add_subsystem('c2', om.ExecComp('y1 = 0.5*x1', x1=np.ones(size), y1=np.ones(size)))
        g1.add_subsystem('c3', om.ExecComp('y1 = 3.5*x1', x1=np.ones(size), y1=np.ones(size)))

        self.add_subsystem('c4', om.ExecComp(['y1 = x1 + 2.0*x2',
                                              'y2 = 3.0*x1 - 5.0*x2'
                                              ], x1=np.ones(size),
                                                 x2=np.ones(size),
                                                 y1=np.ones(size),
                                                 y2=np.ones(size)))

        g2 = self.add_subsystem('g2', om.ParallelGroup())
        g2.add_subsystem('c5', om.ExecComp('y1 = 0.8*x1', x1=np.ones(size), y1=np.ones(size)))
        g2.add_subsystem('c6', om.ExecComp('y1 = 0.5*x1', x1=np.ones(size), y1=np.ones(size)))

        self.add_subsystem('c7', om.ExecComp('y1 = x1 + 3.0*x2', x1=np.ones(size), x2=np.ones(size),
                                             y1=np.ones(size)))

        # make connections
        self.connect('iv.x', 'c1.x1')

        self.connect('c1.y1', 'g1.c2.x1')
        self.connect('c1.y2', 'g1.c3.x1')

        self.connect('g1.c2.y1', 'c4.x1')
        self.connect('g1.c3.y1', 'c4.x2')

        self.connect('c4.y1', 'g2.c5.x1')
        self.connect('c4.y2', 'g2.c6.x1')

        self.connect('g2.c5.y1', 'c7.x1')
        self.connect('g2.c6.y1', 'c7.x2')


class ConvergeDivergeGroups(om.Group):
    """
    Topology: one - two - one - two - one.

    Used for testing parallel reverse scatters. This version contains some
    deeper nesting.
    """

    def __init__(self, size=1, **kwargs):
        super(ConvergeDivergeGroups, self).__init__(**kwargs)

        self.add_subsystem('iv', om.IndepVarComp('x', np.ones(size) * 2.0))

        g1 = self.add_subsystem('g1', om.ParallelGroup())
        g1.add_subsystem('c1', om.ExecComp(['y1 = 2.0*x1**2',
                                            'y2 = 3.0*x1'
                                            ], x1=np.ones(size),
                                                 y1=np.ones(size),
                                                 y2=np.ones(size)))

        g2 = g1.add_subsystem('g2', om.ParallelGroup())
        g2.add_subsystem('c2', om.ExecComp('y1 = 0.5*x1', x1=np.ones(size), y1=np.ones(size)))
        g2.add_subsystem('c3', om.ExecComp('y1 = 3.5*x1', x1=np.ones(size), y1=np.ones(size)))

        g1.add_subsystem('c4', om.ExecComp(['y1 = x1 + 2.0*x2',
                                            'y2 = 3.0*x1 - 5.0*x2'
                                            ], x1=np.ones(size),
                                                 x2=np.ones(size),
                                                 y1=np.ones(size),
                                                 y2=np.ones(size)))

        g3 = self.add_subsystem('g3', om.ParallelGroup())
        g3.add_subsystem('c5', om.ExecComp('y1 = 0.8*x1', x1=np.ones(size), y1=np.ones(size)))
        g3.add_subsystem('c6', om.ExecComp('y1 = 0.5*x1', x1=np.ones(size), y1=np.ones(size)))

        self.add_subsystem('c7', om.ExecComp('y1 = x1 + 3.0*x2', x1=np.ones(size), x2=np.ones(size),
                                             y1=np.ones(size)))

        # make connections
        self.connect('iv.x', 'g1.c1.x1')

        g1.connect('c1.y1', 'g2.c2.x1')
        g1.connect('c1.y2', 'g2.c3.x1')

        self.connect('g1.g2.c2.y1', 'g1.c4.x1')
        self.connect('g1.g2.c3.y1', 'g1.c4.x2')

        self.connect('g1.c4.y1', 'g3.c5.x1')
        self.connect('g1.c4.y2', 'g3.c6.x1')

        self.connect('g3.c5.y1', 'c7.x1')
        self.connect('g3.c6.y1', 'c7.x2')


class FanInSubbedIDVC(om.Group):
    """
    Classic Fan In with indepvarcomps buried below the parallel group, and a summation
    component.
    """

    def __init__(self, size=1, **kwargs):
        super(FanInSubbedIDVC, self).__init__(**kwargs)
        self.size = size

    def setup(self):
        size = self.size

        sub = self.add_subsystem('sub', om.ParallelGroup())
        sub1 = sub.add_subsystem('sub1', om.Group())
        sub2 = sub.add_subsystem('sub2', om.Group())

        sub1.add_subsystem('p1', om.IndepVarComp('x', np.ones(size) * 3.0))
        sub2.add_subsystem('p2', om.IndepVarComp('x', np.ones(size) * 5.0))
        sub1.add_subsystem('c1', om.ExecComp(['y = 2.0*x'], x=np.ones(size), y=np.ones(size)))
        sub2.add_subsystem('c2', om.ExecComp(['y = 4.0*x'], x=np.ones(size), y=np.ones(size)))
        sub1.connect('p1.x', 'c1.x')
        sub2.connect('p2.x', 'c2.x')

        self.add_subsystem('sum',om. ExecComp(['y = z1 + z2'], z1=np.ones(size), z2=np.ones(size),
                                              y=np.ones(size)))
        self.connect('sub.sub1.c1.y', 'sum.z1')
        self.connect('sub.sub2.c2.y', 'sum.z2')

        self.sub.sub1.add_design_var('p1.x')
        self.sub.sub2.add_design_var('p2.x')
        self.add_objective('sum.y', index=0)