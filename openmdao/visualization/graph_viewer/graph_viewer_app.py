
import os
import pstats
import sys
import traceback
import time
import webbrowser
import fnmatch
import threading
from six import StringIO
import tornado.ioloop
import tornado.web
import subprocess
from networkx.drawing.nx_pydot import write_dot
from graphviz import Digraph

from openmdao.utils.hooks import _register_hook
from openmdao.core.group import Group
from openmdao.utils.mpi import MPI


def launch_browser(port):
    time.sleep(1)
    for browser in ['chrome', 'firefox', 'chromium', 'safari']:
        try:
            webbrowser.get(browser).open('http://localhost:%s' % port)
        except:
            pass
        else:
            break


def startThread(fn):
    thread = threading.Thread(target=fn)
    thread.setDaemon(True)
    thread.start()
    return thread


class Application(tornado.web.Application):
    """
    An application that allows a user to view the system graph for any group in a model.
    """
    def __init__(self, problem, port, engine='dot'):
        """
        Inialize the app.

        Parameters
        ----------
        problem : Problem
            The Problem containing the model.
        port : int
            The server port.
        engine : str
            The graphviz layout engine to use.  Should be one of ['dot', 'fdp', 'circo'].
        """
        handlers = [
            (r"/", Index),
            (r"/sysgraph/", Index),
            (r"/sysgraph/([YN])/([_a-zA-Z][_a-zA-Z0-9.]*)", SysGraph),
        ]

        settings = dict(
             template_path=os.path.join(os.path.dirname(__file__), "templates"),
             static_path=os.path.join(os.path.dirname(__file__), "static"),
        )

        super(Application, self).__init__(handlers, **settings)

        self.prob = problem
        self.port = port
        self.engine = engine


def _rel_name(basename, name, delim='/'):
    """
    Return the name relative to basename with each parent level denoted by ..{delim}

    Parameters
    ----------
    basename : str
        Name return string will be relative to.
    name : str
        Name to convert to relative form
    delim : str
        Delimiter for each level of relative pathname.
    """
    baseparts = basename.split('.')
    parts = name.split('.')
    partslen = len(parts)
    common = []
    for i, part in enumerate(baseparts):
        if partslen > i and parts[i] == part:
            common.append(part)
        else:
            break

    if common:
        lst = ['..' + delim] * (len(baseparts) - len(common))
        lst.append('.'.join(parts[len(common):]))
        return ''.join(lst)

    return name



def get_graph_info(prob, group, engine='dot', show_outside=False):
    """
    Get the system graph from the give group and return the graphviz generated SVG string.

    Parameters
    ----------
    prob : Problem
        The Problem containing the Group.
    group : Group
        Retrieve the graph SVG for this Group.
    engine : str
        The graphviz layout engine to use to layout the graph. Should be one of
        ['dot', 'fdp', 'circo'].
    show_outside : bool
        If True, show connections from outside the current group.

    Returns
    -------
    str
        The SVG string describing the graph.
    """
    graph = group.compute_sys_graph()
    title = group.pathname if group.pathname else 'Model'
    parent = None if not group.pathname else group.pathname.rsplit('.', 1)[0]
    if parent == group.pathname:
        parent = None

    g = Digraph(filename=title + '.gv', format='svg', engine=engine)
    g.attr(rankdir='LR', size='600, 600', overlap='false')

    # groups
    with g.subgraph(name='cluster_0') as c:
        c.node_attr.update(style='filled', color='lightblue', shape='rectangle')
        c.attr(label=group.pathname)

        groupset = set(group._subgroups_myproc)
        for grp in groupset:
            c.node(grp.pathname, label=grp.name)

    # components
    with g.subgraph(name='cluster_0') as c:
        c.node_attr.update(color='orange', shape='ellipse')
        for s in group._subsystems_myproc:
            if s not in groupset:
                c.node(s.pathname, label=s.name)
        
        for u, v in graph.edges():
            src = group.pathname + '.' + u if group.pathname else u
            tgt = group.pathname + '.' + v if group.pathname else v
            c.edge(src, tgt)

    # connections from outside the group
    model = prob.model
    if group is not model and show_outside == 'Y':
        out_nodes = set()
        g.attr('node', color='lightgrey', style='filled')
        g.attr('edge', style='dashed')
        gname = group.pathname + '.'
        pname = '.'.join(group.pathname.split('.')[:-1]) + '.'
        plen = len(group.pathname.split('.')) + 1 if group.pathname else 1
        conn_set = set()
        out_depth = len(group.pathname.split('.'))
        for tgt, src in model._conn_global_abs_in2out.items():
            # show connections coming into the group
            if tgt.startswith(gname) and not src.startswith(gname):
                srcabs = '.'.join(src.split('.')[:-1])
                ssys = _rel_name(group.pathname, srcabs)
                if len(ssys) >= len(srcabs):
                    ssys = srcabs
                if ssys not in out_nodes:
                    out_nodes.add(ssys)
                    g.node(srcabs, label=ssys)
                edge = (srcabs, '.'.join(tgt.split('.')[:plen]))
                if edge not in conn_set:
                    conn_set.add(edge)
                    g.edge(*edge)
            # show connections leaving the group
            elif src.startswith(gname) and not tgt.startswith(gname):
                tgtabs = '.'.join(tgt.split('.')[:-1])
                tsys = _rel_name(group.pathname, tgtabs)
                if len(tsys) >= len(tgtabs):
                    tsys = tgtabs
                if tsys not in out_nodes:
                    out_nodes.add(tsys)
                    g.node(tgtabs, label=tsys)
                edge = ('.'.join(src.split('.')[:plen]), tgtabs)
                if edge not in conn_set:
                    conn_set.add(edge)
                    g.edge(*edge)

    svg = g.pipe()

    svg = str(svg.replace(b'\n', b''))
    return svg[1:].strip("'"), group._subgroups_myproc


class SysGraph(tornado.web.RequestHandler):
    def get(self, show_outside='N', pathname=''):
        self.write_graph(show_outside, pathname)

    def write_graph(self, show_outside, pathname):
        app = self.application
        model = app.prob.model

        if pathname:
            system = model._get_subsystem(pathname)
        else:
            system = model

        if not isinstance(system, Group):
            self.write("Components don't have graphs.")
            return
        
        svg, subgroups = get_graph_info(app.prob, system, app.engine, show_outside)
        # print("\n\n\n\n")
        # print(svg)
        pathname = system.pathname
        parent_link = ['/sysgraph']
        if show_outside:
            parent_link.append('Y')
        else:
            parent_link.append('N')
        pth = '.'.join(pathname.split('.')[:-1])
        if pth:
            parent_link.append(pth)
        parent_link = '/'.join(parent_link)

        subgroups = [g.name for g in subgroups]

        self.write("""\
    <html>
    <head>
    <style>
    </style>
    <script src="https://d3js.org/d3.v4.min.js"></script>
    <script>

    var width = 960,
        height = 800;

    var pathnames = %s;
    var subgroups = %s;
    var show_outside = '%s';

    function d3_setup() {
        var svg = d3.select("svg");
        // register click event handler on all the graphviz SVG node elements
        svg.selectAll(".node")
            .on("click", function(d, i) {
                var txt = d3.select(this).select("text").text();
                if (subgroups.includes(txt)) {
                    var ptext = pathnames[0] + "." + txt;
                    if (ptext.startsWith(".")) {
                        ptext = txt;
                    }
                    window.location = "/sysgraph/" + show_outside + "/" + ptext;
                }
            });

        window.onresize = function() {
            width = window.innerWidth * .98;
            d3.select("svg").attr("width", width);
        }
    }

    function toggle_outside()
    {
        if (show_outside == "Y") {
            location.href = "/sysgraph/N/" + pathnames[0]
        }
        else {
            location.href = "/sysgraph/Y/" + pathnames[0]
        }
    }

    window.onload = function() {
        width = window.innerWidth * .98;
        svg = d3.select("svg")
            .attr("width", width)
            .attr("height", height);
        d3_setup();
    };

    </script>
    </head>
    <body>
        <input type="button" onclick="location.href='/';" value="Home" />
        <input type="button" onclick="location.href='%s';" value="Up" />
        <input type="checkbox" onclick="toggle_outside();"  value="N"> Show Outside Connections <br>
    %s
    </body>
    </html>
    """ % ([pathname], subgroups, show_outside, parent_link, svg))


class Index(SysGraph):
    def get(self):
        self.write_graph('N', '')


def _view_graphs_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao view_graphs' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('-p', '--port', action='store', dest='port',
                        default=8009, type=int,
                        help='port used for web server')
    parser.add_argument('--problem', action='store', dest='problem', help='Problem name')
    parser.add_argument('-g', '--group', action='append', default=[], dest='groups',
                        help='Display the graph for the given group.')
    parser.add_argument('-e', '--engine', action='store', dest='engine',
                        default='dot', help='Specify graph layout engine (dot, fdp)')
    parser.add_argument('file', metavar='file', nargs=1,
                        help='profile file to view.')


def _view_graphs_cmd(options):
    """
    Return the post_setup hook function for 'openmdao graphs'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    if options.engine not in {'dot', 'circo', 'fdp', 'neato'}:
        raise RuntimeError("Graph layout engine '{}' not supported.".format(options.engine))

    def _view_graphs(prob):
        if not MPI or MPI.COMM_WORLD.rank == 0:
            view_graphs(prob, progname=options.file[0], port=options.port, 
                        groups=options.groups, engine=options.engine)
        exit()

    # register the hook
    _register_hook('final_setup', class_name='Problem', inst_id=options.problem, post=_view_graphs)

    return _view_graphs


def view_graphs(prob, progname, port=8009, groups=(), engine='dot'):
    """
    Start an interactive graph viewer for an OpenMDAO model.

    Parameters
    ----------
    prob : Problem
        The Problem to be viewed.
    progname: str
        Name of model file.
    port: int
        Port number used by web server.
    """
    app = Application(prob, port, engine)
    app.listen(port)

    print("starting server on port %d" % port)

    serve_thread  = startThread(tornado.ioloop.IOLoop.current().start)
    launch_thread = startThread(lambda: launch_browser(port))

    while serve_thread.isAlive():
        serve_thread.join(timeout=1)

if __name__ == '__main__':
    cmd_view_graphs()
