
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
    prefix = '' if not group.pathname else group.pathname + '.'

    nodes = [{'name': n, 'pathname': prefix + n} for n in graph.nodes()]
    idmap = {d['name']:i for i, d in enumerate(nodes)}

    links = [{'source': idmap[u], 'target': idmap[v]} for u, v in graph.edges()]

    # g = Digraph(filename=title + '.gv', format='svg', engine=engine)
    # g.attr(rankdir='LR', size='600, 600', overlap='false')

    # # groups
    # with g.subgraph(name='cluster_0') as c:
    #     c.node_attr.update(style='filled', color='lightblue', shape='rectangle')
    #     c.attr(label=group.pathname)

    #     groupset = set(group._subgroups_myproc)
    #     for grp in groupset:
    #         c.node(grp.pathname, label=grp.name)

    # # components
    # with g.subgraph(name='cluster_0') as c:
    #     c.node_attr.update(color='orange', shape='ellipse')
    #     for s in group._subsystems_myproc:
    #         if s not in groupset:
    #             c.node(s.pathname, label=s.name)
        
    #     for u, v in graph.edges():
    #         src = group.pathname + '.' + u if group.pathname else u
    #         tgt = group.pathname + '.' + v if group.pathname else v
    #         c.edge(src, tgt)

    # # connections from outside the group
    # model = prob.model
    # if group is not model and show_outside == 'Y':
    #     out_nodes = set()
    #     g.attr('node', color='lightgrey', style='filled')
    #     g.attr('edge', style='dashed')
    #     gname = group.pathname + '.'
    #     pname = '.'.join(group.pathname.split('.')[:-1]) + '.'
    #     plen = len(group.pathname.split('.')) + 1 if group.pathname else 1
    #     conn_set = set()
    #     out_depth = len(group.pathname.split('.'))
    #     for tgt, src in model._conn_global_abs_in2out.items():
    #         # show connections coming into the group
    #         if tgt.startswith(gname) and not src.startswith(gname):
    #             srcabs = '.'.join(src.split('.')[:-1])
    #             ssys = _rel_name(group.pathname, srcabs)
    #             if len(ssys) >= len(srcabs):
    #                 ssys = srcabs
    #             if ssys not in out_nodes:
    #                 out_nodes.add(ssys)
    #                 g.node(srcabs, label=ssys)
    #             edge = (srcabs, '.'.join(tgt.split('.')[:plen]))
    #             if edge not in conn_set:
    #                 conn_set.add(edge)
    #                 g.edge(*edge)
    #         # show connections leaving the group
    #         elif src.startswith(gname) and not tgt.startswith(gname):
    #             tgtabs = '.'.join(tgt.split('.')[:-1])
    #             tsys = _rel_name(group.pathname, tgtabs)
    #             if len(tsys) >= len(tgtabs):
    #                 tsys = tgtabs
    #             if tsys not in out_nodes:
    #                 out_nodes.add(tsys)
    #                 g.node(tgtabs, label=tsys)
    #             edge = ('.'.join(src.split('.')[:plen]), tgtabs)
    #             if edge not in conn_set:
    #                 conn_set.add(edge)
    #                 g.edge(*edge)

    return nodes, links, group._subgroups_myproc


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
        
        nodes, links, subgroups = get_graph_info(app.prob, system, app.engine, show_outside)

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
        line {
            stroke: #ccc;
        }
        text {
            text-anchor: middle;
            font-family: "Helvetica Neue", Helvetica, sans-serif;
            fill: #666;
            font-size: 16px;
        }
        circle {
            fill: lightsteelblue;
            stroke: steelblue;
            stroke-width: 1.5px;
        }

        .tooltip {
            position: absolute;
            background-color: white;
            border: solid;
            border-width: 2px;
            border-radius: 5px;
            padding: 5px;
            opacity: 0;
            pointer-events: none;
        }

    </style>
    <script src="https://d3js.org/d3.v4.min.js"></script>
    <script>

    var width = 960,
        height = 800;

    var pathnames = %(pathnames)s;
    var subgroups = %(subgroups)s;
    var show_outside = '%(show_outside)s';

    var nodes = %(nodes)s;

    var links = %(links)s;

    function d3_setup() {

        var svg = d3.select("svg");

        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 13)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 13)
            .attr('markerHeight', 13)
            .attr('xoverflow', 'visible')
            .append('svg:path')
            .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
            .attr('fill', '#999')
            .style('stroke','none');

        // create a tooltip
        var tooltip = d3.select("body")
            .append("div")
            .attr("class", "tooltip");

        var link = svg.append("g")
            .attr("class", "link")
            .selectAll("line");

        link = link
            .data(links)
            .enter()
            .append("line")
            .attr('marker-end','url(#arrowhead)')
            .attr("x1", function(d) { return d.source.x; })
            .attr("y1", function(d) { return d.source.y; })
            .attr("x2", function(d) { return d.target.x; })
            .attr("y2", function(d) { return d.target.y; });

        var node = svg.append("g")
            .attr("class", "node")
            .selectAll("circle")
            .data(nodes)
            .enter()
            .append("circle")
            .attr("r", 10)
            .attr("cx", function(d) { return d.x; })
            .attr("cy", function(d) { return d.y; })
            .call(d3.drag()
                .subject(function (d) { return d; })
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("click", function(d) {
                if (subgroups.includes(d.name)) {
                    var ptext = pathnames[0] + "." + d.name;
                    if (ptext.startsWith(".")) {
                        ptext = d.name;
                    }
                    window.location = "/sysgraph/" + show_outside + "/" + ptext;
                }
            });

        var simulation = d3.forceSimulation(nodes)
            .force('collision', d3.forceCollide(50))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('link', d3.forceLink().links(links))
            .on('tick', ticked);
    }


    function dragstarted(d) {
        d3.event.sourceEvent.stopPropagation();
        d3.select(this).classed("dragging", true);
    }

    function dragged(d) {
        d3.select(this).attr("cx", d.x = d3.event.x).attr("cy", d.y = d3.event.y);
    }

    function dragended(d) {
        d3.select(this).classed("dragging", false);
    }

    function updateLinks() {
        var u = d3.select('.link')
            .selectAll('line')
            .data(links)

        u.enter()
            .append('line')
            .merge(u)
            .attr('marker-end','url(#arrowhead)')
            .attr('x1', function(d) {return d.source.x})
            .attr('y1', function(d) {return d.source.y})
            .attr('x2', function(d) {return d.target.x})
            .attr('y2', function(d) {return d.target.y})

        u.exit().remove()
    }

    function updateNodes() {
        d3.select('.node')
        .selectAll('circle')
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; })
        .on("mouseover", function(d) { tooltip.style("opacity", 1); })
        .on("mousemove", function(d) {
            tooltip.html(d.pathname)
            .style("left", d3.event.pageX + "px")
            .style("top", d3.event.pageY + "px");
        })
        .on("mouseleave", function(d) { tooltip.style("opacity", 0); })
        .call(d3.drag()
            .subject(function (d) { return d; })
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));
    }

    function ticked() {
        updateLinks();
        updateNodes();
    }

    function toggle_outside()
    {
        if (show_outside == "Y") {
            location.href = "/sysgraph/N/" + pathnames[0];
        }
        else {
            location.href = "/sysgraph/Y/" + pathnames[0];
        }
    }

    window.onload = function() {
        width = window.innerWidth * .98;
        svg = d3.select("svg")
            .attr("width", width)
            .attr("height", height);
        d3_setup();
    };

    window.onresize = function() {
        width = window.innerWidth * .98;
        d3.select("svg").attr("width", width);
    }


    </script>
    </head>
    <body>
        <input type="button" onclick="location.href='/';" value="Home" />
        <input type="checkbox" onclick="toggle_outside();"  value="N"> Show Outside Connections <br>

        <div id="content">
            <svg>
            </svg>
        </div>
    </body>
    </html>
    """ % dict(pathnames=[pathname], subgroups=subgroups, show_outside=show_outside, 
               nodes=nodes, links=links))


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
