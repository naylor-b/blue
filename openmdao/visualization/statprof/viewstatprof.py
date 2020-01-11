
"""Define a function to view statistical profile data."""
import os
import sys
import signal
import json
import contextlib
from itertools import chain
from collections import defaultdict

from six import iteritems

import numpy as np

from openmdao.core.problem import Problem
from openmdao.utils.units import convert_units
from openmdao.utils.mpi import MPI
from openmdao.utils.webview import webview
from openmdao.utils.general_utils import printoptions
from openmdao.core.system import System
from openmdao.core.problem import Problem
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver

try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass


def view_statprof(raw_stat_file, outfile='src_heat_map.html', show_browser=True, title=None):
    """
    Generate a self-contained html file containing a detailed statistical profile viewer.

    Optionally pops up a web browser to view the file.

    Parameters
    ----------
    raw_stat_file : str
        The name of the raw statistical profiling data file.

    outfile : str, optional
        The name of the output html file.  Defaults to 'statprof.html'.

    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.

    title : str, optional
        Sets the title of the web page.
    """
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    dct = defaultdict(int)
    for parts in _rawfile_iter(raw_stat_file):
        pass

    table = []
    idx = 1  # unique ID for use by Tabulator
    with open(raw_stat_file, 'r') as f:
        for line in f:
            parts = line.split()
            fname, line_number = parts[:2]
            hits = parts[-3]
            row = {'id': idx, 'fname': fname, 'line_number': line_number, 'hits': hits}
            table.append(row)
            idx += 1

    if title is None:
        title = ''

    data = {
        'title': title,
        'table': table,
    }

    code_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.join(code_dir, 'libs')
    style_dir = os.path.join(code_dir, 'style')

    with open(os.path.join(code_dir, outfile), "r") as f:
        template = f.read()

    with open(os.path.join(libs_dir, 'tabulator.min.js'), "r") as f:
        tabulator_src = f.read()

    with open(os.path.join(style_dir, 'tabulator.min.css'), "r") as f:
        tabulator_style = f.read()

    jsontxt = json.dumps(data)

    with open(outfile, 'w') as f:
        s = template.replace("<statprof_data>", jsontxt)
        s = s.replace("<tabulator_src>", tabulator_src)
        s = s.replace("<tabulator_style>", tabulator_style)
        f.write(s)

    if show_browser:
        webview(outfile)


# # generic stat profiler.  will work on Windows as well, but has lower resolution
# class StatisticalProfiler_old(object):
#     def __init__(self, sleep_interval=0.01):
#         self._sleep_interval = sleep_interval
#         self._stopped = True
#         self._prof_thread = None
#         self._stats = defaultdict(int)

#     def start(self):
#         print("starting")
#         self._stopped = False
#         self._prof_thread = self.start_thread(self.collecting)

#     def stop(self):
#         self._stopped = True
#         self._prof_thread.join()
#         self._prof_thread = None
#         import pprint
#         pprint.pprint(self._stats)
#         print("stop complete")

#     def save(self):
#         pass

#     def shutdown(self):
#         pass

#     def collecting(self):
#         while True:
#             if self._stopped:
#                 break
#             sleep(self._sleep_interval)
#             self.collect_frame_data()

#     def record(self, frame):
#         if 'self' in frame.f_locals and frame.f_locals['self'] is not self:
#             name = type(frame.f_locals['self']).__name__ + '.' + frame.f_code.co_name
#             self._stats[name] += 1
#             print(name, self._stats[name])

#     def collect_frame_data(self):
#         switch_interval = getswitchinterval()
#         try:
#             setswitchinterval(10000)
#             for frame in _current_frames().values():
#                 self.record(frame)
#         finally:
#             setswitchinterval(switch_interval)

#     def start_thread(self, fn):
#         """
#         Start a daemon thread running the given function.
#         """
#         thread = threading.Thread(target=fn)
#         thread.setDaemon(True)
#         thread.start()
#         return thread


omtypes = (System, Solver, Problem, Driver)


# based on code from plop (https://github.com/bdarnell/plop.git)
# only works on linux, MacOS because it uses signal timers
# TODO: check to see if newest Windows version might support this...
class StatisticalProfiler(object):
    MODES = {
        'prof': (signal.ITIMER_PROF, signal.SIGPROF),
        'virtual': (signal.ITIMER_VIRTUAL, signal.SIGVTALRM),
        'real': (signal.ITIMER_REAL, signal.SIGALRM),
    }

    def __init__(self, outfile='raw_statprof.0', interval=0.005, mode='virtual'):
        self.outfile = outfile
        self.stream = None
        self._stats = defaultdict(int)
        self.interval = interval
        self.mode = mode
        assert mode in StatisticalProfiler.MODES, 'valid modes are: ["prof", "virtual", "real"] but you chose {}'.format(mode)
        timer, sig = StatisticalProfiler.MODES[mode]
        signal.signal(sig, self._statprof_handler)
        signal.siginterrupt(sig, False)

        self.stacks = []
        self.samples_remaining = 0
        self.stopping = False
        self.stopped = False
        self._recording = False

        self.samples_taken = 0
        self.hits = 0

    def start(self, duration=600.0):
        if self.stream is None:
            self.stream = open(self.outfile, 'w')
        self.stopping = False
        self.stopped = False
        self.samples_remaining = int(duration / self.interval)
        timer, sig = StatisticalProfiler.MODES[self.mode]
        signal.setitimer(timer, self.interval, self.interval)

    def stop(self):
        self.stopping = True
        while not self.stopped:
            pass  # need busy wait; ITIMER_PROF doesn't proceed while sleeping
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def record(self, frame):
        global omtypes
        self._recording = True
        try:
            if 'self' in frame.f_locals and frame.f_locals['self'] is not self:
                self.hits += 1
                slf = frame.f_locals['self']
                if isinstance(slf, omtypes):
                    try:
                        pname = slf.msginfo
                    except Exception:
                        pname = type(slf).__name__
                else:
                    pname = type(slf).__name__
                print(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name, pname, file=self.stream)
            else:
                print(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name, 'N/A', file=self.stream)
        finally:
            self._recording = False

    def _statprof_handler(self, sig, current_frame):
        self.samples_remaining -= 1
        if self._recording:
            return

        if self.samples_remaining <= 0 or self.stopping:
            signal.setitimer(StatisticalProfiler.MODES[self.mode][0], 0, 0)
            self.stopped = True
            print('', file=self.stream)  # blank line
            print(self.samples_taken, file=self.stream)
            return

        for tid, frame in iteritems(sys._current_frames()):
            while frame is not None:
                if frame.f_code.co_name == '_statprof_py_file':
                    break
                if frame.f_code.co_name != '_statprof_handler':
                    self.record(frame)
                frame = frame.f_back

        self.samples_taken += 1


def _statprof_setup_parser(parser):
    parser.add_argument('-i', '--interval', action='store', dest='interval', type=float,
                        default=.005, help='Sampling interval.')
    parser.add_argument('-d', '--duration', action='store', dest='duration', type=float,
                        default=30., help='Sampling duration in seconds. File will be executed multiple '
                        'times if duration has not been reached.')
    parser.add_argument('--sampling_mode', action='store', dest='sampling_mode',
                        default='virtual', help='Sampling mode. Must be one of ["prof", "virtual", "real"].')
    parser.add_argument('--groupby', action='store', dest='groupby', default='line', 
                        help='How to group stats. Must be one of ["instance", "line", "function", "instfunction"]')
    parser.add_argument('--no_browser', action='store_true', dest='noshow',
                        help="Don't pop up a browser to view the data.")
    parser.add_argument('file', metavar='file', nargs='*',
                        help='Raw profile data file or a python file.')


def _statprof_exec(options, user_args):
    """
    Called from the command line (openmdao statprof command) to create a statistical profile.
    """

    if not options.file:
        print("No files to process.")
        sys.exit(0)

    if len(options.file) > 1:
        print("statprof can only process a single file.", file=sys.stderr)
        sys.exit(-1)

    if options.file[0].endswith('.py'):
        outfile = 'statprof.raw'
        if MPI:
            outfile = outfile + '.' + str(MPI.COMM_WORLD.rank)

        samples_taken = _statprof_py_file(options, outfile, user_args)
    else:  # assume it's a raw statprof data file
        outfile = options.file[0]

    if options.noshow:
        _process_raw_statfile(outfile, options)
    else:
        view_statprof(_get_statfile_name(options))


def _get_statfile_name(options):
    return 'statprof_{}.out'.format(options.groupby)


def _rawfile_iter(fname):
    with open(fname, 'r') as f:
        read_samples = False
        for line in f:
            if line.startswith('<'):
                continue
            line = line.strip()
            if not line:
                read_samples = True
                continue
            if read_samples:
                # read samples_taken
                yield [line.strip()]
                break
            else:
                yield line.split()


def _process_raw_statfile(fname, options):
    if options.groupby not in ('instance', 'line', 'function', 'instfunction'):
        raise RuntimeError("Illegal option for --groupby.  Must be 'instance', 'line', or 'function'.")

    total_hits = 0
    outstream = open(_get_statfile_name(options), 'w')
    dct = defaultdict(int)

    if options.groupby == 'line':
        for parts in _rawfile_iter(fname):
            if len(parts) == 1:
                samples_taken = int(parts[0])
                continue
            dct[parts[0], parts[1]] += 1
        display_line_data(dct, samples_taken, outstream)
    elif options.groupby == 'instance':
        for parts in _rawfile_iter(fname):
            if len(parts) == 1:
                samples_taken = int(parts[0])
                continue
            dct[tuple(parts[1:])] += 1
        display_instance_data(dct, samples_taken, outstream)
    elif options.groupby == 'instfunction':
        for parts in _rawfile_iter(fname):
            if len(parts) == 1:
                samples_taken = int(parts[0])
                continue
            fname, lnum, func = parts[:3]
            obj = ' '.join(parts[3:])
            if func == '<module>':
                dct[fname, lnum] += 1
            else:
                dct[func, obj] += 1
        display_instance_func_data(dct, samples_taken, outstream)

    outstream.close()


def display_line_data(dct, total_hits, stream=sys.stdout):
    for key, hits in sorted(dct.items(), key=lambda x: x[1]):
        print("{}  {} hits  {:<5.2f}%".format(key, hits, hits/total_hits*100), 
              file=stream)

def display_instance_data(dct, total_hits, stream=sys.stdout):
    for key, hits in sorted(dct.items(), key=lambda x: x[1]):
        print("{}  {} hits  {:<5.2f}%".format(key, hits, hits/total_hits*100),
              file=stream)

def display_instance_func_data(dct, total_hits, stream=sys.stdout):
    for key, hits in sorted(dct.items(), key=lambda x: x[1]):
        print("{}  {} hits  {:<5.2f}%".format(key, hits, hits/total_hits*100),
              file=stream)


def _statprof_py_file(options, outfile, user_args):
    """
    Run statistical profiling on the given python script.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    outfile : str
        Output filename.
    user_args : list of str
        Command line options after '--' (if any).  Passed to user script.

    Returns
    -------
    int
        Total number of samples taken.
    """
    progname = options.file[0]
    sys.path.insert(0, os.path.dirname(progname))

    # update sys.argv in case python script takes cmd line args
    sys.argv[:] = [progname] + user_args

    with open(progname, 'rb') as fp:
        code = compile(fp.read(), progname, 'exec')

    prof = StatisticalProfiler(outfile, interval=options.interval, 
                               mode=options.sampling_mode)
    prof.start(duration=options.duration)

    while not (prof.stopped or prof.stopping):
        globals_dict = {
            '__file__': progname,
            '__name__': '__main__',
            '__package__': None,
            '__cached__': None,
        }

        exec (code, globals_dict)

    prof.stop()

    return prof.samples_taken
