import os, sys, glob

import argparse
import pandas as pd

import time

from indexer import *
from tqdm import tqdm
import logging

from instamatic.formats import *

__version__ = "2017-03-09"
__author__ = "Stef Smeets"
__email__ = "stef.smeets@mmk.su.se"


TEMPLATE = """title: indexing
experiment:
  azimuth: -6.61
  pixelsize: 0.003957
  stretch: 2.43
projections:
  dmax: 10.0
  dmin: 1.0
  thickness: 100
cell:
  name: LTA
  params:
  - 24.61
  - 24.61
  - 24.61
  - 90
  - 90
  - 90
  spgr: Fm-3c
data:
  csv_out: results.csv
  drc_out: indexing
  glob: data/*.tiff
"""


def get_files(file_pat):
    """Grab files from globbing pattern or stream file"""
    if os.path.exists(file_pat):
        f = open(file_pat, "r")
        fns = [line.strip() for line in f if not line.startswith("#")]
    else:
        fns = glob.glob(file_pat)
    return fns


def printer(data):
    """Print things to stdout on one line dynamically"""
    sys.stdout.write("\r\x1b[K"+data.__str__())
    sys.stdout.flush()


def merge_csv(csvs):
    """Read amd combine csv files `csvs` into one df,

    Returns: pd.DataFrame with combined items"""
    combined = pd.concat((read_csv(csv) for csv in csvs))

    for csv in csvs:
        os.unlink(csv)

    return combined


class ProgressBar(object):
    """docstring for ProgressBar"""
    def __init__(self):
        
        self.reverse = False
        self.a = ' '
        self.b = '='
        self.width = 10
        self.delay = 0.2

    def loop(self):
        for i in range(self.width):
            if self.reverse:
                i = self.width-i
            time.sleep(self.delay)
            printer('['+self.b*(i)+self.a*(self.width-i)+']')
        self.reverse = not self.reverse
        for i in range(self.width):
            if self.reverse:
                i = self.width-i
            time.sleep(self.delay)
            printer('['+self.a*(self.width-i)+self.b*i+']')
        self.a,self.b = self.b,self.a
        
    def clear(self):
        printer('')


def multi_run(arg, procs=1, dry_run=False, logfile=None):
    import subprocess as sp
    from multiprocessing import cpu_count

    d = yaml_ordered_load(sys.argv[1])

    file_pat  = d["data"]["glob"]
    csv_out   = d["data"]["csv_out"]

    fns = get_files(file_pat)

    nfiles = len(fns)
    print "Found {} files".format(nfiles)

    csv_outs = []
    for i in xrange(procs):
        print " Chunk #{}: {} files".format(i, len(fns[i::procs]))
        root, ext = os.path.splitext(csv_out)
        csv_outs.append("{}_{}{}".format(root, i, ext))
    print

    cores = cpu_count()

    assert procs >= 0, 'Expected a positive number of processors'
    if procs > cores:
        print '{} cores detected, are you sure you want to run {} processes? [y/n]'.format(cores, procs)
        if not raw_input(' >> [y] ').lower() == 'y':
            sys.exit()

    processes = []

    t1 = time.time()

    print 'Starting processes...'
    for i in xrange(procs):
        cmd = "instamatic.index.exe {} -c {} {}".format(arg, i, procs)
        if logfile:
            cmd += " --logfile {}".format(logfile)

        print "     >>", cmd,

        if not dry_run:
            # CREATE_NEW_CONSOLE is windows only
            p = sp.Popen(cmd, creationflags=sp.CREATE_NEW_CONSOLE)
            processes.append(p)
            print ';; started (PID={})'.format(p.pid)
        else:
            print ';; not started'

    if dry_run:
        return
   
    pb = ProgressBar()
    while any(p.poll() == None for p in processes):
        pb.loop()
    pb.clear()

    t2 = time.time()

    all_results = read_csv(csv_outs)
    all_results = all_results.sort_values(by="score", ascending=False)
    for csv in csv_outs:
        os.unlink(csv)
    write_ycsv(csv_out, data=all_results, metadata=d)
    print "Writing results to {}".format(csv_out)

    print "Time taken: {:.0f} s / {:.1f} s per image".format(t2-t1, (t2-t1)/nfiles)
    print
    print " >> Done << "


def run(arg, chunk=None, dry_run=False, log=None):
    log = log or logging.getLogger(__name__)

    if len(sys.argv) == 1:
        print "Usage: instamatic.index indexing.inp"
        print
        print "Example input file:"
        print 
        print TEMPLATE
        exit()

    d = yaml_ordered_load(arg)

    azimuth   = d["experiment"]["azimuth"]
    stretch   = d["experiment"]["stretch"]
    pixelsize = d["experiment"]["pixelsize"]
    
    dmin      = d["projections"]["dmin"]
    dmax      = d["projections"]["dmax"]
    thickness = d["projections"]["thickness"]
    
    file_pat  = d["data"]["glob"]
    csv_out   = d["data"]["csv_out"]
    drc_out   = d["data"]["drc_out"]

    sigma_min  = d["background"]["sigma_min"]
    sigma_max  = d["background"]["sigma_max"]
    threshold  = d["background"]["threshold"]

    azimuth = np.radians(stretch)
    stretch = stretch / (2*100)
    tr_mat = affine_transform_ellipse_to_circle(azimuth, stretch)

    refine = True
    method = "powell"
    radius = 3
    nsolutions = 25
    beam_center_sigma = 10

    if isinstance(d["cell"], (tuple, list)):
        pixelsize = d["experiment"]["pixelsize"]
        indexer = IndexerMulti.from_cells(d["cell"], pixelsize=pixelsize, **d["projections"])
    else:
        projector = Projector.from_parameters(thickness=d["projections"]["thickness"], **d["cell"])
        indexer = Indexer.from_projector(projector, pixelsize=d["experiment"]["pixelsize"])

    fns = get_files(file_pat)

    if chunk:
        offset, cpu_count = chunk
        root, ext = os.path.splitext(csv_out)
        csv_out = "{}_{}{}".format(root, offset, ext)
        fns = fns[offset::cpu_count]
        prefix = "Chunk #{}: ".format(offset)
    else:
        prefix = ""

    nfiles = len(fns)
    print "Found {} files".format(nfiles)

    if not os.path.exists(drc_out):
        os.mkdir(drc_out)
    
    all_results = {}
    
    t = tqdm(fns, desc=fns[0])

    for i, fn in enumerate(t):
        # print "{}/{}: {}".format(i, nfiles, fn), 
        t.set_description(fn)
        t.update()

        img, _ = read_image(fn)
        
        center = find_beam_center(img, sigma=beam_center_sigma)  # cx cy
        
        img = apply_transform_to_image(img, tr_mat, center=center)
        
        img = remove_background_gauss(img, sigma_min, sigma_max, threshold=threshold)
        results = indexer.index(img, center, nsolutions=nsolutions)
        
        refined = indexer.refine_all(img, results, sort=True, method=method, vary_center=True, vary_scale=True)

        best = refined[0]

        all_results[fn] = best
            
        hklie = indexer.get_intensities(img, best, radius=radius)

        root, ext = os.path.splitext(os.path.basename(fn))
        out = os.path.join(drc_out, root+".hkl")

        np.savetxt(out, hklie, fmt="%4d%4d%4d %7.1f %7.1f")

        log.info("%s%d/%d %s -> %7.0f", prefix, i, nfiles, fn, best.score)

    if chunk:
        write_csv(csv_out, all_results)
    else:
        write_ycsv(csv_out, data=all_results, metadata=d)
    print "Writing results to {}".format(csv_out)
    
    time_taken = t.last_print_t - t.start_t
    print "Time taken: {:.0f} s / {:.1f} s per image".format(time_taken, (time_taken)/nfiles)
    print
    print " >> DONE <<"


def main():
    usage = """instamatic.index indexing.inp"""

    description = """
Program for indexing electron diffraction images.

""" 
    
    epilog = 'Updated: {}'.format(__version__)
    
    parser = argparse.ArgumentParser(#usage=usage,
                                    description=description,
                                    epilog=epilog, 
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    version=__version__)
    
    parser.add_argument("args", 
                        type=str, metavar="FILE",
                        help="Path to input file.")

    parser.add_argument("-j", "--procs", metavar='N',
                        action="store", type=int, dest="procs",
                        help="Number of processes to use (default = 1)")

    parser.add_argument("-c", "--chunk", metavar='N', nargs=2,
                        action="store", type=int, dest="chunk",
                        help="Used internally to specify the chunk number to process.")

    parser.add_argument("-d", "--dry",
                        action="store_true", dest="dry_run",
                        help="Runs the program, but doesn't start any processes.")

    parser.add_argument("-l", "--logfile",
                        action="store", type=str, dest="logfile",
                        help="Specify logfile (default=indexing.log).")
    
    parser.set_defaults(procs=1,
                        chunk=None,
                        dry_run=False,
                        resize=False,
                        logfile="indexing.log"
                        )
    
    options = parser.parse_args()
    arg = options.args

    if not arg:
        parser.print_help()
        sys.exit()

    if options.procs > 1:
        multi_run(arg, procs=options.procs, dry_run=options.dry_run, logfile=options.logfile)
    else:
        logging.basicConfig(format="%(asctime)s | %(module)s:%(lineno)s | %(levelname)s | %(message)s", filename=options.logfile, level=logging.DEBUG)
        log = logging.getLogger(__name__)
        log.info("Start indexing")
        run(arg, options.chunk, log=log)


if __name__ == '__main__':
    main()
