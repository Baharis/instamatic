from formats import write_tiff
import numpy as np
import matplotlib.pyplot as plt
from find_crystals import find_crystals
import json
from calibrate import CalibStage, CalibBeamShift, CalibDirectBeam, get_diffraction_pixelsize
from TEMController import config
import fileio


def get_status():
    status = {"stage_lowmag": {"name":"Stage (lowmag)", "ok":False, "msg":"no, please run instamatic.calibrate_stage_lowmag"},
                "stage_mag1": {"name":"Stage (mag1)", "ok":False, "msg":"no, please run instamatic.calibrate_mag1"},
                "beamshift": {"name":"BeamShift", "ok":False, "msg":"no, please run instamatic.calibrate_beamshift"},
                "directbeam": {"name":"DirectBeam", "ok":False, "msg":"no, please run instamatic.calibrate_directbeam"},
                "holes": {"name":"Holes", "ok":False, "msg": "no, please run instamatic.map_holes"},
                "radius": {"name":"Radius", "ok":False, "msg": "no, please run instamatic.prepare_experiment"},
                "params": {"name":"Params", "ok":False, "msg": "no"},
                "ready": {"name":"Ready", "ok":True, "msg": "Experiment is ready!!"}
            }

    try:
        calib = CalibStage.from_file()
    except IOError:
        pass
    else:
        status["stage_lowmag"].update({"ok":True, "msg": "OK"})

    try:
        calib = CalibBeamShift.from_file()
    except IOError:
        pass
    else:
        status["beamshift"].update({"ok":True, "msg": "OK"})

    try:
        calib = CalibDirectBeam.from_file()
    except IOError:
        pass
    else:
        status["directbeam"].update({"ok":True, "msg": "OK"})

    try:
        params = json.load(open("params.json","r"))
    except IOError:
        pass
    else:
        if "angle" in params:
            status["stage_mag1"].update({"ok":True, "msg":"OK, angle={:.2f} deg.".format(np.degrees(params["angle"]))})
        keys = "magnification", "diff_difffocus", "diff_brightness"
        missing_keys = [key for key in keys if key not in params]
        if missing_keys:
            status["params"].update({"ok":False, "msg":"no, missing {}".format(", ".join(missing_keys))})
        else:
            status["params"].update({"ok":True, "msg":"OK"})

    try:
        experiment = fileio.load_experiment()
    except Exception as e:
        pass
    else:
        status["radius"].update({"ok":True, "msg":"OK, {:.2f} um".format(experiment["radius"] / 1000.0)})

    try:
        hole_coords = fileio.load_hole_stage_positions()
    except Exception:
        pass
    else:
        status["holes"].update({"ok":True, "msg":"OK, {} locations stored".format(len(hole_coords))})

    if not all([status[key]["ok"] for key in status]):
        status["ready"].update({"ok":False, "msg":"Experiment is NOT ready!!"})

    return status


def get_grid(nx, ny=0, radius=1, borderwidth=0.8):
    """Make a grid (size=n*n), and return the coordinates of those
    fitting inside a circle (radius=r)
    nx: `int`
    ny: `int` (optional)
        Used to define a mesh nx*ny, if ny is missing, nx*nx is used
    radius: `float`
        radius of hole
    borderwidth: `float`, 0.0 - 1.0
        define a border around the circumference not to place any points
        should probably be related to the effective camera size: 
    """
    xr = np.linspace(-1, 1, nx)
    if ny:
        yr = np.linspace(-1, 1, ny)
    else:
        yr = xr
    xgrid, ygrid = np.meshgrid(xr, yr)
    sel = xgrid**2 + ygrid**2 < 1.0*(1-borderwidth)
    xvals = xgrid[sel].flatten()
    yvals = ygrid[sel].flatten()
    return xvals*radius, yvals*radius


def get_offsets(box_x, box_y=0, radius=75, padding=2, k=1.0, angle=0, plot=False):
    """
    box_x: float or int,
        x-dimensions of the box in micrometers. 
        if box_y is missing, box_y = box_x
    box_y: float or int,
        y-dimension of the box in micrometers (optional)
    radius: int or float,
        size of the hole in micrometer
    padding: int or float
        distance between boxes in micrometers
    k: float,
        scaling factor for the borderwidth
    """
    nx = 1 + int(2.0*radius / (box_x+padding))
    if box_y:
        ny = 1 + int(2.0*radius / (box_y+padding))
        diff = 0.5*(2*max(box_x, box_y)**2)**0.5
    else:
        diff = 0.5*(2*(box_x)**2)**0.5
        ny = 0
    
    borderwidth = k*(1.0 - (radius - diff) / radius)
       
    x_offsets, y_offsets = get_grid(nx=nx, ny=ny, radius=radius, borderwidth=borderwidth)
    
    if angle:
        sin = np.sin(angle)
        cos = np.cos(angle)
        r = np.array([
                    [ cos, -sin],
                    [ sin,  cos]])
        x_offsets, y_offsets = np.dot(np.vstack([x_offsets, y_offsets]).T, r).T

    if plot:
        from matplotlib import patches
        num = len(x_offsets)
        textstr = "grid: {} x {}\nk: {}\nborder: {:.2f}\nradius: {:.2f}\nboxsize: {:.2f} x {:.2f} um\nnumber: {}".format(nx, ny, k, borderwidth, radius, box_x, box_y, num)
        
        print
        print textstr
        
        cx, cy = (box_x/2.0, box_y/2.0)
        if angle:
            cx, cy = np.dot((cx, cy), r)
        
        if num < 1000:
            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(111)
            plt.scatter(0, 0)
            plt.scatter(x_offsets, y_offsets, picker=8, marker="+")
            circle = plt.Circle((0, 0), radius, fill=False, color="blue")
            ax.add_artist(circle)
            circle = plt.Circle((0, 0), radius*(1-borderwidth/2), fill=False, color="red")
            ax.add_artist(circle)
            
            for dx, dy in zip(x_offsets, y_offsets):
                rect = patches.Rectangle((dx - cx, dy - cy), box_x, box_y, fill=False, angle=np.degrees(-angle))
                ax.add_artist(rect)
            
            ax.text(1.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_aspect('equal')
            plt.show()
    
    return np.vstack((x_offsets, y_offsets)).T


class Experiment(object):
    """docstring for Experiment"""
    def __init__(self, ctrl, config):
        super(Experiment, self).__init__()
        self.ctrl = ctrl
        self.camera = ctrl.cam.name

        self.load_calibration(**config)

    def load_calibration(self, **kwargs):
        """Load user specified config and calibration files"""
       
        d = fileio.load_experiment()
        self.hole_centers = d["centers"]
        self.hole_radius = d["radius"] / 1000 # nm -> um
    
        self.calib_stage = CalibStage.from_file()
        self.calib_beamshift = CalibBeamShift.from_file()
        self.calib_directbeam = CalibDirectBeam.from_file()
    
        self.magnification   = kwargs["magnification"]
        self.image_binsize   = kwargs.get("image_binsize",       2   )
        self.image_exposure  = kwargs.get("image_exposure",      0.1 )
        self.image_spotsize  = kwargs.get("image_spotsize",      1   )
        self.image_dimensions = config.mag1_dimensions[self.magnification]
    
        self.diff_binsize    = kwargs.get("diff_binsize",        2   )
        self.diff_exposure   = kwargs.get("diff_exposure",       0.1 )
        self.diff_brightness = kwargs["diff_brightness"]
        self.diff_difffocus  = kwargs["diff_difffocus"]
        self.diff_spotsize   = kwargs.get("diff_spotsize",       5   )
        self.diff_cameralength = kwargs.get("diff_cameralength",       1500)
        self.diff_pixelsize  = get_diffraction_pixelsize(self.diff_difffocus, self.diff_cameralength, binsize=self.diff_binsize, camera=self.camera)
    
        self.crystal_spread = kwargs.get("crystal_spread", 2.5)

        # self.sample_rotation_angles = ( -10, -5, 5, 10 )
        self.sample_rotation_angles = ()
    
        self.camera_rotation_angle = config.camera_rotation_vs_stage_xy

        box_x, box_y = self.image_dimensions

        offsets = get_offsets(box_x, box_y, self.hole_radius, k=1, padding=2, angle=self.camera_rotation_angle, plot=False)
        self.offsets = offsets * 1000

    def initialize_microscope(self):
        """Intialize microscope"""

        import atexit
        atexit.register(self.ctrl.restore)

        self.ctrl.mode_diffraction()
        print raw_input(" >> Getting neutral diffraction shift, press enter to continue")
        self.neutral_diffshift = np.array(self.ctrl.diffshift.get())
        print self.neutral_diffshift
        print "DiffShift(x={}, y={})".format(*self.neutral_diffshift)
    
        self.ctrl.mode_mag1()
        self.ctrl.magnification.value = self.magnification
        self.ctrl.brightness.max()
        self.neutral_beamshift = self.calib_beamshift.center()
        self.ctrl.beamshift.set(*self.neutral_beamshift) # calib_beamshift.reference_shift?

    def image_mode(self):
        """Switch to image mode (mag1), reset beamshift/diffshift, spread beam"""
        
        print
        print " >> Switching back to image mode" 

        self.ctrl.beamshift.set(*self.neutral_beamshift)
        self.ctrl.diffshift.set(*self.neutral_diffshift)

        self.ctrl.mode_mag1()
        self.ctrl.brightness.max()

    def diffraction_mode(self):
        """Switch to diffraction mode, focus the beam, and set the correct focus
        """
        print
        print " >> Switching to diffraction mode"
        self.ctrl.brightness.set(self.diff_brightness)
        self.ctrl.mode_diffraction()
        self.ctrl.difffocus.value = self.diff_difffocus # difffocus must be set AFTER switching to diffraction mode

    def report_status(self):
        """Report experiment status"""

        print
        print "Imaging     : binsize = {}".format(self.image_binsize)
        print "              exposure = {}".format(self.image_exposure)
        print "              magnification = {}".format(self.magnification)
        print "              spotsize = {}".format(self.image_spotsize)
        print "Diffraction : binsize = {}".format(self.diff_binsize)
        print "              exposure = {}".format(self.diff_exposure)
        print "              brightness = {}".format(self.diff_brightness)
        print "              spotsize = {}".format(self.diff_spotsize)
        print
        print "Usage:"
        print "    type 'next' to go to the next hole"
        print "    type 'auto' to enable automatic data collection (until next hole)"
        print "    type 'plot' to toggle plotting mode"
        print "    type 'exit' to quit"
        print "    hit 'Ctrl+C' to interrupt the script"
        print ""

    def loop_centers(self):
        """Loop over holes in the copper grid
        Move the stage to all positions defined in centers

        Return
            di: dict, contains information on holes
        """

        for i, (x, y) in enumerate(self.hole_centers):
            try:
                self.ctrl.stageposition.set(x=x, y=y)
            except ValueError as e:
                print e
                print " >> Moving to next hole..."
                print
                continue
            else:
                print "\n >> Going to next hole center \n    ->", self.ctrl.stageposition
                di = {"exp_hole_number": i, "exp_hole_center": (x,y)}
                yield i, di
            
    def loop_positions(self, exp_hole_center, **kwargs):
        """Loop over positions in a hole in the copper grid
        Move the stage to each of the positions in self.offsets

        Return
            dj: dict, contains information on positions
        """
        x, y = exp_hole_center
        for j, (x_offset, y_offset) in enumerate(self.offsets):
            try:
                self.ctrl.stageposition.set(x=x+x_offset, y=y+y_offset)
            except ValueError as e:
                print e
                print " >> Moving to next position..."
                print
                continue
            else:
                print "\n     >> Going to next position \n        ->", self.ctrl.stageposition
                dj = {"exp_image_number": j, "exp_hole_offset": (x_offset, y_offset)}
                yield j, dj

    def loop_crystals(self, crystal_coords):
        """Loop over crystal coordinates (pixels)
        Switch to diffraction mode, and shift the beam to be on the crystal

        Return
            dk: dict, contains information on beam/diffshift

        """

        ncrystals = len(crystal_coords)
        if ncrystals == 0:
            raise StopIteration("No crystals found.")

        beamshift_coords = self.calib_beamshift.pixelcoord_to_beamshift(crystal_coords)
        for k, beamshift in enumerate(beamshift_coords):
            print " >> Focusing on crystal {}/{}".format(k+1, ncrystals)
            self.ctrl.beamshift.set(*beamshift)
            self.diffraction_mode()
        
            # compensate beamshift
            beamshift_offset = beamshift - self.neutral_beamshift
            pixelshift = self.calib_directbeam.beamshift2pixelshift(beamshift_offset)
        
            diffshift_offset = self.calib_directbeam.pixelshift2diffshift(pixelshift)
            diffshift = self.neutral_diffshift - diffshift_offset
        
            self.ctrl.diffshift.set(*diffshift.astype(int))

            dk = {"exp_pattern_number": k, "exp_diffshift_offset": diffshift_offset, "exp_beamshift_offset": beamshift_offset}
            yield k, dk

    def run(self, ctrl=None, **kwargs):
        """Run serial electron diffraction experiment"""

        self.initialize_microscope()

        d_image = {
                "exp_neutral_diffshift": self.neutral_beamshift,
                "exp_neutral_beamshift": self.neutral_diffshift,
                "ImageDimensions": self.image_dimensions
        }
        d_diff = {
                "exp_neutral_diffshift": self.neutral_beamshift,
                "exp_neutral_beamshift": self.neutral_diffshift,
                "ImagePixelsize": self.diff_pixelsize
        }

        for i, di in self.loop_centers():
            print di
            auto = False
            plot = False
            for j, dj in self.loop_positions(**di):
                if not auto:
                    answer = raw_input("\n (Press <enter> to save an image and continue) \n >> ")
                    if answer == "exit":
                        print " >> Interrupted..."
                        exit()
                    elif answer == "next":
                        print " >> Going to next hole"
                        break
                    elif answer == "auto":
                        auto = True
                    elif answer == "plot":
                        plot = not plot
    
                outfile = "image_{:04d}_{:04d}".format(i, j)
                comment = "Hole {} image {}\n".format(i, j)
    
                self.ctrl.tem.setSpotSize(self.image_spotsize)
                img, h = self.ctrl.getImage(binsize=self.image_binsize, exposure=self.image_exposure, comment=comment)
                self.ctrl.tem.setSpotSize(self.diff_spotsize)
    
                crystal_coords = find_crystals(img, h["Magnification"], spread=self.crystal_spread, plot=False) * self.image_binsize
    
                for d in (d_image, di, dj):
                    h.update(d)
                h["exp_crystal_coords"] = crystal_coords.tolist()

                write_tiff(outfile, img, header=h)
    
                # plot_props(img, crystals, fname=outfile+".png")
    
                for k, dk in self.loop_crystals(crystal_coords):
                    outfile = "image_{:04d}_{:04d}_{:04d}".format(i, j, k)
                    comment = "Hole {} image {} Crystal {}".format(i, j, k)
                    img, h = self.ctrl.getImage(binsize=self.diff_binsize, exposure=self.diff_exposure, comment=comment, verbose=False)
                    
                    for d in (d_diff, di, dj, dk):
                        h.update(d)

                    write_tiff(outfile, img, header=h)
                 
                    for rotation_angle in self.sample_rotation_angles:
                        print " >> Rotation angle = ".format(rotation_angle)
                        self.ctrl.stageposition.a = rotation_angle
    
                        outfile = "image_{:04d}_{:04d}_{:04d}_{}".format(i, j, k, rotation_angle)
                        img, h = self.ctrl.getImage(binsize=self.diff_binsize, exposure=self.diff_exposure, comment=comment, verbose=False)
                                                    
                        for d in (d_diff, di, dj, dk):
                            h.update(d)

                        write_tiff(outfile, img, header=h)
    
                    self.ctrl.stageposition.a = 0
    
                self.image_mode()


def main_gui():
    from gui import main
    main.start()


def main():
    import TEMController

    status = get_status()

    print "\nCalibration:"
    for key in "stage_lowmag", "stage_mag1", "beamshift", "directbeam":
        print "    {name:20s}: {msg:s}".format(**status[key])

    print "\nExperiment"
    for key in "holes", "radius", "params", "ready":
        print "    {name:20s}: {msg:s}".format(**status[key])

    ready = status["ready"]["ok"]

    if ready:
        params = json.load(open("params.json","r"))
        if raw_input("\nExperiment ready. Enter 'go' to start. >> ") != "go":
            exit()
        print
        ctrl = TEMController.initialize()

        exp = Experiment(ctrl, params)
        exp.report_status()
        exp.run()
    else:
        print "\nExperiment not ready yet!!"


if __name__ == '__main__':
    main()
