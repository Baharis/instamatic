from __future__ import annotations

import inspect
import threading
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.ttk import *

from instamatic.utils.spinbox import Spinbox

from .base_module import BaseModule, ModuleFrameMixin

# This GUI is tailored to the "default" cRED workflow:
#   - Choose a rotation span (deg) and a desired rotation speed (deg/s)
#   - The experiment will convert that into a Tecnai/FEI footfree speed (0..1)
#     using the installed stage-rotation calibration YAML.
#
# It removes the need to "enable footfree speed" via a checkbox, and runs the
# experiment in footfree mode by default.


class ExperimentalcRED(LabelFrame, ModuleFrameMixin):
    """GUI panel for doing cRED experiments (calibrated rotation speed)."""

    def __init__(self, parent):
        LabelFrame.__init__(self, parent, text='Continuous rotation electron diffraction')
        self.parent = parent

        sbwidth = 10
        self.init_vars()

        frame = Frame(self)
        Label(frame, text='Exposure time (s):').grid(row=1, column=0, sticky='W')
        exposure_time = Spinbox(
            frame,
            textvariable=self.var_exposure_time,
            width=sbwidth,
            from_=0.0,
            to=100.0,
            increment=0.01,
        )
        exposure_time.grid(row=1, column=1, sticky='W', padx=10)

        Checkbutton(frame, text='Beam unblanker', variable=self.var_unblank_beam).grid(
            row=1, column=2, sticky='W'
        )

        # Optional: minimum time between diffraction frames (start-to-start).
        Label(frame, text='Frame interval (s):').grid(row=2, column=0, sticky='W')
        self.e_frame_interval = Spinbox(
            frame,
            textvariable=self.var_frame_interval,
            width=sbwidth,
            from_=0.0,
            to=100.0,
            increment=0.01,
        )
        self.e_frame_interval.grid(row=2, column=1, sticky='W', padx=10)

        Separator(frame, orient=HORIZONTAL).grid(row=4, columnspan=3, sticky='ew', pady=10)

        # Image interval (defocused tracking images)
        Checkbutton(
            frame,
            text='Enable image interval',
            variable=self.var_enable_image_interval,
            command=self.toggle_interval_buttons,
        ).grid(row=5, column=2, sticky='W')
        self.c_toggle_defocus = Checkbutton(
            frame,
            text='Toggle defocus',
            variable=self.var_toggle_diff_defocus,
            command=self.toggle_diff_defocus,
            state=DISABLED,
        )
        self.c_toggle_defocus.grid(row=6, column=2, sticky='W')

        Label(frame, text='Image interval:').grid(row=5, column=0, sticky='W')
        self.e_image_interval = Spinbox(
            frame,
            textvariable=self.var_image_interval,
            width=sbwidth,
            from_=1,
            to=9999,
            increment=1,
            state=DISABLED,
        )
        self.e_image_interval.grid(row=5, column=1, sticky='W', padx=10)

        Label(frame, text='Diff defocus:').grid(row=6, column=0, sticky='W')
        self.e_diff_defocus = Spinbox(
            frame,
            textvariable=self.var_diff_defocus,
            width=sbwidth,
            from_=-10000,
            to=10000,
            increment=100,
            state=DISABLED,
        )
        self.e_diff_defocus.grid(row=6, column=1, sticky='W', padx=10)

        Label(frame, text='Image exposure (s):').grid(row=7, column=0, sticky='W')
        self.e_image_exposure = Spinbox(
            frame,
            textvariable=self.var_exposure_time_image,
            width=sbwidth,
            from_=0.0,
            to=100.0,
            increment=0.01,
            state=DISABLED,
        )
        self.e_image_exposure.grid(row=7, column=1, sticky='W', padx=10)

        self.RelaxButton = Button(
            frame, text='Relax beam', command=self.relax_beam, state=DISABLED
        )
        self.RelaxButton.grid(row=7, column=2, sticky='EW')

        # Rotation settings (default workflow)
        Separator(frame, orient=HORIZONTAL).grid(row=8, columnspan=3, sticky='ew', pady=10)

        Label(frame, text='Rotation span (deg):').grid(row=9, column=0, sticky='W')
        self.e_rotation_span = Spinbox(
            frame,
            textvariable=self.var_rotation_span_deg,
            width=sbwidth,
            from_=-70.0,
            to=70.0,
            increment=1.0,
        )
        self.e_rotation_span.grid(row=9, column=1, sticky='W', padx=10)

        Label(frame, text='Rotation speed (deg/s):').grid(row=10, column=0, sticky='W')
        self.e_rotation_speed = Spinbox(
            frame,
            textvariable=self.var_rotation_speed_deg_s,
            width=sbwidth,
            from_=0.01,
            to=50.0,
            increment=0.01,
        )
        self.e_rotation_speed.grid(row=10, column=1, sticky='W', padx=10)

        self.c_use_calib = Checkbutton(
            frame,
            text='Use rotation calibration',
            variable=self.var_use_stage_rotation_calibration,
        )
        self.c_use_calib.grid(row=10, column=2, sticky='W')

        Label(frame, text='Fallback footfree speed (0-1):').grid(row=11, column=0, sticky='W')
        self.e_footfree_speed = Spinbox(
            frame,
            textvariable=self.var_footfree_speed,
            width=sbwidth,
            from_=0.01,
            to=1.0,
            increment=0.01,
        )
        self.e_footfree_speed.grid(row=11, column=1, sticky='W', padx=10)

        self.c_auto_stop = Checkbutton(
            frame,
            text='Auto stop at target',
            variable=self.var_auto_stop_on_target,
        )
        self.c_auto_stop.grid(row=11, column=2, sticky='W')

        # Optional: explicit calibration file path (if auto-search fails)
        Label(frame, text='Rotation calib file:').grid(row=12, column=0, sticky='W')
        self.e_calib_file = Entry(
            frame, textvariable=self.var_stage_rotation_calibration_file, width=40
        )
        self.e_calib_file.grid(row=12, column=1, sticky='EW', padx=10)
        Button(frame, text='Browse', command=self.browse_calib_file).grid(
            row=12, column=2, sticky='EW'
        )

        coll_row0 = 13
        self.lb_coll0 = Label(frame, text='')
        self.lb_coll1 = Label(frame, text='')
        self.lb_coll2 = Label(frame, text='')
        self.lb_coll0.grid(row=coll_row0, column=0, columnspan=3, sticky='EW')
        self.lb_coll1.grid(row=coll_row0 + 1, column=0, columnspan=3, sticky='EW')
        self.lb_coll2.grid(row=coll_row0 + 2, column=0, columnspan=3, sticky='EW')
        frame.grid_columnconfigure(1, weight=1)
        frame.pack(side='top', fill='x', expand=False, padx=10, pady=10)

        # Output formats
        frame = Frame(self)
        Label(frame, text='Select output formats:').grid(row=5, columnspan=2, sticky='EW')
        Checkbutton(frame, text='.tiff', variable=self.var_save_tiff).grid(
            row=5, column=2, sticky='EW'
        )
        Checkbutton(frame, text='XDS (.smv)', variable=self.var_save_xds).grid(
            row=5, column=3, sticky='EW'
        )
        Checkbutton(frame, text='DIALS (.smv)', variable=self.var_save_dials).grid(
            row=6, column=2, sticky='EW'
        )
        Checkbutton(frame, text='REDp (.mrc)', variable=self.var_save_red).grid(
            row=6, column=3, sticky='EW'
        )
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)
        frame.grid_columnconfigure(3, weight=1)
        frame.pack(side='top', fill='x', padx=10, pady=10)

        # Start/stop
        frame = Frame(self)
        self.CollectionButton = Button(
            frame, text='Start Collection', command=self.start_collection
        )
        self.CollectionButton.grid(row=1, column=0, sticky='EW')

        self.CollectionStopButton = Button(
            frame, text='Stop Collection', command=self.stop_collection, state=DISABLED
        )
        self.CollectionStopButton.grid(row=1, column=1, sticky='EW')

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.pack(side='bottom', fill='x', padx=10, pady=10)

        self.stopEvent = threading.Event()

    def init_vars(self):
        self.var_exposure_time = DoubleVar(value=0.5)
        self.var_unblank_beam = BooleanVar(value=False)

        # 0 => auto (= exposure time)
        self.var_frame_interval = DoubleVar(value=0.0)

        self.var_image_interval = IntVar(value=10)
        self.var_diff_defocus = IntVar(value=1500)
        self.var_enable_image_interval = BooleanVar(value=False)
        self.var_toggle_diff_defocus = BooleanVar(value=False)
        self.var_exposure_time_image = DoubleVar(value=0.01)

        # Default workflow: footfree + calibrated speed + relative span
        self.var_rotation_span_deg = DoubleVar(value=60.0)
        self.var_rotation_speed_deg_s = DoubleVar(value=0.50)
        self.var_use_stage_rotation_calibration = BooleanVar(value=True)

        # Optional explicit path to calib_stage_rotation.yaml
        self.var_stage_rotation_calibration_file = StringVar(value='')

        # Fallback: direct speed setting (0..1) if calibration is disabled/unavailable
        self.var_footfree_speed = DoubleVar(value=0.20)
        self.var_auto_stop_on_target = BooleanVar(value=True)

        self.mode = 'footfree'

        self.var_save_tiff = BooleanVar(value=True)
        self.var_save_xds = BooleanVar(value=True)
        self.var_save_dials = BooleanVar(value=True)
        self.var_save_red = BooleanVar(value=True)

    def start_collection(self):
        self.stopEvent.clear()

        if self.var_toggle_diff_defocus.get():
            self.var_toggle_diff_defocus.set(False)
            self.toggle_diff_defocus()

        self.CollectionStopButton.config(state=NORMAL)
        self.CollectionButton.config(state=DISABLED)

        self.lb_coll1.config(text='Data collection has started (footfree).')
        self.lb_coll2.config(text='Click STOP COLLECTION to end the experiment.')

        self.parent.bind_all('<space>', self.stop_collection)

        params = self.get_params()
        self.q.put(('cred', params))

    def stop_collection(self, event=None):
        self.stopEvent.set()
        self.parent.unbind_all('<space>')

        self.CollectionStopButton.config(state=DISABLED)
        self.CollectionButton.config(state=NORMAL)
        self.lb_coll1.config(text='')
        self.lb_coll2.config(text='')

    def get_params(self):
        params = {
            'exposure_time': self.var_exposure_time.get(),
            'frame_interval': self.var_frame_interval.get(),
            'exposure_time_image': self.var_exposure_time_image.get(),
            'unblank_beam': self.var_unblank_beam.get(),
            'enable_image_interval': self.var_enable_image_interval.get(),
            'image_interval': self.var_image_interval.get(),
            'diff_defocus': self.var_diff_defocus.get(),
            # Rotation params (new)
            'mode': self.mode,
            'rotation_span_deg': self.var_rotation_span_deg.get(),
            'rotation_speed_deg_s': self.var_rotation_speed_deg_s.get(),
            'use_stage_rotation_calibration': self.var_use_stage_rotation_calibration.get(),
            'stage_rotation_calibration_file': self.var_stage_rotation_calibration_file.get()
            or None,
            # Fallback/legacy
            'footfree_speed': self.var_footfree_speed.get(),
            'auto_stop_on_target': self.var_auto_stop_on_target.get(),
            'write_tiff': self.var_save_tiff.get(),
            'write_xds': self.var_save_xds.get(),
            'write_dials': self.var_save_dials.get(),
            'write_red': self.var_save_red.get(),
            'stop_event': self.stopEvent,
        }
        return params

    def browse_calib_file(self):
        fn = askopenfilename(
            title='Select stage rotation calibration YAML',
            filetypes=[('YAML files', '*.yaml *.yml'), ('All files', '*.*')],
        )
        if fn:
            self.var_stage_rotation_calibration_file.set(fn)

    def toggle_interval_buttons(self):
        enable = self.var_enable_image_interval.get()
        if enable:
            self.e_image_interval.config(state=NORMAL)
            self.e_image_exposure.config(state=NORMAL)
            self.e_diff_defocus.config(state=NORMAL)
            self.c_toggle_defocus.config(state=NORMAL)
            self.RelaxButton.config(state=NORMAL)
        else:
            self.e_image_interval.config(state=DISABLED)
            self.e_image_exposure.config(state=DISABLED)
            self.e_diff_defocus.config(state=DISABLED)
            self.c_toggle_defocus.config(state=DISABLED)
            self.RelaxButton.config(state=DISABLED)

    def relax_beam(self):
        difffocus = self.var_diff_defocus.get()
        self.q.put(('relax_beam', {'value': difffocus}))

    def toggle_diff_defocus(self):
        toggle = self.var_toggle_diff_defocus.get()
        difffocus = self.var_diff_defocus.get()
        self.q.put(('toggle_difffocus', {'value': difffocus, 'toggle': toggle}))


def acquire_data_cRED(controller, **kwargs):
    controller.log.info('Start cRED experiment')
    from instamatic.experiments import cRED

    # Filter kwargs for backward compatibility (avoid unexpected keyword errors)
    try:
        sig = inspect.signature(cRED.Experiment.__init__)
        accepted = set(sig.parameters.keys())
        accepted.discard('self')
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        pass

    expdir = controller.module_io.get_new_experiment_directory()
    expdir.mkdir(exist_ok=True, parents=True)

    cexp = cRED.Experiment(
        ctrl=controller.ctrl,
        path=expdir,
        flatfield=controller.module_io.get_flatfield(),
        log=controller.log,
        **kwargs,
    )

    success = cexp.start_collection()
    if not success:
        return

    controller.log.info('Finish cRED experiment')

    if controller.use_indexing_server:
        controller.q.put(('autoindex', {'task': 'run', 'path': cexp.smv_path}))


module = BaseModule(
    name='cred', display_name='cRED', tk_frame=ExperimentalcRED, location='bottom'
)
commands = {'cred': acquire_data_cRED}


if __name__ == '__main__':
    root = Tk()
    ExperimentalcRED(root).pack(side='top', fill='both', expand=True)
    root.mainloop()
