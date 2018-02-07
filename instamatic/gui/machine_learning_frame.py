from __future__ import print_function
from __future__ import absolute_import
from Tkinter import *
from ttk import *
import tkFileDialog

from instamatic.formats import read_image
import numpy as np
import matplotlib.pyplot as plt
from .mpl_frame import ShowMatplotlibFig

import os

def treeview_sort_column(tv, col, reverse):
    """https://stackoverflow.com/a/22032582"""
    l = [(tv.set(k, col), k) for k in tv.get_children('')]
    l.sort(key=lambda t: float(t[0]), reverse=reverse)

    for index, (val, k) in enumerate(l):
        tv.move(k, '', index)

    tv.heading(col, command=lambda: treeview_sort_column(tv, col, not reverse))

class MachineLearningFrame(object, LabelFrame):
    """docstring for MachineLearningFrame"""
    def __init__(self, parent):
        LabelFrame.__init__(self, parent, text="Neural Network")
        self.parent = parent

        self.init_vars()

        frame = Frame(self)
        columns = ('frame', 'number', 'quality', 'size', 'x', 'y')
        self.tv = tv = Treeview(frame, columns=columns, show='headings')

        # for loop does not work here for some reason...
        tv.heading('frame', text='Frame', command=lambda: treeview_sort_column(tv, 'frame', False))
        tv.column('frame', anchor='center', width=15)
        tv.heading('number', text='Number', command=lambda: treeview_sort_column(tv, 'number', False))
        tv.column('number', anchor='center', width=15)
        tv.heading('quality', text='Quality', command=lambda: treeview_sort_column(tv, 'quality', False))
        tv.column('quality', anchor='center', width=15)
        tv.heading('size', text='Size', command=lambda: treeview_sort_column(tv, 'size', False))
        tv.column('size', anchor='center', width=15)
        tv.heading('x', text='X', command=lambda: treeview_sort_column(tv, 'x', False))
        tv.column('x', anchor='center', width=15)
        tv.heading('y', text='Y', command=lambda: treeview_sort_column(tv, 'y', False))
        tv.column('y', anchor='center', width=15)

        tv.grid(sticky = (N,S,W,E))
        frame.grid_rowconfigure(0, weight = 1)
        frame.grid_columnconfigure(0, weight = 1)
        frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        frame = Frame(self)

        self.BrowseButton = Button(frame, text="Load data", command=self.load_table)
        self.BrowseButton.grid(row=1, column=0, sticky="EW")

        self.ShowButton = Button(frame, text="Show image", command=self.show_image)
        self.ShowButton.grid(row=1, column=1, sticky="EW")

        self.GoButton = Button(frame, text="Go to crystal", command=self.go_to_crystal)
        self.GoButton.grid(row=1, column=2, sticky="EW")
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.pack(side="bottom", fill="x", padx=10, pady=10)


    def init_vars(self):
        self.fns = {}
        self.var_directory = StringVar(value=os.path.realpath("."))

    def set_trigger(self, trigger=None, q=None):
        self.triggerEvent = trigger
        self.q = q

    def load_table(self):
        fn = tkFileDialog.askopenfilename(parent=self.parent, initialdir=self.var_directory.get(), title="Select crystal data")
        if not fn:
            return
        fn = os.path.realpath(fn)

        import csv

        with open(fn, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                fn, frame, number, quality, size, stage_x, stage_y = row
                self.tv.insert('', 'end', text=fn, values=(frame, number, quality, size, stage_x, stage_y))
                self.fns[(int(frame), int(number))] = fn

        print("CSV data `{}` loaded".format(fn))

    def go_to_crystal(self):
        row = self.tv.item(self.tv.focus())
        try:
            frame, number, quality, size, stage_x, stage_y = row["values"]
        except ValueError:  # no row selected
            print("No row selected")
            return

        self.q.put(("ctrl", { "task": "stageposition", 
                        "x": float(stage_x),
                        "y": float(stage_y) } ))
        self.triggerEvent.set()

    def show_image(self):
        row = self.tv.item(self.tv.focus())
        try:
            frame, number, quality, size, stage_x, stage_y = row["values"]
        except ValueError:  # no row selected
            print("No row selected")
            return

        fn = row["text"]

        up = os.path.dirname
        root = up(up(fn))
        name = os.path.basename(fn).replace("_{:04d}.".format(number), ".")
        image_fn = os.path.join(os.path.join(root, "images"), name)

        data, data_h = read_image(fn)
        img, img_h = read_image(image_fn)

        cryst_x, cryst_y = img_h["exp_crystal_coords"][number]

        fig = plt.figure()
        ax1 = plt.subplot(121, title="Image\nframe={} | number={}\nsize={} | x,y=({}, {})".format(frame, number, size, stage_x, stage_y), aspect="equal")
        ax2 = plt.subplot(122, title="Diffraction pattern\nquality={}".format(quality, aspect="equal"))
        
        img = np.rot90(img, k=3)                            # img needs to be rotated to match display
        cryst_y, cryst_x = img.shape[0] - cryst_x, cryst_y  # rotate coordinates

        coords = img_h["exp_crystal_coords"]
        for c_x, c_y in coords:
            c_y, c_x = img.shape[0] - c_x, c_y
            ax1.plot(c_y, c_x, marker="+", color="blue",  mew=2)
        
        ax1.imshow(img, vmax=np.percentile(img, 99.5))
        ax1.plot(cryst_y, cryst_x, marker="+", color="red",  mew=2)

        ax2.imshow(data, vmax=np.percentile(data, 99.5))

        ShowMatplotlibFig(self, fig, title=fn)


if __name__ == '__main__':
    root = Tk()
    MachineLearningFrame(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
