from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from instamatic.tools import *

from scipy.stats import linregress
from instamatic.TEMController import config

import pickle


def get_diffraction_pixelsize(difffocus, cameralength, binsize=1, camera="orius"):
    a,b,c = config.diffraction_pixelsize_fit_parameters
    def f(x, a, b, c):
        return a*(x-c) + b
    
    return f(difffocus, a, b, c) * config.diffraction_pixelsize[cameralength]/binsize


def fit_affine_transformation(a, b, rotation=True, scaling=True, translation=False, shear=False, as_params=False, **x0):
    params = lmfit.Parameters()
    params.add("angle", value=x0.get("angle", 0), vary=rotation, min=-np.pi, max=np.pi)
    params.add("sx"   , value=x0.get("sx"   , 1), vary=scaling)
    params.add("sy"   , value=x0.get("sy"   , 1), vary=scaling)
    params.add("tx"   , value=x0.get("tx"   , 0), vary=translation)
    params.add("ty"   , value=x0.get("ty"   , 0), vary=translation)
    params.add("k1"   , value=x0.get("k1"   , 1), vary=shear)
    params.add("k2"   , value=x0.get("k2"   , 1), vary=shear)
    
    def objective_func(params, arr1, arr2):
        angle = params["angle"].value
        sx    = params["sx"].value
        sy    = params["sy"].value 
        tx    = params["tx"].value
        ty    = params["ty"].value
        k1    = params["k1"].value
        k2    = params["k2"].value
        
        sin = np.sin(angle)
        cos = np.cos(angle)

        r = np.array([
            [ sx*cos, -sy*k1*sin],
            [ sx*k2*sin,  sy*cos]])
        t = np.array([tx, ty])

        fit = np.dot(arr1, r) + t
        return fit-arr2
    
    method = "leastsq"
    args = (a, b)
    res = lmfit.minimize(objective_func, params, args=args, method=method)
    
    lmfit.report_fit(res)
    
    angle = res.params["angle"].value
    sx    = res.params["sx"].value
    sy    = res.params["sy"].value 
    tx    = res.params["tx"].value
    ty    = res.params["ty"].value
    k1    = res.params["k1"].value
    k2    = res.params["k2"].value
    
    sin = np.sin(angle)
    cos = np.cos(angle)
    
    r = np.array([
        [ sx*cos, -sy*k1*sin],
        [ sx*k2*sin,  sy*cos]])
    t = np.array([tx, ty])
    
    if as_params:
        return res.params
    else:
        return r, t
