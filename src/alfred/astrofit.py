import os, re
import copy as cp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
import corner
import zeus
from scipy.integrate import cumulative_trapezoid
from astropy import cosmology, units, constants

import joblib
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import alfred.emulator as emulator
import alfred.surveys as surveys
import alfred.keras_xe_emul as keras_xe_emul
from alfred.parameters import *
from alfred.utils import get_sims


telescopes ={
    'SO-LAT': {'fsky':0.4, 'fwhm':1.5, 'noise':6.0},
    'SO-SAT': {'fsky':0.1, 'fwhm':10.0, 'noise':2.5},
    'CMB-S4': {'fsky':0.6, 'fwhm':1.0, 'noise': 1.4142},
    'CMB-HD': {'fsky':0.5, 'fwhm':0.5, 'noise':2.7},
}


delta_ell = np.mean(np.diff(ells))
Planck = 0.054
Planck_err = 0.007
priors2d = np.load(f'{base_dir}/inference/priors/2dpriors.npz')

df = pd.read_pickle(f"{base_dir}/metadata/LoReLi_database_loggedparams.pkl")
xe_histories = np.load(f'{base_dir}/metadata/ion_histories_full.npz', allow_pickle=True)
xe_histories = xe_histories['arr_0'].item()

labels = df.columns
priors = [(df[p].to_numpy().min(), df[p].to_numpy().max()) for p in labels]
sims = cp.deepcopy(df.index.to_numpy())
features = cp.deepcopy(df.to_numpy())

pass_prior = np.load(f'{base_dir}/inference/priors/pass_prior.npy')
pass_Planck = np.load(f'{base_dir}/inference/priors/pass_Planckprior.npy')

scalerX_v2 = joblib.load(f"{base_dir}/emulators/LoReLi_settings/scalerX_LoReLi_style.pkl")
scalerY_v2 = joblib.load(f"{base_dir}/emulators/LoReLi_settings/scalerY_LoReLi_style.pkl")
model_v2 = tf.keras.models.load_model(f"{base_dir}/emulators/LoReLi_settings/NN_LoReLi_style_model.keras")
emu_v2 = {'scalerX': scalerX_v2,
    'scalerY': scalerY_v2,
    'model': model_v2}

scalerX_v3 = joblib.load(f"{base_dir}/emulators/nn_v3/scalerX.pkl")
scalerY_v3 = joblib.load(f"{base_dir}/emulators/nn_v3/scalerY.pkl")
model_v3 = tf.keras.models.load_model(f"{base_dir}/emulators/nn_v3/model.keras", safe_mode=False)
emu_v3 = {'scalerX': scalerX_v3,
    'scalerY': scalerY_v3,
    'model': model_v3}

ztau = np.linspace(0,20,1000)

def xe2tau(z, xe):
        """
        Computes redshift evolution of the model's optical depth.

        Parameters
        ----------
            z: (array of) float(s)
                Redshift range used to compute the optical depth.
        """
        cos = cosmology.FlatLambdaCDM(
            H0=h * 100, Tcmb0=T_cmb, Ob0=Ob_0, Om0=Om_0
        )
        z = np.sort(z)
       # xe = np.sort(xe)[::-1]

        integ = constants.c.value * constants.sigma_T.value * nh * xe / cos.H(z).si.value * (1+z)**2
        # tofz = cumulative_trapezoid(integ[::-1], z, initial=0)[::-1]
        tofz = cumulative_trapezoid(integ, z, initial=0)

        return tofz


def lnprior(theta_fit, truths=None, priors=priors, which_params=None,
                    priors2d=priors2d, add2d=True,
                    Planck=Planck, addPlanck=False,
                    labels=labels, verbose=False):

        
    theta_dict = dict(zip(labels, truths))
    for i, key in enumerate(which_params):
        theta_dict[key] = theta_fit[i]

    theta = list(theta_dict.values())

    for i, p in enumerate(priors):
        low, high = p
        if not (low <= theta[i] <= high):
            if verbose:
                print(f'failed 1d check on {i}th parameter')
            return -np.inf
        
    if add2d:
        Mmin = theta[3] # CAREFUL! currently hardcoded (also log10)
        tau = theta[2]  # CAREFUL! currently hardcoded (also log10)

        below = priors2d['m'] * Mmin + priors2d['b_below']
        above = priors2d['m'] * Mmin + priors2d['b_above']

        if not(below <= tau <= above):
            if verbose:
                print('failed 2d check')

            return -np.inf
        
    if addPlanck:
        print('I am checking the tau prior!')
        ztau = np.linspace(0,20,100)
        xemu = keras_xe_emul.xe_emul_array(ztau, np.asarray(theta), plot=False)
        tau = xe2tau(ztau, xemu)[-1]

        if not ((Planck - 2 * Planck_err) <= tau <=(Planck + 2 * Planck_err)):
            if verbose:
                print(f'failed Planck tau check')

            return -np.inf

    return 0.

def lnprob(theta, data, err, truths, lmask, which_params,
            priors=priors, priors2d=priors2d, add2d=True,
            Planck=Planck, addPlanck=False, emu=None, sn=None,
            labels=labels):
    
    lp = lnprior(theta, truths=truths, priors=priors, which_params=which_params, add2d=add2d,
                    Planck=Planck, addPlanck=addPlanck)
    if not np.isfinite(lp):
        return -np.inf#, 0.
    ln = lnlike(theta, data, err, truths, lmask, which_params, emu=emu, sn=sn)
    return lp + ln #, model

def lnlike(theta_fit, data, err, truths, lmask, which_params, emu=None, sn=None,
           labels=labels):
    
    if lmask is None:
        lmask = range(data.size) # want every ell in this case

    if not sn:
        theta_dict = dict(zip(which_params, theta_fit))

        if truths is not None:
            theta = cp.deepcopy(truths)
        elif truths is None:
            theta = cp.deepcopy(theta_fit)

        for i, p in enumerate(priors):
            if labels[i] in which_params:
             #   print(f'{labels[i]} is in')
                theta[i] = theta_dict[labels[i]]

        guess_model = emulator.kemu(theta, **emu)
        
        return -0.5 * np.sum((data[lmask] - guess_model[lmask]) ** 2.0 / err[lmask]**2.0)

    elif sn:
        fn_L = f'{home_dir}/spectra/kSZ/LoReLi/nells30/kSZ_LoReLi_simu{sn}.npz'
        ksz = np.load(fn_L, allow_pickle=True)
        signal = ksz['kSZ']

        return -0.5 * np.sum((data[lmask] - signal[lmask]) ** 2.0 / err[lmask]**2.0)
    

def chi2_contribution(theta, data, err, lmask=None, emu=None, sn=None):
    if lmask is None:
        lmask = range(data.size) # want every ell in this case
    guess_model = emulator.kemu(theta, **emu)

    return -0.5 * (data[lmask] - guess_model[lmask]) ** 2.0 / err[lmask]**2.0
