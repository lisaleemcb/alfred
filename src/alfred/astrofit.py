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
from astropy import cosmology, units, constants

import joblib
import tensorflow as tf

import alfred.emulator as emulator
import alfred.surveys as surveys
from alfred.parameters import *
from alfred.utils import get_sims


telescopes ={
    'SO-LAT': {'fsky':0.4, 'fwhm':1.5, 'noise':6.0},
    'SO-SAT': {'fsky':0.1, 'fwhm':10.0, 'noise':2.5},
    'CMB-S4': {'fsky':0.6, 'fwhm':1.0, 'noise': 1.4142},
    'CMB-HD': {'fsky':0.5, 'fwhm':0.5, 'noise':2.7},
}

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

scalerX = joblib.load(f"{base_dir}/emulators/LoReLi_settings/scalerX_LoReLi_style.pkl")
scalerY = joblib.load(f"{base_dir}/emulators/LoReLi_settings/scalerY_LoReLi_style.pkl")
model = tf.keras.models.load_model(f"{base_dir}/emulators/LoReLi_settings/NN_LoReLi_style_model.keras")
emu = {'scalerX': scalerX,
    'scalerY': scalerY,
    'model': model}



def lnprior(theta_fit, truths=None, priors=priors, which_params=None,
                    priors2d=priors2d, add_2d=True,
                    Planck=Planck, add_Planck=False,
                    labels=labels, verbose=False):
        
    theta_dict = dict(zip(which_params, theta_fit))
    theta = cp.deepcopy(truths)

    for i, p in enumerate(priors):
        if labels[i] in which_params:
            theta[i] = theta_dict[labels[i]]
        low, high = p
        if not (low <= theta[i] <= high):
            if verbose:
                print(f'failed 1d check on {i}th parameter')
            return -np.inf
        
    if add_2d:
        Mmin = theta[3] # CAREFUL! currently hardcoded (also log10)
        tau = theta[2]  # CAREFUL! currently hardcoded (also log10)

        below = priors2d['m'] * Mmin + priors2d['b_below']
        above = priors2d['m'] * Mmin + priors2d['b_above']

        if not(below <= tau <= above):
            if verbose:
                print('failed 2d check')

            return -np.inf
        
    if add_Planck:
        tau = .054
        if not ((Planck - Planck_err) <= tau <=(Planck + Planck_err)):
            if verbose:
                print(f'failed Planck tau check')

            return -np.inf

    return 0.

def lnprob(theta, data, err, truths, lmask, which_params,
            priors=priors, priors2d=priors2d, add_2d=True,
            Planck=Planck, add_Planck=False, emu=emu, sn=None,
            labels=labels):
    
    lp = lnprior(theta, truths=truths, priors=priors, which_params=which_params, add_2d=add_2d,
                    Planck=Planck, add_Planck=add_Planck)
    if not np.isfinite(lp):
        return -np.inf#, 0.
    ln = lnlike(theta, data, err, truths, lmask, which_params, sn=sn)
    return lp + ln #, model

def lnlike(theta_fit, data, err, truths, lmask, which_params, emu=emu, sn=None,
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

        guess_model =emulator.kemu(theta, **emu)
        
        return -0.5 * np.sum((data[lmask] - guess_model[lmask]) ** 2.0 / err[lmask]**2.0)

    elif sn:
        fn_L = f'{home_dir}/spectra/kSZ/LoReLi/nells30/kSZ_LoReLi_simu{sn}.npz'
        ksz = np.load(fn_L, allow_pickle=True)
        signal = ksz['kSZ']

        return -0.5 * np.sum((data[lmask] - signal[lmask]) ** 2.0 / err[lmask]**2.0)
    

def chi2_contribution(theta, data, err, lmask=None, emu=emu, sn=None):
    if lmask is None:
        lmask = range(data.size) # want every ell in this case
    guess_model = emulator.kemu(theta, **emu)

    return -0.5 * (data[lmask] - guess_model[lmask]) ** 2.0 / err[lmask]**2.0
