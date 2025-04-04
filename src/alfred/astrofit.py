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
priors = np.stack([df.to_numpy().min(axis=0), df.to_numpy().max(axis=0)]).T
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


def lnprior(theta, truths, priors,
            priors2d=priors2d, add2d=True,
            Planck=Planck, addPlanck=False,
            verbose=False):

    pass1d = np.all((priors[:,0] <= theta) & (priors[:,1] >= theta), axis=1)
    pass2d = np.ones_like(pass1d, dtype=bool)
    passPlanck = np.ones_like(pass1d, dtype=bool)

    if add2d:
        Mmin = theta[:,3] # CAREFUL! currently hardcoded (also log10)
        tau = theta[:,2]  # CAREFUL! currently hardcoded (also log10)

        below = priors2d['m'] * Mmin + priors2d['b_below']
        above = priors2d['m'] * Mmin + priors2d['b_above']

        pass2d =  (below <= tau) & (tau <= above)
    
    if addPlanck:
        if verbose:
            print('I am checking the tau prior!')
        xemu = keras_xe_emul.xe_emul_array(ztau, theta, plot=False)
        tau = xe2tau(ztau, xemu)[:,-1]

        passPlanck =  ((Planck - 2*Planck_err) <= tau) & (tau <= (Planck + 2*Planck_err))

    passes = pass1d & pass2d & passPlanck

    return np.where(passes, 0, -np.inf)

def lnprob(guess, model, data, err, truths, priors,
            which_params='all', vectorize=False,
            priors2d=priors2d, add2d=True,
            Planck=Planck, addPlanck=False):

    if guess.ndim == 1:
        guess = guess[None,:]
    
    theta_dict = cp.deepcopy(truths)
    # this seems complicated but it works even when the listed params are out of order

    if np.any(which_params == 'all'):
        which_params = list(truths.keys())

    theta = np.zeros((guess.shape[0], len(truths)))

    for i, key in enumerate(truths.keys()):
        if key in which_params:
            theta[:,i] = guess[:,which_params.index(key)]
        else:
            theta[:,i] = np.ones(guess.shape[0])
            theta[:,i] *= truths[key]

    lp = lnprior(theta, truths, priors,
                priors2d=priors2d, add2d=add2d,
                Planck=Planck, addPlanck=addPlanck)

    
    # if not np.isfinite(lp):
    #     return -np.inf#, 0.
    ln = lnlike(theta, model, data, err)

    if not vectorize:
        return (lp + ln)[0]

    return lp + ln #, model

def lnlike(theta, model, data, err):

    test = model(theta)
    return -0.5 * ((data - test) ** 2.0 / err**2.0).sum(axis=1)


def chi2_contribution(theta, data, err, lmask=None, emu=None, sn=None):
    if lmask is None:
        lmask = range(data.size) # want every ell in this case
    guess_model = emulator.kemu(theta, **emu)

    return -0.5 * (data[lmask] - guess_model[lmask]) ** 2.0 / err[lmask]**2.0

class MCMC:
    def __init__(self, config, base_dir=base_dir):

        self.mcmc_dir = f"{base_dir}/inference/mcmc_runs/{config['title']}"

    def init_run(self):
        if self.verbose:
            print(f"Now running initialisation of MCMC run")
            print(f'-----------------------------------------')
            print(f"putting mcmc chains in {self.mcmc_dir}...")

        os.makedirs(self.mcmc_dir)


        lmask = np.where((config['lmin'] < ells) & (ells < config['lmax']))
    which_params = config['which_params_to_fit']

    if which_params == 'all':
        which_params = df.columns

    theta_dict = {}
    for pname in df.columns:
        theta_dict[pname] = config[pname]

    print(f"simulating data with the parameter values:")
    print(theta_dict)
    # theta_true =  df[df.columns].mean().to_numpy()
    theta_true = np.asarray(list(theta_dict.values()))
