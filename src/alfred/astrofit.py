import os, re, time
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

# scalerX_v2 = joblib.load(f"{base_dir}/emulators/v2/scalerX_v2.pkl")
# scalerY_v2 = joblib.load(f"{base_dir}/emulators/v2/scalerY_v2.pkl")
# model_v2 = tf.keras.models.load_model(f"{base_dir}/emulators/v2/NNv2_model.keras")
# emu_v2 = {'scalerX': scalerX_v2,
#     'scalerY': scalerY_v2,
#     'model': model_v2}

scalerX_v3 = joblib.load(f"{base_dir}/emulators/nn_v3/scalerX.pkl")
scalerY_v3 = joblib.load(f"{base_dir}/emulators/nn_v3/scalerY.pkl")
model_v3 = tf.keras.models.load_model(f"{base_dir}/emulators/nn_v3/model.keras", safe_mode=False)
emu_v3 = {'scalerX': scalerX_v3,
    'scalerY': scalerY_v3,
    'model': model_v3}

scalerX_v3p1 = joblib.load(f"{base_dir}/emulators/nn_v3.1/scalerX.pkl")
scalerY_v3p1 = joblib.load(f"{base_dir}/emulators/nn_v3.1/scalerY.pkl")
model_v3p1 = tf.keras.models.load_model(f"{base_dir}/emulators/nn_v3.1/model.keras", safe_mode=False)
emu_v3p1 = {'scalerX': scalerX_v3p1,
    'scalerY': scalerY_v3p1,
    'model': model_v3p1}

scalerX_v4 = joblib.load(f"{base_dir}/emulators/nn_v4_CVadded/scalerX.pkl")
scalerY_v4 = joblib.load(f"{base_dir}/emulators/nn_v4_CVadded/scalerY.pkl")
model_v4 = tf.keras.models.load_model(f"{base_dir}/emulators/nn_v4_CVadded/model.keras", safe_mode=False)
emu_v4 = {'scalerX': scalerX_v4,
    'scalerY': scalerY_v4,
    'model': model_v4}

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
            Planck=Planck, Planck_err=Planck_err, addPlanck=False,
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

        passPlanck =  ((Planck - 2.0*Planck_err) <= tau) & (tau <= (Planck + 2.0*Planck_err))

    passes = pass1d & pass2d & passPlanck

    return np.where(passes, 0, -np.inf)

def fill(guess, truths, which_params):
    if guess.ndim == 1:
            guess = guess[None,:]
        
    theta_dict = cp.deepcopy(truths)
    # this seems complicated but it works even when the listed params are out of order

    if np.any(which_params == 'all'):
        which_params = list(truths.keys())

    theta = np.zeros((guess.shape[0], len(truths)))

    for i, key in enumerate(truths.keys()):
        if key in which_params:
            theta[:,i] = guess[:,list(which_params).index(key)]
        else:
            theta[:,i] = np.ones(guess.shape[0])
            theta[:,i] *= truths[key]

    return theta

def lnprob(guess, model, data, err, truths, priors,
            which_params='all', vectorize=False,
            priors2d=priors2d, add2d=True,
            Planck=Planck, Planck_err=Planck_err, addPlanck=False):

    theta = fill(guess, truths, which_params)
    
    lp = lnprior(theta, truths, priors,
                priors2d=priors2d, add2d=add2d,
                Planck=Planck, Planck_err=Planck_err, addPlanck=addPlanck)

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
    def __init__(self,
                    config,
                    emu=None,
                    priors=priors,
                    priors2d=priors2d,
                    Planck=Planck,
                    Planck_err=Planck_err,
                    base_dir=base_dir,
                    dir=None,
                    verbose=False):
        
        self.config = config
        self.verbose = verbose
        self.title = self.config['title']
        if dir is None:
            self.mcmc_dir = f"{base_dir}/inference/mcmc_runs/{self.title}"
        elif dir is not None:
            self.mcmc_dir = f"{dir}/{self.title}"

        self.telescope = self.config['survey']
        self.lmin = self.config['lmin']
        self.lmax = self.config['lmax']
        if self.config['emu'] == 'v2':
            self.emu = emu_v2
        elif self.config['emu'] == 'v3':
            self.emu = emu_v3
        elif self.config['emu'] == 'v3.1':
            self.emu = emu_v3p1
        elif self.config['emu'] == 'emu':
            self.emu = emu
        self.addnoise = self.config['addnoise']

        self.which_params = self.config['which_params_to_fit']
        self.priors = priors
        self.add2d = self.config['add2d']
        if self.add2d:
            self.priors2d = priors2d
        self.addPlanck = self.config['addPlanck']
        if self.addPlanck:
            self.Planck = Planck
            self.Planck_err = Planck_err
        else:
            self.Planck = None
            self.Planck_err = None
        self.vectorize = self.config['vectorize']

        self.nwalkers = self.config['nwalkers']
        self.burnin = self.config['burnin']
        self.nsteps = self.config['nsteps']
        self.ndim = len(self.which_params)
        self.rescale_cov = self.config['rescale_cov']

    def init_data(self):
        if self.verbose:
            print(f'-----------------------------------------')
            print(f"Now initialising data for mcmc run")
            print(f'-----------------------------------------')

        self.truths = {}
        for pname in df.columns:
            self.truths[pname] = self.config[pname]

        self.theta_true = np.asarray(list(self.truths.values()))
        self.lmask = np.where((self.lmin < ells) & (ells < self.lmax))[0]
        self.datapoints = emulator.kemu(self.theta_true, **self.emu, log_data=True)
        self.err_cov = surveys.error_cov(ells, self.datapoints, surveys.telescopes[self.telescope])
        self.err =np.sqrt(np.diag(self.err_cov))


        if self.addnoise:
            if self.verbose:
                print(f"adding noise to simulated data assuming {self.telescope} specifications...")

            self.datapoints = self.datapoints + np.random.normal(scale=self.err)

        self.datapoints = self.datapoints[self.lmask]
        self.err = self.err[self.lmask]

        if self.verbose:
            print(f"simulating data with the parameter values:")
            print(self.truths)
            print(f"using emulator version {self.emu}...")


    def init_run(self):
        if self.verbose:
            print(f"Now running initialisation of MCMC run")
            print(f'-----------------------------------------')
            print(f"putting mcmc chains in {self.mcmc_dir}...")
            print()
            print(f"Running the mcmc for {self.title} on the params:")
            print(f"\t{self.which_params}...")

        os.makedirs(self.mcmc_dir)

        def model(p, lmask=self.lmask, emu=self.emu):
            if p.ndim == 1:
                return emulator.kemu(p, **emu)[lmask]
            elif p.ndim == 2:
                return emulator.kemu(p, **emu)[:, lmask]


        self.model = model
        self.lnprob_prepped = lambda params: lnprob(params, self.model, self.datapoints,
                                            self.err, self.truths, self.priors,
                                            which_params=self.which_params, priors2d=self.priors2d, add2d=self.add2d,
                                            Planck=self.Planck, Planck_err=self.Planck_err, addPlanck=self.addPlanck,
                                            vectorize=self.vectorize)

    def start_run(self, save=True):
        if self.verbose:
            print(f"Now starting MCMC run")
            print(f'-----------------------------------------')
            print(f"saving data to ...{self.mcmc_dir}/data")

        np.savez(f"{self.mcmc_dir}/data", truths=self.truths, datapoints=self.datapoints, err=self.err)

        if self.verbose:
            plt.plot(ells[self.lmask], self.model(self.theta_true))
            plt.errorbar(ells[self.lmask], self.datapoints.flatten(), marker='.', ls='', yerr=self.err)

        chains_fn = f"{self.mcmc_dir}/saved_chains.h5"
        save_progress = zeus.callbacks.SaveProgressCallback(chains_fn, ncheck=100)
        autocorr_check = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=50, discard=0.5)
        R_check = zeus.callbacks.SplitRCallback(ncheck=100, epsilon=0.01, nsplits=2, discard=0.5)
        miniter_check = zeus.callbacks.MinIterCallback(nmin=500)

        p0 = pass_Planck[:12]
        # p0 = pass_prior[50:62]
        # p0[6] = pass_prior[52]
        #p0 = pass_prior[100:112]
        p0 = p0[:,np.where(np.isin(labels, self.which_params))[0]]
                
        if self.verbose:
            print(f"fitting tau prior is {self.addPlanck}...")
            print(f"evaluating likelihood function in vector mode is {self.vectorize}...")

            print('========================================================')

            print('Okay, here we go!')

        start_time = time.time()

        sampler = zeus.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob_prepped,
                                        vectorize=self.vectorize)

        sampler.run_mcmc(p0, self.burnin)
        burnin_samples = sampler.get_chain()
        start = burnin_samples[-1] # Get the burnin samples

        end_time = time.time()

        if self.verbose:
            print('--------------------------------------------------------')
            print(f'Burn in phase took {(end_time - start_time) / 60:.3f} minutes...')
            print(f'Starting proper run...')
            print('--------------------------------------------------------')

        start_time = time.time()

        sampler = zeus.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob_prepped,
                #      args=[datapoints, err, theta_true, lmask, which_params],
                        moves=zeus.moves.GlobalMove(self.rescale_cov), vectorize=self.vectorize)
        
        sampler.run_mcmc(start, self.nsteps, callbacks=[save_progress, autocorr_check, R_check, miniter_check])

        end_time = time.time()

        if self.verbose:
            print(f'finished MCMC in {(end_time - start_time) / (60 * 60):.2} hours')
            if save:
                print(f'saving files in {self.mcmc_dir}...')

        if save:
            np.save(f'{self.mcmc_dir}/burnin', burnin_samples)
            np.save(f'{self.mcmc_dir}/tau', autocorr_check.estimates)
            np.save(f'{self.mcmc_dir}/R', R_check.estimates)

        if self.verbose:
            print('Done, YAY!')

        return sampler

