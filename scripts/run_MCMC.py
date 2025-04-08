import os
import re
import time
import copy as cp
import argparse
import toml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import zeus

from scipy.interpolate import PchipInterpolator, CubicSpline

import alfred.utils as utils
import alfred.emulator as emulator
import alfred.KSZ as KSZ
import alfred.peefit as peefit
import alfred.surveys as surveys

import joblib
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import warnings

from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

from alfred.parameters import *
from alfred.astrofit import *


import argparse

def main():
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Load a config file for setting up an mcmc")
    parser.add_argument("--config", type=str, help="Name of config file (toml format)")
   
    # Parse arguments
    args = parser.parse_args()

    print('========================================================')
    print("HELLO, WELCOME TO YOUR MCMC RUN!")
    print('========================================================')

    config = toml.load(f"{home_dir}/alfred/scripts/{args.config}")
    print(f"Now initialising mcmc run {config['title']}...")
    print()

    mcmc_dir = f"{base_dir}/inference/mcmc_runs/{config['title']}"
    print(f"putting mcmc chains in {mcmc_dir}...")
    os.makedirs(mcmc_dir)

    delta_ell = np.diff(ells).mean()
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

    if config['emu'] == 'v2':
        emu = emu_v2
    elif config['emu'] == 'v3':
        emu = emu_v3
    elif config['emu'] == 'v3.1':
        emu = emu_v3p1
    elif config['emu'] == 'v4':
        emu = emu_v4

    print(f"using emulator version {config['emu']}...")
    datapoints = emulator.kemu(theta_true, **emu, log_data=True)
    truths = dict(zip(df.columns, theta_true))

    err_cov = surveys.error_cov(ells, datapoints, surveys.telescopes[config['survey']])
    err =np.sqrt(np.diag(err_cov))

    if config['addnoise'] == True:
        print(f"adding noise to simulated data...")
        datapoints = datapoints + np.random.normal(scale=err)

    print()
    print(f"Running the mcmc for {config['title']} on the params:")
    print(f"\t{config['which_params_to_fit']}...")

    model = lambda p: emulator.kemu(p, **emu)
    lnprob_prepped = lambda params: lnprob(params, model, datapoints, err, truths, priors,
            which_params=which_params,
            priors2d=priors2d, add2d=config['add2d'],
            Planck=Planck, addPlanck=config['addPlanck'], vectorize=config['vectorize'])
    
    chains_fn = f"{mcmc_dir}/saved_chains.h5"
    save_progress = zeus.callbacks.SaveProgressCallback(chains_fn, ncheck=100)
    autocorr_check = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=50, discard=0.5)
    R_check = zeus.callbacks.SplitRCallback(ncheck=100, epsilon=0.01, nsplits=2, discard=0.5)
    miniter_check = zeus.callbacks.MinIterCallback(nmin=500)

    nwalkers = config['nwalkers']
    burnin = config['burnin']
    nsteps = config['nsteps']
    ndim = len(which_params)

    random_samples = []
    for i, p in enumerate(priors):
        low, high = p
        s = np.random.uniform(low, high, 1000) # not robust!
        random_samples.append(s)

    random_samples = np.column_stack(random_samples)

    p0 = random_samples[lnprior(random_samples, truths, priors,
                        add2d=True, addPlanck=True, verbose=True) == 0][:12]
    
    p0 = p0[:,np.where(np.isin(labels, which_params))[0]]
    
    np.savez(f"{mcmc_dir}/data", truths=truths, datapoints=datapoints, err=err, p0=p0)
            
    print(f"fitting tau prior is {config['addPlanck']}...")
    print(f"evaluating likelihood function in vector mode is {config['vectorize']}...")

    print('========================================================')

    print('Okay, here we go!')

    start_time = time.time()

    sampler = zeus.EnsembleSampler(nwalkers, ndim, lnprob_prepped, vectorize=config['vectorize'])

    sampler.run_mcmc(p0, burnin)
    burnin_samples = sampler.get_chain()
    start = burnin_samples[-1] # Get the burnin samples

    end_time = time.time()

    print('--------------------------------------------------------')
    print(f'Burn in phase took {(end_time - start_time) / 60:.3f} minutes...')
    print(f'Starting proper run...')
    print('--------------------------------------------------------')

    start_time = time.time()

    sampler = zeus.EnsembleSampler(nwalkers, ndim, lnprob_prepped,
              #      args=[datapoints, err, theta_true, lmask, which_params],
                    moves=zeus.moves.GlobalMove(config['rescale_cov']), vectorize=config['vectorize'])
    
    sampler.run_mcmc(start, nsteps, callbacks=[save_progress, autocorr_check, R_check, miniter_check])

    end_time = time.time()

    print(f'finished MCMC in {(end_time - start_time) / (60 * 60):.3} hours, saving files in {mcmc_dir}...')

    np.save(f'{mcmc_dir}/burnin', burnin_samples)
    np.save(f'{mcmc_dir}/tau', autocorr_check.estimates)
    np.save(f'{mcmc_dir}/R', R_check.estimates)

    print('Done, YAY!')

if __name__ == "__main__":
    main()
