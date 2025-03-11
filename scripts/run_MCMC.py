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
import alfred.analyse as analyse
import alfred.surveys as surveys

import joblib
import tensorflow as tf


from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

from alfred.parameters import *
from alfred.astrofit import *


import argparse

def main():
    parser = argparse.ArgumentParser(description="Load a config file for setting up an mcmc")
    parser.add_argument("--config", type=str, help="Name of config file (toml format)")
   
    # Parse arguments
    args = parser.parse_args()

    print("Here we go!!!")

    config = toml.load(f"{home_dir}/alfred/scripts/{args.config}")
    print(f"Now initialising mcmc run {config['title']}...")


    mcmc_dir = f"{base_dir}/inference/mcmc_runs/{config['title']}"
    print(f"Putting mcmc chains in {mcmc_dir}")
    os.makedirs(mcmc_dir)

    delta_ell = np.diff(ells).mean()
    lmask = np.where((config['lmin'] < ells) & (ells < config['lmax']))
    which_params = config['which_params_to_fit']

    if which_params == 'all':
        which_params = df.columns

    theta_dict = {}
    for pname in df.columns:
        theta_dict[pname] = config[pname]

    print(f"data pvals are:")
    print(theta_dict)
    # theta_true =  df[df.columns].mean().to_numpy()
    theta_true = np.asarray(list(theta_dict.values()))
    
    datapoints = emulator.kemu(theta_true, **emu, log_data=True)

    err_cov = surveys.error_cov(ells, datapoints, surveys.telescopes[config['survey']])
    err =np.sqrt(np.diag(err_cov))

    lnprob_prepped = lambda params: lnprob(params, datapoints, err, theta_true, None, which_params,
            priors=priors, priors2d=priors2d, add2d=config['add2d'],
            Planck=Planck, addPlanck=config['addPlanck'], emu=emu, sn=None,
            labels=labels)
    
    chains_fn = f"{mcmc_dir}/saved_chains.h5"
    save_progress = zeus.callbacks.SaveProgressCallback(chains_fn, ncheck=100)
    autocorr_check = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=50, discard=0.5)
    R_check = zeus.callbacks.SplitRCallback(ncheck=100, epsilon=0.01, nsplits=2, discard=0.5)
    miniter_check = zeus.callbacks.MinIterCallback(nmin=500)

    nwalkers = config['nwalkers']
    burnin = config['burnin']
    nsteps = config['nsteps']
    ndim = len(which_params)

    p0 = pass_Planck[:12]
    # p0 = pass_prior[50:62]
    # p0[6] = pass_prior[52]
    #p0 = pass_prior[100:112]
    p0 = p0[:,np.where(np.isin(labels, which_params))[0]]

    print(p0)
            
    print('Okay, here we go!')
    print(f"Running the mcmc for {config['title']} on the params:")
    print(f"\t{config['which_params_to_fit']}...")

    start_time = time.time()

    sampler = zeus.EnsembleSampler(nwalkers, ndim, lnprob_prepped)

    sampler.run_mcmc(p0, burnin)
    burnin_samples = sampler.get_chain()
    start = burnin_samples[-1] # Get the burnin samples

    sampler = zeus.EnsembleSampler(nwalkers, ndim, lnprob_prepped,
              #      args=[datapoints, err, theta_true, lmask, which_params],
                    moves=zeus.moves.GlobalMove(config['rescale_cov']))
    
    sampler.run_mcmc(start, nsteps, callbacks=[save_progress, autocorr_check, R_check, miniter_check])

    end_time = time.time()

    print(f'finished MCMC in {(end_time - start_time) / (60 * 60)} hours, saving files in {mcmc_dir}...')

    np.save(f'{mcmc_dir}/burnin', burnin_samples)
    np.save(f'{mcmc_dir}/tau', autocorr_check.estimates)
    np.save(f'{mcmc_dir}/R', R_check.estimates)

    print('Done, YAY!')

if __name__ == "__main__":
    main()
