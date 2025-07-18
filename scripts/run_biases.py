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

import keras
import joblib
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import warnings

from types import SimpleNamespace
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

from alfred.parameters import *
from alfred.astrofit import *


import argparse

def main():
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="sets what experiment to use.")
    parser.add_argument("--survey", type=str, help="which survey to use")
    parser.add_argument("--savedir", type=str, help="where to save mcmcs")
    parser.add_argument("--nndir", type=str, help="which emulator runs to use")
    parser.add_argument("--version", type=str, help="which version of emulator")
    
    # Parse arguments
    args = parser.parse_args()
    
    print('reading in database...')

    validation_sims = np.load(f"{base_dir}/emulators/setrandomseed3/validation_sims.npy")
    df_validation = df.loc[validation_sims].copy()
 

    config = toml.load(f"{home_dir}/alfred/scripts/config_files/mcmc_config.toml")
    config = SimpleNamespace(**config)
    print(f"Now initialising mcmc run from config file {config.title}...")

    if args.survey:
        config.survey = args.survey
    config.savedir = f"biases_randomseed5/{args.savedir}"
    path = f"{base_dir}/inference/{config.savedir}"
    print(f'Saving to directory {path}...')
    os.makedirs(path)

    sampled_sims = df.sample(n=100).index.to_list()

    fn_sampled = f"{path}/sampled_pvals.npy"
    print(f'Saving sampled parameter values to {fn_sampled}')
    np.save(fn_sampled, sampled_sims)

     #=================================================================
    # RUNNING MCMC
    #=================================================================
    config.burnin = 1000
    config.nsteps = 5000
    config.nndir = args.nndir

    mob = []
    ratios_all = []

    for n in range(10):
        print(f"Now on run {n}...")
        # if n != 0:
        #     continue
        config.emu_version = f"emu{args.version}_run{n}"
        emu = summon_emu(f"{config.nndir}/{config.emu_version}", verbose=True)

        mob.append(emu)

        # path_ratios = f'{base_dir}/emulators/{config.nndir}/{config.emu_version}/ratios.npy'
        # ratios = np.load(path_ratios)
        # ratios_all.append(ratios)
    residuals = []
    ratios = []

    for sn in df_validation.index:
        tspec = utils.spectra(sn)[indices]
        espec = emulator.mechkemu(df.loc[sn].to_numpy(), mob)

        residuals.append(tspec - espec)
        ratios.append(tspec / espec)

    residuals = np.asarray(residuals)
    ratios = np.asarray(ratios)

    emuerr_file = f"{base_dir}/emulators/{config.nndir}/ensemble_error_{args.version}.npy"
    np.save(emuerr_files, np.std(residuals, axis=1))
    ratios_all = np.asarray(ratios_all)
    Amean = np.mean(ratios)
    Asigma = np.std(ratios)
    Ashape = np.mean(ratios, axis=1)

    for sn in sampled_sims:
        config.title = f"bias_simu{sn}"
        config.sn = sn
        mcmc_run = MCMC(config,
                        ells=ells[indices],
                        p0=draws(ndraws=config.nwalkers)[:,2:], 
                        emu=mob,
                        emuerr_file=emuerr_file,
                        # datapoints=datapoints,
                        A=np.random.uniform(.99,1.01, size=config.nwalkers),
                        Ashape=Ashape,
                        Astats=[Amean, Asigma],
                        # dryrun=True,
                        # showfigs=True,
                        verbose=True,
                        debug=False)

        mcmc_run.init_data()
        mcmc_run.init_run(savefig=True)
        sampler = mcmc_run.start_run(save=True)

        print()

if __name__ == "__main__":
    main()

