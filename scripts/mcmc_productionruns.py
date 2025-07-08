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
import keras
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import warnings
from types import SimpleNamespace
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

import alfred.surveys as surveys
import alfred.emulator as emulator
import alfred.utils as utils
from alfred.parameters import *
from alfred.astrofit import *


def main():
    # Set up argument parser
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
    config.savedir = f"{args.savedir}/{config.survey}"

    noiseless = {'title': 'noiseless',
                    'addnoise': False,
                    'addPlanck': False}

    noiseless_Planck = {'title': 'noiseless_Planck',
                    'addnoise': False,
                    'addPlanck': True}

    addnoise = {'title': 'addnoise',
                    'addnoise': True,
                    'addPlanck': False}

    addnoise_Planck = {'title': 'addnoise_Planck',
                    'addnoise': True,
                    'addPlanck': True}

    setups = [noiseless, noiseless_Planck, addnoise, addnoise_Planck]


    #=================================================================
    # RUNNING MCMC
    #=================================================================
    testing = False

    if testing:
        config.burnin = 2
        config.nsteps = 10

    config.nndir = args.nndir

    for i, setup in enumerate(setups):
        for n in range(5):
            print(f"Now on run {n}...")
            # if n != 0:
            #     continue
            config.emu_version = f'emu{args.version}_run{n}'

            for key in setup.keys():
                        print(f"{key} = {setup[key]}")
                        setattr(config, key, setup[key])

            config.title += f"/run{n}"

            if os.path.exists(f'{base_dir}/emulators/{config.nndir}/{config.emu_version}/residuals.npy'):
                print(f'residuals file at {base_dir}/emulators/{config.nndir}/{config.emu_version}/residuals.npy already exists')
            else:
                print('creating residuals files...')
                emu = summon_emu(f"{config.nndir}/{config.emu_version}")
                tspec = np.zeros((len(validation_sims), len(ells[indices])))
                for i, sn in enumerate(df_validation.index):
                    tspec[i] = utils.spectra(sn)[indices]

                espec = emulator.kemu(df_validation.to_numpy(), **emu)

                residuals = tspec - espec
                np.save(f"{base_dir}/emulators/{config.nndir}/{config.emu_version}/residuals.npy", residuals)

                print(f"residuals file saved to {base_dir}/emulators/{config.emu_version}/residuals.npy")
                print()
            
            if os.path.exists(f'{base_dir}/emulators/{config.nndir}/{config.emu_version}/ratios.npy'):
                print(f'residuals file at {base_dir}/emulators/{config.nndir}/{config.emu_version}/ratios.npy already exists')
                ratios = np.load(f"{base_dir}/emulators/{config.nndir}/{config.emu_version}/ratios.npy")
            else:
                print('creating ratios file...')
                emu = summon_emu(f"{config.nndir}/{config.emu_version}")
                tspec = np.zeros((len(validation_sims), len(ells[indices])))
                for i, sn in enumerate(df_validation.index):
                    tspec[i] = utils.spectra(sn)[indices]

                espec = emulator.kemu(df_validation.to_numpy(), **emu)

                ratios = tspec / espec
                np.save(f"{base_dir}/emulators/{config.nndir}/{config.emu_version}/ratios.npy", ratios)

                print(f"ratios file saved to {base_dir}/emulators/{config.nndir}/{config.emu_version}/ratios.npy")

                print()

            mcmc_run = MCMC(config,
                            ells=ells[indices],
                            p0=draws(ndraws=config.nwalkers)[:,2:], 
                        # emu=emu,
                        # datapoints=datapoints,
                            A=np.random.uniform(.99,1.01, size=config.nwalkers),
                            Ashape=ratios.mean(axis=0),
                            Astats=[np.mean(ratios), np.std(ratios)],
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

