import os
import re
import time
import copy as cp
import argparse
import toml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
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

    config = toml.load(f"{home_dir}/alfred/scripts/config_files/mcmc_config.toml")
    config = SimpleNamespace(**config)

    validation_sims = np.load(f"{base_dir}/emulators/setrandomseed3/validation_sims.npy")
    df_validation = df.loc[validation_sims].copy()


    config = toml.load(f"{home_dir}/alfred/scripts/config_files/mcmc_config.toml")
    config = SimpleNamespace(**config)
    print(f"Now initialising mcmc run from config file {config.title}...")

    config.savedir = f"test_hksz_withhksz/{config.survey}"


    datapoints = utils.spectra(config.sn)[indices]
    err_cov = surveys.error_cov(ells[indices],
                            datapoints,
                            surveys.telescopes[config.survey],
                            include_emulator=False)
    err =np.sqrt(np.diag(err_cov))
    noise = np.random.normal(scale=err)


    #=================================================================
    # RUNNING MCMC
    #=================================================================
    testing = False

    if testing:
        config.burnin = 2
        config.nsteps = 100

    mob = []
    residuals_all = []
    ratios_all = []

    for n in range(5):
        print(f"Now on run {n}...")
        # if n != 0:
        #     continue
        config.emu_version = f"emuv5.1_run{n}"
        emu = summon_emu(f"{config.nndir}/{config.emu_version}", verbose=True)

        mob.append(emu)

        path_residuals = f'{base_dir}/emulators/{config.nndir}/{config.emu_version}/residuals.npy'
        path_ratios = f'{base_dir}/emulators/{config.nndir}/{config.emu_version}/ratios.npy'

        if os.path.exists(path_residuals):
            print(f'{path_residuals} already exists')
            residuals = np.load(path_residuals)
        else:
            print('creating residuals files...')
            emu = summon_emu(f"{config.nndir}/{config.emu_version}")
            tspec = np.zeros((len(validation_sims), len(ells[indices])))
            for i, sn in enumerate(df_validation.index):
                tspec[i] = utils.spectra(sn)[indices]

            espec = emulator.kemu(df_validation.to_numpy(), **emu)

            residuals = tspec - espec
            np.save(path_residuals, residuals)

            print(f"residuals file saved to {path_residuals}")
            print()
        
        residuals_all.append(residuals)
        
        if os.path.exists(path_ratios):
            print(f'{path_ratios} already exists')
            ratios = np.load(path_ratios)
        else:
            print('creating ratios file...')
            emu = summon_emu(f"{config.nndir}/{config.emu_version}")
            tspec = np.zeros((len(validation_sims), len(ells[indices])))
            for i, sn in enumerate(df_validation.index):
                tspec[i] = utils.spectra(sn)[indices]

            espec = emulator.kemu(df_validation.to_numpy(), **emu)

            ratios = tspec / espec
            np.save(path_ratios, ratios)

            print(f"ratios file saved to {path_ratios}")

            print()

        ratios_all.append(ratios)

    residuals_all = np.asarray(residuals_all)
    print(f"residuals shape is {residuals_all.shape}")
    emu_error = np.std(residuals_all, axis=1)
    print(f"emu error : {emu_error.shape}")
    emu_error = np.mean(emu_error, axis=0)

    print(f"emu error : {emu_error.shape}")


    emuerr_file = f"{base_dir}/emulators/{config.nndir}/ensemble_error_v5.1.npy"
    print(f"emu err: {emu_error.shape}")
    print(f'saving average error file to {emuerr_file}...')
    np.save(emuerr_file, emu_error)

    ratios_all = np.asarray(ratios_all)
    Amean = np.mean(ratios_all)
    Asigma = np.std(ratios_all)
    Ashape = np.mean(np.mean(ratios_all,axis=0), axis=0)

    noiseless = {'title': 'noiseless',
                    'addnoise': False,
                    'addPlanck': False}

    for key in noiseless.keys():
        print(f"{key} = {noiseless[key]}")
        setattr(config, key, noiseless[key])

    config.use_data = True
    config.burnin = 1000
    config.nsteps = 2000

    mcmc_run = MCMC(config,
                    ells=ells[indices],
                    p0=draws(ndraws=config.nwalkers)[:,2:], 
                    emu=mob,
                    emuerr_file=emuerr_file,
            #       datapoints=datapoints,
                    fit_hksz=True,
                    add_fg_residuals=False,
                    A=np.random.uniform(.99,1.01, size=config.nwalkers),
                    Ashape=Ashape,
                    Astats=[Amean, Asigma],
                    dryrun=False,
                    showfigs=True,
                    Planck=whatthetau(df.loc[config.sn].to_numpy())[0],
                    verbose=True,
                    debug=False)

    mcmc_run.init_data()
    mcmc_run.init_run(savefig=True)
    sampler = mcmc_run.start_run(save=True)

    print()

if __name__ == "__main__":
    main()