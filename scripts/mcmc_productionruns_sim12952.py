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

def get_spectra(emu, config, dir='setrandomseed3', base_dir=base_dir):
    truth = df.loc[config.sn].to_numpy()

    validation_sims = np.load(f"{base_dir}/emulators/{dir}/validation_sims.npy")
    true = np.zeros((len(validation_sims), len(ells[indices])))
    emulated = np.zeros((len(validation_sims), len(ells[indices])))

    for si, sn in enumerate(validation_sims):
        truspec = utils.spectra(sn)[indices]
        emuspec = emulator.kemu(df.loc[sn].to_numpy(), **emu)
        
        true[si] = truspec
        emulated[si] = emuspec

    return true, emulated


def extract_A(versions, config, axes=None, nruns=5, dir='setrandomseed3', base_dir=base_dir):
    truth = df.loc[config.sn].to_numpy()
    validation_sims = np.load(f"{base_dir}/emulators/{dir}/validation_sims.npy")
    shapes = np.zeros((len(versions), nruns, len(validation_sims), len(ells[indices])))

    for i, version in enumerate(versions):
        for n in range(nruns):
            path = f"{dir}/emu{version}_run{n}"
            emu = utils.summon_emu(path, verbose=True)

            true, emulated = get_spectra(emu, config, dir=dir)
            A = true / emulated

            shapes[i,n] = A

    return shapes

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="sets what experiment to use.")
    parser.add_argument("--survey", type=str, help="which survey to use")
    
    # Parse arguments
    args = parser.parse_args()
    
    print('reading in database...')
    sims = utils.get_sims('nells30_v5', base_dir=f"{base_dir}/spectra/kSZ/LoReLi")
    df = pd.read_pickle(f'{base_dir}/metadata/LoReLi_database_loggedparams.pkl')
    df = df.loc[df.index.intersection(sims)]

    # validation = df.sample(n=int(.2 * len(df)))
    # np.save(f"{dir}/validation_sims.npy", validation.index.to_list())
    config = toml.load(f"{home_dir}/alfred/scripts/config_files/mcmc_config.toml")
    print(f"Now initialising mcmc run {config['title']}...")
    print()
    dir = f"{base_dir}/inference/productionruns_sim12952"
    print(f"Saving analysis to {dir}...")
    print()
    config['survey'] = args.survey
    config['mcmcdir'] = dir
    config = SimpleNamespace(**config)

    noiseless = {'addnoise': False,
                'addPlanck': False}

    noiseless_Planck = {'addnoise': False,
                'addPlanck': True}

    addnoise = {'addnoise': True,
                'addPlanck': False}

    addnoise_Planck = {'addnoise': True,
                'addPlanck': True}

    setups = [noiseless, noiseless_Planck, addnoise, addnoise_Planck]

    #=================================================================
    # RUNNING MCMCs
    #=================================================================
    config.sn = '12952'
    testing = False

    if testing:
        config.burnin = 10
        config.nsteps = 100

    emu = summon_emu(config.emu)
    true, emulated = get_spectra(emu, config,
                         dir=config.nndir,
                         base_dir=base_dir)
    ratios = true / emulated
    np.save(f"{dir}/ratios.npy", ratios)

    datapoints = utils.spectra(config.sn)[indices]
    err_cov = surveys.error_cov(ells[indices],
                            datapoints,
                            surveys.telescopes[config.survey],
                            include_emulator=False)
    err =np.sqrt(np.diag(err_cov))
    noise = np.random.normal(scale=err)


    for i, setup in enumerate(setups):
        config.addnoise = False
        config.addPlanck = False

        title = f"{args.survey}"
        for key in setup.keys():
            print(f"{key} = {setup[key]}")
            setattr(config, key, setup[key])

        datapoints = utils.spectra(config.sn)[indices]
        if config.addnoise:
            datapoints += noise
            title += f"_noise"

        config.addnoise = False

        if config.addPlanck:
            title += f"_Planck"

        config.title = title
        print(f"Saving to {config.title}...")
        print()
        print(ratios.mean(axis=0))

        print(f"config settings are:")
        for name, value in vars(config).items():
            print(f"{name} = {value}")
        print()

        Amean = np.mean(ratios.mean(axis=0))
        Asigma = np.std(ratios, axis=0).mean()

        print(f"Running with Amean={Amean} and Asigma={Asigma}")
        mcmc_run = MCMC(config,
                        dir=f"{dir}",
                        ells=ells[indices],
                        p0=draws(ndraws=config.nwalkers)[:,2:], 
                        emu=emu,
                        datapoints=datapoints,
                        A=np.random.uniform(.99,1.01, size=config.nwalkers),
                        Ashape=ratios.mean(axis=0),
                        Astats=[Amean, Asigma],
                #     dryrun=True,
                    #    showfigs=True,
                        verbose=True)

        mcmc_run.init_data()
        mcmc_run.init_run(savefig=True)
        sampler = mcmc_run.start_run(save=True)

        print()

if __name__ == "__main__":
    main()

