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

from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

import alfred.surveys as surveys
import alfred.emulator as emulator
from alfred.parameters import *
from alfred.astrofit import *


def main():

    print('========================================================')
    print("MCMC RUNS")
    print('========================================================')

    config_files = [f"{home_dir}/alfred/scripts/config_files/mcmc_CMB-HD.toml",
                   f"{home_dir}/alfred/scripts/config_files/mcmc_CMB-HD_Planck.toml",
                   f"{home_dir}/alfred/scripts/config_files/mcmc_CMB-HD_noise.toml", 
                    f"{home_dir}/alfred/scripts/config_files/mcmc_CMB-HD_noise_Planck.toml",
                    f"{home_dir}/alfred/scripts/config_files/mcmc_CMB-S4.toml",
                    f"{home_dir}/alfred/scripts/config_files/mcmc_CMB-S4_Planck.toml",
                    f"{home_dir}/alfred/scripts/config_files/mcmc_CMB-S4_noise.toml",
                    f"{home_dir}/alfred/scripts/config_files/mcmc_CMB-S4_noise_Planck.toml",
                    f"{home_dir}/alfred/scripts/config_files/mcmc_SO-LAT.toml",
                    f"{home_dir}/alfred/scripts/config_files/mcmc_SO-LAT_Planck.toml",
                    f"{home_dir}/alfred/scripts/config_files/mcmc_SO-LAT_noise.toml",
                    f"{home_dir}/alfred/scripts/config_files/mcmc_SO-LAT_noise_Planck.toml"
]
    
    for config_i, c in enumerate(config_files):
        config = toml.load(c)
        print(f"Now initialising mcmc run {config['title']}...")
        print()

        dir = f"{base_dir}/inference/productionruns1"
        
        print(f"Saving analysis to {dir}...")

        print(f"config settings are:")
        print(f"\t{config}")
        print()

        config['use_data'] = True

        if config['use_data']:
            print(f"using calculated spectra from simu{config['sn']} as datapoints...")
            datapoints = utils.spectra(config['sn'])[indices]

        else:
            datapoints = None

        print('====================================================================')
        print(f"Running analysis with {emu_dir} and {config['survey']}")
        print('====================================================================')


        if config['load_emulator']:
            print(f'loading emulator from file in {emu_dir}...')

            emu = utils.summon_emu('')
            
            testing = False
            if testing:
                config['burnin'] = 10
                config['nsteps'] = 10

            config['emu'] = 'input_emu'

            print(f"Finished loading emulator, now running MCMC...")
        mcmc_run = MCMC(config,
                        dir=f"{dir}",
                        ells=ells[indices],
                        p0=draws(ndraws=config.nwalkers)[:,2:], 
                        emu=summon_emu(f"{nndir}/emu{version}_run{n}"),
                        datapoints=datapoints,
                        A=np.random.uniform(.99,1.01, size=config.nwalkers),
                        Ashape=shapes[0,n].mean(axis=0),
                        Astats=[Amean, Asigma],
                #     dryrun=True,
                    #    showfigs=True,
                        verbose=True)

        mcmc_run.init_data()
        mcmc_run.init_run(savefig=True)
        sampler = mcmc_run.start_run(save=True)


if __name__ == "__main__":
    main()
