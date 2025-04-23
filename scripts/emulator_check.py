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

import alfred.surveys as surveys
import alfred.emulator as emulator
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
    print("INVESTIGATION INTO EMULATOR EFFICACY")
    print('========================================================')

    config = toml.load(f"{home_dir}/alfred/scripts/{args.config}")
    print(f"Now initialising mcmc run {config['title']}...")
    print()

    dir = f"{base_dir}/inference/emulator_tests_newtrainingset"

    failedreion = np.load(f'{base_dir}/metadata/sims_failed.npy')
    sims = utils.get_sims('nells30_v3.1', base_dir=f"{base_dir}/spectra/kSZ/LoReLi")

    print('reading in database...')
    df = pd.read_pickle(f'{base_dir}/metadata/LoReLi_database_loggedparams.pkl')
    df = df.loc[df.index.intersection(sims)]
    #df = df.drop(sims_nan, errors='ignore')
    df = df.drop(failedreion, errors='ignore')


    indices = list(np.concatenate([np.arange(ells.size)[2:13], np.arange(ells.size)[13::2]]))
    if config['load_datasets'] == True:

        print('loading datasets...')
        features = df.to_numpy()
        dataset_v0 = np.load(f"{dir}/dataset_v0.npy")
        dataset_v1 = np.load(f"{dir}/dataset_v1.npy")
        dataset_v2 = np.load(f"{dir}/dataset_v2.npy")

        test_sims = np.load(f'{dir}/test_sims.npy', allow_pickle=True)
        train_sims = np.load(f'{dir}/train_sims.npy', allow_pickle=True)

        test_indices = df.index.get_indexer(test_sims)
        mask = np.zeros(len(df), dtype=bool)
        mask[test_indices] = True

        print('datasets loaded!')

    elif config['load_datasets'] == False:

        test_indices = np.random.randint(0, len(df)-1, int(.2 * len(df)))
        mask = np.zeros(len(df), dtype=bool)
        mask[test_indices] = True
        test_sims = df.index.to_numpy()[mask]
        train_sims = df.index.to_numpy()[~mask]

        np.save(f'{dir}/test_sims', test_sims)
        np.save(f'{dir}/train_sims', train_sims)

        print('forming datasets...')
        
        features = df.to_numpy()
        dataset_v0 = np.zeros((len(features), ells.size))
        dataset_v1 = np.zeros((len(features), ells.size))

        for i, sn in enumerate(df.index):
            dataset_v0[i] = utils.spectra(sn, dir='nells30_v2', key='kSZ')
            dataset_v1[i] = utils.spectra(sn, dir='nells30_v3.1', key='Dell')

        np.save(f'{dir}/dataset_v0', dataset_v0)
        np.save(f'{dir}/dataset_v1', dataset_v1)
        np.save(f'{dir}/dataset_v2', dataset_v1[:,indices])

        
    datasets = [dataset_v0, dataset_v1, dataset_v1[:,indices]]
    ellsets = [ells, ells, ells[indices]]


    sn = '11364'
    if config['use_data']:
        print('using calculated spectra as datapoints...')
        datapoints = utils.spectra(sn)

    else:
        datapoints = None

    fig, ax = plt.subplots(1,3, sharey=True, figsize=(12,5))
    fig.subplots_adjust(wspace=0.0)

    print('calculating uncertainities...')
    splits = []
    CVs = []
    noise = []
    for i, d in enumerate(datasets):
        CVs.append(surveys.sample_var(ellsets[i], d.mean(axis=0), surveys.telescopes['CMB-HD']))
        noise.append(surveys.noise(ellsets[i], surveys.telescopes['CMB-HD'])  / np.sqrt(np.diff(ellsets[i]).mean()))
        splits.append([features[~mask], features[mask], d[~mask], d[mask]])

        ax[i].errorbar(ellsets[i], d.mean(axis=0), yerr=noise[i], alpha=.5, marker='.', label='CMB-HD noise')
        ax[i].errorbar(ellsets[i], d.mean(axis=0), yerr=CVs[i], alpha=.75, marker='.', label='cosmic variance')

        ax[i].set_xlabel('ell')
        ax[i].set_title(f"dataset_v{i}")

    ax[0].set_ylabel('Dell')

    fig.savefig(f"{dir}/datasets.png")

    CV_n_noise = [CVs[i] + noise[i] for i in range(len(CVs))]

    print('initialising inputs...')

    uncertainties = [[None, None, None], [None, None, None], CVs, CV_n_noise]

    hyperparameters_v0 = {'neurons': 500,
                          'epochs': 100}
    
    hyperparameters_v1 = {'neurons': 1024,
                          'epochs': 500}
    
    hyperparameters_v2 = {'neurons': 1024,
                          'epochs': 500}
    
    hyperparameters_v3 = {'neurons': 1024,
                          'epochs': 500}

    
    hps = [hyperparameters_v0, hyperparameters_v1, hyperparameters_v2, hyperparameters_v3]
   
    for i, hp in enumerate(hps):
        label_emu = f"emu_v{i}"

        for j, data in enumerate(datasets):
            label_data = f"dataset_v{j}"

            if j==0:
                continue

            print('====================================================================')
            print(f"Running analysis with {label_emu} and {label_data}")
            print('====================================================================')

            run_dir = f"{dir}/{label_emu}_{label_data}"

            if config['load_emulator']:
                print(f'loading emulator from file in {run_dir}...')
                scalerX = joblib.load(f"{run_dir}/nn_emulator/scalerX.pkl")
                scalerY = joblib.load(f"{run_dir}/nn_emulator/scalerY.pkl")
                model = tf.keras.models.load_model(f"{run_dir}/nn_emulator/model.keras")

                emu = {'scalerX': scalerX,
                    'scalerY': scalerY,
                    'model': model}
   

            else:
                os.mkdir(run_dir)

                print(f"Print using inputs settings: \n\t{hp}")
                hp['uncertainties'] = uncertainties[i][j]

                nn = emulator.NeuralNetwork(hp, splits=splits[j],
                        scale_data=True, log_data=True, verbose=True)
                nn.prep_data()
                nn.regress()

                fig, ax = plt.subplots()

                ax.plot(nn.history.history['val_loss'], label='Validation Loss')
                ax.loglog(nn.history.history['loss'], label='Training Loss', alpha=0.5)

                ax.set_xlabel('Epoch')

                ax.set_title(f'emulator v{i}, dataset v{j}')
                ax.set_ylabel('Loss')

                fig.suptitle('Model Loss')
                fig.legend()
                fig.savefig(f"{run_dir}/modelloss_{label_emu}_{label_data}")

                nn.save(f"nn_emulator", path=run_dir)

                print(f"Finished constructing emulator for {label_emu} and {label_data}, now running MCMC")

                emu = {'scalerX': nn.scalerX,
                    'scalerY': nn.scalerY,
                    'model': nn.model}


            config['title'] = f"mcmc_run"
            config['emu'] = 'input_emu'

            if j==2:
                if datapoints is not None:
                    datapoints = datapoints[indices]

            # lmask = None
            # if j == 2:
            #     lmask = indices

            mcmc_run = MCMC(config, dir=run_dir, ells=ellsets[j], emu=emu, datapoints=datapoints, verbose=True)
            mcmc_run.init_data(savefig=True)
            mcmc_run.init_run()

            mcmc_run.start_run(save=True)


if __name__ == "__main__":
    main()
