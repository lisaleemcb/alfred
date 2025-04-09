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
    print("PREPARING BATCH MCMC RUN")
    print('========================================================')

    config = toml.load(f"{home_dir}/alfred/scripts/{args.config}")
    print(f"Now initialising mcmc run {config['title']}...")
    print()

    dir = f"{base_dir}/inference/emulator_tests"

    indices = list(np.concatenate([np.arange(ells.size)[2:13], np.arange(ells.size)[13::2]]))
    test_indices = np.random.randint(0, len(df)-1, int(.2 * len(df)))
    mask = np.zeros(len(df), dtype=bool)
    mask[test_indices] = True
    test_sims = df.index.to_numpy()[mask]
    train_sims = df.index.to_numpy()[~mask]

    np.save(f'{dir}/test_sims', test_sims)
    np.save(f'{dir}/train_sims', train_sims)

    features = df.to_numpy()
    dataset_v2 = np.zeros((len(features), ells.size))
    dataset_v3 = np.zeros((len(features), ells.size))

    for i, sn in enumerate(df.index):
        dataset_v2[i] = utils.spectra(sn, dir='nells30_v2', key='kSZ')
        dataset_v3[i] = utils.spectra(sn, dir='nells30_v3.1', key='Dell')

    datasets = [dataset_v2, dataset_v3, dataset_v3[:,indices]]
    ellsets = [ells, ells, ells[indices]]

    fig, ax = plt.subplots(1,3, sharey=True, figsize=(12,5))
    fig.subplots_adjust(wspace=0.0)

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

    CV_n_noise = [cv + noise for cv in CVs]

    uncertainties = [[None, None, None], [None, None, None], CVs, CV_n_noise]

    hyperparameters_v1 = {'neurons': 500,
                          'epochs': 100}
    
    hyperparameters_v2 = {'neurons': 1024,
                          'epochs': 500}
    
    hyperparameters_v3 = {'neurons': 1024,
                          'epochs': 500}
    
    hyperparameters_v4 = {'neurons': 1024,
                          'epochs': 500}

    
    hps = [hyperparameters_v1, hyperparameters_v2, hyperparameters_v3, hyperparameters_v4]
   
    for i, hp in enumerate(hps):
        label_emu = f"emu_v{i}"

        for j, data in enumerate(datasets):
            label_data = f"dataset_v{j}"

            print('====================================================================')
            print(f"Running analysis for emulator {label_emu} and dataset {label_data}")
            print('====================================================================')

            hp['uncertainities'] = uncertainties[i][j]
             

            nn = emulator.NeuralNetwork(hp, splits=splits[j],
                    scale_data=True, log_data=True, verbose=True)
            nn.prep_data()
            nn.regress()

            fig, ax = plt.subplots()

            ax.plot(nn.history.history['val_loss'], label='Validation Loss')
            ax.loglog(nn.history.history['loss'], label='Training Loss', alpha=0.5)

            ax.set_xlabel('Epoch')

            ax.set_title(f'Model Loss')
            ax[0].set_ylabel('Loss')

            fig.suptitle('Model Loss')
            fig.savefig(f"{dir}/modelloss_{label_emu}_{label_data}")

            nn.save(f"{label_emu}_{label_data}", path=dir)

            emu = {'scalerX': nn.scalerX,
                   'scalerY': nn.scalerY,
                   'model': nn.model}

            print(f"Finished constructing emulator for {label_emu} and dataset {label_data}, now runnng MCMC")

            config['title'] = f"{config['title']}_{label_emu}_{label_data}"
            config['emulator'] = emu

            mcmc_run = MCMC(config, dir=dir, verbose=True)
            mcmc_run.init_data()
            mcmc_run.init_run()

            mcmc_run.start_run(save=True)


if __name__ == "__main__":
    main()
