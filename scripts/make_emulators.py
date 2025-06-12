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


config = toml.load(f"{home_dir}/alfred/scripts/config_files/mcmc_CMB-HD.toml")
config = SimpleNamespace(**config)

batchdir = 'batch_setrandomseed'
dir = f"{base_dir}/emulators/{batchdir}"
print(f"Saving emulators to {dir}...")

# print(f"config settings are:")
# print(f"\t{config}")
# print()

print('reading in database...')
sims = utils.get_sims('nells30_v5', base_dir=f"{base_dir}/spectra/kSZ/LoReLi")
df = pd.read_pickle(f'{base_dir}/metadata/LoReLi_database_loggedparams.pkl')
df = df.loc[df.index.intersection(sims)]

validation = df.sample(n=int(.2 * len(df)))
np.save(f"{dir}/validation_sims.npy", validation.index.to_list())

df = df.drop(validation.index, errors='ignore')

dataset = np.zeros((len(df), 30))
for i, sn in enumerate(df.index):
    dataset[i] = utils.spectra(sn)

indices = list(np.concatenate([np.arange(ells.size)[3:13], np.arange(ells.size)[13::2]]))
# cv = surveys.sample_var(ells[indices], utils.spectra(config.sn)[indices], surveys.telescopes['CMB-HD'])
# noise = surveys.noise(ells[indices], surveys.telescopes['CMB-HD'])  / np.sqrt(np.diff(ells[indices]).mean())

emu_labels = ['v5.0', 'v5.1', 'v5.2', 'v5.3']
seeds = [42, 123, 587, 1122, 4680]
setups = []

for label in emu_labels:
    hp = np.load(f"{base_dir}/emulators/{label}/training_files.npz", allow_pickle=True)
    setups.append(hp)

nruns = 5
for index, label in enumerate(emu_labels):
    print(f"Running for setup {emu_labels[index]}...")

   # splits = [setup['X_train'], setup['X_test'], setup['y_train'], setup['y_test']]
    for n in range(nruns):

        nn = emulator.NeuralNetwork(setups[index]['hyperparameters'].item(), seed=seeds[n],
                                features=df.to_numpy(), dataset=dataset[:,indices],
                                verbose=True)
        nn.prep_data()
        nn.regress()

        nn.save(f"{batchdir}/emu{label}_run{n}")