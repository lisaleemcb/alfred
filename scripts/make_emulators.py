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


validation_sims = np.load(f"{dir}/validation_sims.npy")
# validation = df.sample(n=int(.2 * len(df)))
# np.save(f"{dir}/validation_sims.npy", validation.index.to_list())

df = df.drop(validation_sims, errors='ignore')

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


config = toml.load(f"{home_dir}/alfred/scripts/config_files/mcmc_CMB-HD.toml")
print(f"Now initialising mcmc run {config['title']}...")
print()
config = SimpleNamespace(**config)

dir = f"{base_dir}/inference/emulator_tests"
print(f"Saving analysis to {dir}...")

print(f"config settings are:")
print(f"\t{config}")
print()

dataset = np.zeros((len(df), 30))
for i, sn in enumerate(df.index):
    dataset[i] = utils.spectra(sn)

indices = list(np.concatenate([np.arange(ells.size)[3:13], np.arange(ells.size)[13::2]]))
cv = surveys.sample_var(ells[indices], utils.spectra(config.sn)[indices], surveys.telescopes['CMB-HD'])
noise = surveys.noise(ells[indices], surveys.telescopes['CMB-HD'])  / np.sqrt(np.diff(ells[indices]).mean())

emu_labels = ['v5.0', 'v5.1', 'v5.2', 'v5.3']

nruns = 5
emus = []
for index, version in enumerate(emu_labels):
    print(f"Running for setup {emu_labels[index]}...")
    runs = []

   # splits = [setup['X_train'], setup['X_test'], setup['y_train'], setup['y_test']]
    for i in range(nruns):
    #    print(f"On run {i} of {emu_labels[index]} setup")
        emu = utils.summon_emu(f"{dir}/emu{version}_run{i}", verbose=True)
        emus.append(emu)    

fig, ax = plt.subplots(4,len(emu_labels), sharex=True, sharey='row', figsize=(16,8))
fig.subplots_adjust(wspace=0.0, hspace=0.0)

all_L1s = []
all_ratios = []
all_spectra = np.zeros((nruns, len(ells[indices])))

sn = '12958'
sampled_sims = df.sample(n=10).index
#sn = '11364'
true = utils.spectra(sn)
err_cov = surveys.error_cov(ells, true, surveys.telescopes['CMB-HD'])
err = np.sqrt(np.diag(err_cov))

for index, version in enumerate(emu_labels):
    print(f"Running for setup {emu_labels[index]}...")

    L1s = []
    ratios = []
   # splits = [setup['X_train'], setup['X_test'], setup['y_train'], setup['y_test']]
    for i in range(nruns):
      #  nnfiles = np.load(f"{base_dir}/emulators/{dir}/emu{version}_run{i}")

        emu = emus[5*index+i]
        print(5*index+i)

        ax[0,index].errorbar(ells[indices], np.zeros_like(ells[indices]),
                              yerr=err[indices], color='black', alpha=.25)
        ax[1,index].errorbar(ells[indices], np.zeros_like(ells[indices]),
                              yerr=err[indices], color='black', alpha=.25)
        ax[2,index].axhline(1.0, color='black', alpha=.25)
        ax[3,index].axhline(1.0, color='black', alpha=.25)

        emu_spectra = emulator.kemu(df.loc[sn].to_numpy(), **emu)
     #   all_spectra[i] = true_spectra[indices] - emu_spectra
        ax[0,index].plot(ells[indices], true[indices] - emu_spectra,
                          color=colors[i], alpha=.75)
        ax[2,index].plot(ells[indices], true[indices] / emu_spectra,
                          color=colors[i], alpha=.75)


        true_spectra= []
        emulated_spectra = []
        for sim in validation_sims:
            tspec = utils.spectra(sim)[indices]
            espec = emulator.kemu(df.loc[sim].to_numpy(), **emu)

            true_spectra.append(tspec)
            emulated_spectra.append(espec)
      #      all_L1s.append(np.asarray(L1s).mean(axis=0))

        true_spectra = np.asarray(true_spectra)
        emulated_spectra = np.asarray(emulated_spectra)

        L1 = (true_spectra - emulated_spectra).mean(axis=0)
        ratio = (true_spectra / emulated_spectra).mean(axis=0)
        
        all_L1s.append(L1)
        all_ratios.append(ratio)
        
        ax[1,index].plot(ells[indices],L1, color=colors[i], alpha=.5)
        ax[3,index].plot(ells[indices], ratio, color=colors[i], alpha=.5)

        L1s.append(L1)
        ratios.append(ratio)

    #ax[0,index].plot(ells[indices], all_spectra.mean(axis=0), color='red', alpha=1.0) 
    ax[1,index].plot(ells[indices], np.asarray(all_L1s).mean(axis=0), color='red', alpha=1.0)
    ax[3,index].plot(ells[indices], np.asarray(all_ratios).mean(axis=0), color='red', alpha=1.0)  

    ax[0,index].set_title(f"Emulator {emu_labels[index]}")
# ax[1].errorbar(ells[indices], np.zeros_like(ells[indices]), yerr=, color='black', alpha=.25)
ax[0,0].set_ylabel(f"(data - model) \n fiducial")
ax[1,0].set_ylabel(f"(data - model) \n validation sims")
ax[2,0].set_ylabel(f"(data / model) \n fiducial")
ax[3,0].set_ylabel(f"(data / model) \n validation sims")

ax[0,0].legend()
ax[1,0].set_ylim(-0.02, 0.02)

fig.savefig(f"{batchdir}/residuals.png")