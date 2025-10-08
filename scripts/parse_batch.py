import os
from random import sample
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

tf.config.set_visible_devices([], "GPU")

import warnings
from types import SimpleNamespace
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

import alfred.surveys as surveys
import alfred.emulator as emulator
import alfred.utils as utils
from alfred.parameters import *
from alfred.astrofit import *


def parse_batch(dir, ext):
    sampled_sims = []
    truths = []
    ravg = []
    ravg_random = []
    medians = []
    low68 = []
    high68 = []
    low95 = []
    high95 = []
    edges68 = []
    edges95 = []

    pattern = re.compile(r"(\d{5})")
    for i in range(10):
        print(f"On {i}")

        directory = f"{dir}/samples_{i}"
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            # print(filepath)

            match = pattern.search(filename)
            if match:
                sn = match[0]
                # print(f"On sim {sn}")
                if sn in sampled_sims:
                    print(f"resampled")
                    continue
                if sn in ["18861", "19048", "17587", "17734"]:
                    continue
                if not os.path.exists(f"{filepath}/saved_chains.h5"):
                    continue

                s, l = utils.load_samples(filepath, verbose=False)
                samples = make_tauchains(
                    s[:, :3], truths=df.loc[sn].to_dict(), dropA=False
                )
                truth = np.asarray(
                    [*df.loc[sn].to_numpy(), *whatthetau(df.loc[sn].to_numpy())]
                )
                ravg.append(np.mean(np.where(samples < truth[None, 2:], 1, 0), axis=0))

                samples_random = draws(ndraws=samples.shape[0])[2:]
                ravg_random.append(
                    np.mean(np.where(samples_random < truth[None, 2:], 1, 0), axis=0)
                )

                ci = 68
                lower_q = (100 - ci) / 2
                upper_q = 100 - lower_q
                low, high = np.percentile(samples, [lower_q, upper_q], axis=0)
                low68.append(low)
                high68.append(high)
                edges68.append((low, high))

                ci = 95
                lower_q = (100 - ci) / 2
                upper_q = 100 - lower_q
                low, high = np.percentile(samples, [lower_q, upper_q], axis=0)
                low95.append(low)
                high95.append(high)
                edges95.append((low, high))

                sampled_sims.append(sn)
                truths.append(truth)
                medians.append(np.median(samples, axis=0))

    sampled_sims = np.asarray(sampled_sims)
    truths = np.asarray(truths)
    ravg = np.asarray(ravg)
    medians = np.asarray(medians)
    low68 = np.asarray(low68)
    high68 = np.asarray(high68)
    low95 = np.asarray(low95)
    high95 = np.asarray(high95)
    edges68 = np.asarray(edges68)
    edges95 = np.asarray(edges95)

    intervals68 = edges68 - np.stack([medians, medians], axis=1)
    intervals95 = edges95 - np.stack([medians, medians], axis=1)

    intervals68 = np.swapaxes(intervals68, 1, 2)
    intervals95 = np.swapaxes(intervals95, 1, 2)

    savedir = f"{home_dir}/batchstats_{ext}"
    np.save(f"{savedir}/sims_{ext}", sampled_sims)
    np.save(f"{savedir}/truths_{ext}", truths)
    np.save(f"{savedir}/ravg_{ext}", ravg)
    np.save(f"{savedir}/ravg_random_{ext}", ravg_random)
    np.save(f"{savedir}/medians_{ext}", medians)
    np.save(f"{savedir}/low68_{ext}", low68)
    np.save(f"{savedir}/high68_{ext}", high68)
    np.save(f"{savedir}/low95_{ext}", low95)
    np.save(f"{savedir}/high95_{ext}", high95)
    np.save(f"{savedir}/CI68_{ext}", high68 - low68)
    np.save(f"{savedir}/CI95_{ext}", high95 - low95)
    np.save(f"{savedir}/intervals68_{ext}", np.abs(intervals68))
    np.save(f"{savedir}/intervals95_{ext}", np.abs(intervals95))

    return


def main():
    dir = f"{base_dir}/inference/biases"
    print(f"Now running through {dir}...")
    parse_batch(dir, "v1")

    dir = f"{base_dir}/inference/biases_rerun_lowernoise"
    print(f"Now running through {dir}...")
    parse_batch(dir, "v2")


if __name__ == "__main__":
    main()
