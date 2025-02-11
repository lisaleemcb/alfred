import os
import re
import time
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import zeus

from scipy.interpolate import PchipInterpolator, CubicSpline

import alfred.utils as utils
from alfred.parameters import *
from alfred.emulator import kemu
import alfred.KSZ
import alfred.analyse as analyse

import joblib
import tensorflow as tf


from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

import argparse


def main():
    home_dir = '/home/emc-brid'  # glx
    #home_dir = '/Users/emcbride/alfred' # personal ordi
    # home_dir = '/jet/home' # bridges2


    #baddies = ['15593', '13492', '13493'] # these don't have redshift files

    print("Here we go!!!")

    sim_path = '{home_dir}/ps_ee'
    ion_path = '{home_dir}/ion_histories_full.npz'
    Pee_path = '{home_dir}/spectra/Pee'
    kSZ_path = '{home_dir}/spectra/kSZ'
    fits_path = '{home_dir}/lklhd_files'
    params_path = '{home_dir}/param_files'
    redshift_file = '{home_dir}/redshift_list.dat'

 
    scalerX = joblib.load("scalerX_v3.pkl")
    scalerY = joblib.load("scalerY_v3.pkl")
    model = tf.keras.models.load_model('NNv3_model.keras')

    ells = np.linspace(1,15000, 30)
    delta_ell = np.diff(ells).mean()
    ells_error = np.load(f'{home_dir}/ells_for_regressor.npy')

    df = pd.read_pickle(f'{home_dir}/LoReLi_database_loggedparams.pkl')
    theta_true =  df[df.columns].mean().to_numpy()
    datapoints = kemu(theta_true, scalerX=scalerX, scalerY=scalerY, model=model, log_data=True)


    a682 = np.load(f'{home_dir}/a682_MLerror.npy')
    b682 = np.load(f'{home_dir}/b682_MLerror.npy')


    emu_error_spline = CubicSpline(ells_error, np.maximum(np.abs(a682), b682), bc_type='natural')
    emu_error = emu_error_spline(ells)

    labels = df.columns
    priors = [(df[p].to_numpy().min(), df[p].to_numpy().max()) for p in labels]

    # err_cov = np.diag(sample_var(ells, datapoints, telescope_specs)**2
    #                     + (noise(ells, telescope_specs, pol=False)/np.sqrt(delta_ell))**2)

    # err_cov_emu = np.diag(sample_var(ells, datapoints, telescope_specs)**2
    #                     + (noise(ells, telescope_specs, pol=False)/np.sqrt(delta_ell))**2 
    #                     + (emu_error/np.sqrt(delta_ell))**2)

    err_cov_justemu = np.diag((emu_error/np.sqrt(delta_ell))**2)

    sims = cp.deepcopy(df.index.to_numpy())
    features = cp.deepcopy(df.to_numpy())

    priors2d = np.load(f'{home_dir}/priors2D_coeffs_logMminvslogtau_v2.npz')
    pass_prior = np.load(f'{home_dir}/pass_prior.npy')

    def lnprior(theta, priors=priors, priors2d=priors2d, add_2d=True):
        for i, p in enumerate(priors):
            low, high = p
            if not (low <= theta[i] <= high):
                return -np.inf
            
            if add_2d:
                Mmin = theta[3] # CAREFUL! currently hardcoded (also log10)
                tau = theta[2]  # CAREFUL! currently hardcoded (also log10)

                below = priors2d['m'] * Mmin + priors2d['b_below']
                above = priors2d['m'] * Mmin + priors2d['b_above']

                if not(below <= tau <= above):
                    return -np.inf

        return 0.
    
    def lnprob(theta,  data, err, add_2d=True, sn=None):
        lp = lnprior(theta, add_2d=add_2d)
        if not np.isfinite(lp):
            return -np.inf#, 0.
        ln = lnlike(theta, data, err, sn)

        return lp + ln #, model

    def lnlike(theta, data, err, sn):
        if not sn:
            guess_model = kemu(theta, scalerX=scalerX, scalerY=scalerY, model=model, log_data=True)
            
            return -0.5 * np.sum((data - guess_model) ** 2.0 / err**2.0)

        elif sn:
            fn_L = f'{data_path}/kSZ_LoReLi_simu{sn}.npz'
            ksz = np.load(fn_L, allow_pickle=True)
            signal = ksz['kSZ']

        return -0.5 * np.sum((data - signal) ** 2.0 / err**2.0)


    chains_fn = "saved_chains.h5"
    save_progress = zeus.callbacks.SaveProgressCallback(chains_fn, ncheck=100)

    autocorr_check = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=50, discard=0.5)
    R_check = zeus.callbacks.SplitRCallback(ncheck=100, epsilon=0.01, nsplits=2, discard=0.5)
    miniter_check = zeus.callbacks.MinIterCallback(nmin=500)

    nwalkers = 12
    burnin = 500
    nsteps = 10
    ndim = len(theta_true)

    #p0 = pass_prior[:12]
    p0 = pass_prior[50:62]
    #p0 = pass_prior[100:112]
            
    print('Okay, here we go!')
    start_time = time.time()

    sampler = zeus.EnsembleSampler(nwalkers, ndim, lnprob,
                    args=[datapoints, np.sqrt(np.diag(err_cov_justemu))])

    sampler.run_mcmc(p0, burnin)

    burnin_samples = sampler.get_chain()
    start = burnin_samples[-1] # Get the burnin samples

    sampler = zeus.EnsembleSampler(nwalkers, ndim, lnprob,
                    args=[datapoints, np.sqrt(np.diag(err_cov_justemu))], moves=zeus.moves.GlobalMove())
    sampler.run_mcmc(start, nsteps, callbacks=[save_progress, autocorr_check, R_check, miniter_check])


    end_time = time.time()

    print('finished MCMC, saving files...')

    np.save('burnin', burnin_samples)
    np.save('samples', sampler.get_chain())
    np.save('logps', sampler.get_log_prob())
    np.save('tau', autocorr_check.estimates)
    np.save('R', R_check.estimates)

    print('Done, YAY!')

if __name__ == "__main__":
    main()
