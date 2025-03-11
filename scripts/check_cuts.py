import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import alfred.utils as utils
from alfred.parameters import *
import alfred.KSZ
import alfred.analyse as analyse

from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, CubicSpline
from catwoman.shelter import Cat

import argparse


def main():

    sn = '15060'
    
    print("Here we go!!!")

    path = '/Users/emcbride/sims'
    Pdd_fn = '/Users/emcbride/kSZ/data/Pdd.npy'
    errs_fn = '/Users/emcbride/kSZ/data/EMMA/EMMA_frac_errs.npz'

    #data_dir = '/data/cluster/emc-brid'
    #home_dir = '/home/emc-brid'
    home_dir = '/Users/emcbride'
    base_dir = f'{home_dir}/Datasets/LoReLi'
    # home_dir = '/jet/home/emcbride'

    spectra_path = 'ps_ee'
    ion_path = 'metadata/ion_histories_full.npz'
    Pee_path = 'spectra/Pee'
    kSZ_path = 'spectra/kSZ'
    fits_path = 'inference/lklhd/lklhd_files'
    params_path = f'metadata/param_files'
    redshift_file = f'metadata/redshift_list.dat'


    ells = np.arange(0, 15000, 500)
    ells[0] = 100  


    sim = Cat(sn, skip_early=True,
                            base_dir=base_dir,
                            path_spectra=spectra_path,
                            path_redshifts=redshift_file,
                            LoReLi_format=True,
                            verbose=True)
        


    print('smoothing Pee...')
    k, Pee = utils.smooth_Pee(sim)

    print('Pee smoothed...')
    print()

    Pdd = np.load(Pdd_fn)
    errs = np.load(errs_fn)
    EMMA_k = errs['k']
    frac_err_EMMA = errs['err']
    err_spline  = CubicSpline(EMMA_k, frac_err_EMMA)

    k0 = 3
    kf = 18
    krange = (k0, kf)

    z0 = np.where(sim.xe > .01)[0][0]
    zf = np.where(sim.xe > .9)[0][0] + 1
    zrange = (z0, zf)

    z_inter = np.linspace(5,25, 100)
    Pdd_spline = CubicSpline(z_inter, Pdd[:,k0:kf])
    Pdd_inter = Pdd_spline(sim.z[z0:zf])

    truths = [modelparams_Gorce2022['alpha0'], modelparams_Gorce2022['kappa']]
    priors =[(modelparams_Gorce2022['alpha0'] * .1, modelparams_Gorce2022['alpha0'] * 2.0),
                (0, modelparams_Gorce2022['kappa'] * 5.0),
                (modelparams_Gorce2022['k_f'] * .25, modelparams_Gorce2022['k_f'] * 5.0),
                (modelparams_Gorce2022['g'] * .25, modelparams_Gorce2022['g'] * 5.0)]

    fit = alfred.analyse.Fit(zrange, krange, modelparams_Gorce2022, sim, priors,
                                    frac_err=err_spline(sim.k[slice(*krange)]),
                                    Pdd=Pdd_inter, initialise=False)

    lkhd = fit.direct_lklhd_eval()

    kmin_cutoffs = np.concatenate([np.geomspace(1e-6,1e-3, 8), np.linspace(5e-3,1e-1, 12)])
    kmax_cutoffs = np.concatenate([np.linspace(1,3.5,20), np.geomspace(5,10000,10)])

    xemin_cutoffs = np.linspace(0.0, .1, 25)
    xemax_cutoffs = np.linspace(.95, 1.0, 25)

    deltas = [dz, dz / 2., dz / 5.]

    path = f"{base_dir}/cuts"

    Dells_kmin = []
    for kmin in kmin_cutoffs:
        print(f"Now on kmin={kmin}...")
    #for kmin in kmin_cutoffs:
        ksz = alfred.KSZ.KSZ_power(verbose=False, interpolate_xe=False, interpolate_Pee=False,
                helium=True, helium2=True, alpha0=fit.lklhd_params['alpha0'], kappa=fit.lklhd_params['kappa'],
                xemax=.97, kmin=kmin)
        
        ksz.run_camb(force=True)
        ksz.init_reionisation_history()

        Dell = ksz.run_ksz(ells=ells, patchy=True, Dells=True)[:,0]
        
        Dells_kmin.append(Dell)

    fn = f"{path}/kmin_cutoffs"
    np.savez(fn, cuts=kmin_cutoffs, Dells=Dells_kmin)
    print()
    print(f"Saving kmin cuts to {fn}")
    print()

    Dells_kmax = []
    for kmax in kmax_cutoffs:
        print(f"Now on kmax={kmax}...")
    #for kmin in kmin_cutoffs:
        ksz = alfred.KSZ.KSZ_power(verbose=False, interpolate_xe=False, interpolate_Pee=False,
                helium=True, helium2=True, alpha0=fit.lklhd_params['alpha0'], kappa=fit.lklhd_params['kappa'],
                xemax=.97, kmax=kmax)
        
        ksz.run_camb(force=True)
        ksz.init_reionisation_history()

        Dell = ksz.run_ksz(ells=ells, patchy=True, Dells=True)[:,0]
        
        Dells_kmax.append(Dell)

    fn = f"{path}/kmax_cutoffs"
    np.savez(fn, cuts=kmax_cutoffs, Dells=Dells_kmax)
    print()
    print(f"Saving kmax cuts to {fn}")
    print()

    Dells_xemin = []
    for xemin in xemin_cutoffs:
        print(f"Now on xemin={xemin}...")
    #for kmin in kmin_cutoffs:
        ksz = alfred.KSZ.KSZ_power(verbose=False, interpolate_xe=False, interpolate_Pee=False,
                helium=True, helium2=True, alpha0=fit.lklhd_params['alpha0'], kappa=fit.lklhd_params['kappa'],
                xemax=.97, xemin=xemin)
        
        ksz.run_camb(force=True)
        ksz.init_reionisation_history()

        Dell = ksz.run_ksz(ells=ells, patchy=True, Dells=True)[:,0]
        
        Dells_xemin.append(Dell)

    fn = f"{path}/xemin_cutoffs"
    np.savez(fn, cuts=xemin_cutoffs, Dells=Dells_xemin)
    print() 
    print(f"Saving xemin cuts to {fn}")
    print()

    Dells_xemax = []
    for xemax in xemax_cutoffs:
        print(f"Now on xemax={xemax}...")
    #for kmin in kmin_cutoffs:
        ksz = alfred.KSZ.KSZ_power(verbose=False, interpolate_xe=False, interpolate_Pee=False,
                helium=True, helium2=True, alpha0=fit.lklhd_params['alpha0'], kappa=fit.lklhd_params['kappa'],
                xemax=xemax)
        
        ksz.run_camb(force=True)
        ksz.init_reionisation_history()

        Dell = ksz.run_ksz(ells=ells, patchy=True, Dells=True)[:,0]
        
        Dells_xemax.append(Dell)

    fn = f"{path}/xemax_cutoffs"
    np.savez(fn, cuts=xemax_cutoffs, Dells=Dells_xemax)
    print()
    print(f"Saving xemax cuts to {fn}")
    print()

    Dells_dz= []
    zintegs = []
    for delta in deltas:
        print(f"Now on dz={delta}...")
    #for kmin in kmin_cutoffs:
        ksz = alfred.KSZ.KSZ_power(verbose=False, interpolate_xe=True, interpolate_Pee=True,
                helium=True, helium2=True, k_data=k, Pee_data=Pee, z_data=sim.z, xe_data=sim.xe,
                xemax=.97, dz=delta)
        
        print(f"dz={ksz.dz}")
        
        ksz.run_camb(force=True)
        ksz.init_reionisation_history()

        Dell = ksz.run_ksz(ells=ells, patchy=True, Dells=True)[:,0]
        
        Dells_dz.append(Dell)
        zintegs.append(ksz.z_integ)

    fn = f"{path}/dz_cutoffs"
    np.savez(fn, cuts=deltas, Dells=Dells_dz, zintegs=zintegs)
    print()
    print(f"Saving dz cuts to {fn}")
    print()

    print('Done, YAY!')

if __name__ == "__main__":
    main()
