import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import alfred.utils as utils
from alfred.parameters import *
import alfred.KSZ
import alfred.peefit as peefit

from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load a numpy file which is a list of sims and save to a directory.")
    parser.add_argument("--sims", type=str, help="Path to the numpy file (.npy or .npz) with the sims to parse")
    parser.add_argument("--save_dir", type=str, help="Directory name for where to save the kSZ spectra")
    parser.add_argument("--n", type=str, help="Integer which just keeps track of which slurm index this is running")
    
    # Parse arguments
    args = parser.parse_args()

    # Pee_path = '/Users/emcbride/spectra/Pee'
    # kSZ_path = '/Users/emcbride/spectra/kSZ'
    # fits_path = '/Users/emcbride/lklhd_files'
    # params_path = '/Users/emcbride/kSZ/data/LoreLi_summaries/param_files'

    #baddies = ['15593', '13492', '13493'] # these don't have redshift files

    print("Here we go!!!")

    data_dir = '/data/cluster/emc-brid'
    home_dir = '/home/emc-brid'
    # home_dir = '/Users/emcbride'
    base_dir = f'{data_dir}/Datasets/LoReLi'
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

    if args.sims:
        if os.path.exists(args.sims):
            #:process_file(args.file)
                # Load the numpy file
            file_path = args.sims
            sims = np.load(file_path)
            print('Sims list loaded for this run:')
            print(f'\t {sims}')

    else:
        pattern = re.compile(r".*simu\d+.*")

        # List to store the extracted numbers
        sims = []
        # Loop through files in the directory
        for item in os.listdir(f'{base_dir}/{Pee_path}'):
            match = pattern.match(item)
            if match:
                # Extract the number (as an integer) and store it
                sims.append(int(match.group(1)))

        print(f'There are {len(sims)} sims available to parse! Getting started...')
        print()

    if args.save_dir:
        save_dir = f'{base_dir}/{kSZ_path}/LoReLi/{args.save_dir}'

    else:
        save_dir = f'{base_dir}/{kSZ_path}/LoReLi/nells{ells.size}'

    # check is a dir exists for this numbers of ells
    # and if not, make one
    
    if not os.path.exists(save_dir):
        # Create the folder
        os.makedirs(save_dir, exist_ok=True)
        print(f"Folder created: {save_dir}")
    else:
        print(f"Folder already exists: {save_dir}") 

    print(f"Running through sims in {args.n}th file")

    sims_empty = []
    sims_nan = []
    sims_failedreion = []
    print(f'Now simulating {len(sims)} kSZ spectra!')
    for j, sn in enumerate(sims):

        start_time = time.time()
        print('==================================')
        print(f'Now on the {j+1}th run for sim {sn}')
        print('==================================')

       # fit_fn = f'{fits_path}/bestfit_params_simu{sn}.npz'
    # Gorce_fn = f'{kSZ_path}/Gorce/kSZ_Gorce_simu{sn}'
        LoReLi_fn = f'{save_dir}/kSZ_LoReLi_simu{sn}'

        if os.path.exists(f'{LoReLi_fn}.npz'):
            if os.path.isfile(f'{LoReLi_fn}.npz'):
                print('Spectra already calculated, skipping...')
                continue

        print()
        print('loading params...')
        # if os.path.exists(fit_fn):
        #     if os.path.isfile(fit_fn):
        #         bf = np.load(fit_fn, allow_pickle=True) 
        #     print(fit_fn)
        # else:
        #     print('no fits file! skipping...')
        #     continue

        # print(f'params for sim {sn} loaded...')
        # bf = bf['bf'].item()
        # print(bf)
        # print()
        # alpha0 = bf[str(sn)]['alpha0']
        # kappa = bf[str(sn)]['kappa']
        # print('Checking for redshift file...')
        # if not os.path.isfile(f'{sim_path}/simu{sn}/redshift_list.dat'):
        #     print(f"No redshift file for sim {sn}, skipping...")
        #     sims_noredshiftfile.append(sn)
        #     continue
        

        #data = np.load(f'{Pee_path}/simu{sn}_Pee_spectra.npz', allow_pickle=True)
        sim = Cat(sn, skip_early=False,
                            base_dir=base_dir,
                            path_spectra=spectra_path,
                            path_redshifts=redshift_file,
                            LoReLi_format=True,
                            verbose=False)

        if np.isnan(utils.find_index(sim.xe)):
            print(f'Sim {sn} is missing redshifts! Skipping...')
            sims_empty.append(sn)
            continue

        print('Check cleared...loading data...')

        sim = Cat(sn, skip_early=True,
                            base_dir=base_dir,
                            path_spectra=spectra_path,
                            path_redshifts=redshift_file,
                            LoReLi_format=True,
                            verbose=True)
        
        print('data loaded...')
        print('')

        if np.any(np.isnan(sim.Pee)):
            print(f'Skipping sim {sn} due to nans in data!')
            sims_nan.append(sn)
            continue

        if sim.xe.max() < .97:
            print(f'Sim {sn} does not reach .97 ionisation fraction!')
            sims_failedreion.append(sn)

        print('smoothing Pee...')
        k, Pee = utils.smooth_Pee(sim)

        print('Pee smoothed...')
        print()

        print('simulating Gorce spectrum...')
        print('----------------------------')

        print('skipping Gorce for the moment (the model, never the real the thing!)')
        #sim = Cat(sn, verbose=True)
        # Gorce = alfred.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=False,
        #             Pee_data=None, xe_data=sim.xe, z_data=sim.z, k_data=None, alpha0=alpha0, kappa=kappa,
        #             kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False)
        
        print()
        print('simulating LoReLi spectrum...')
        print('----------------------------')
        
        # LoReLi = alfred.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=True,
        #             Pee_data=sim.Pee, xe_data=sim.xe, z_data=sim.z, k_data=sim.k, alpha0=alpha0, kappa=kappa,
        #             kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False)
        
        Dell = alfred.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=True,
                    Pee_data=Pee, xe_data=sim.xe, z_data=sim.z, k_data=k, helium=True, helium2=True,
                    kmin=k[0], kmax=k[-1], xemin=0.0, xemax=.97, verbose=True)
        
        print()
        
    #  np.savez(Gorce_fn, ells=ells, kSZ=Gorce)
        np.savez(LoReLi_fn, ells=ells, Dell=Dell)
        #np.savez(LoReLi_fn, ells=ells, kSZ=LoReLi)

        print(f'saving spectra for simulation {sn} at {LoReLi_fn}...')
        end_time = time.time()
        print(f"One kSZ run took {(end_time - start_time) / 60.0 :.3f} minutes")
        print(f'{j+1} sims completed, {len(sims)-(j + 1)} to go!')

    np.save(f'sims_nan_{args.n}', sims_nan)
    np.save(f'sims_failedreion_{args.n}', sims_failedreion)
    print('Done, YAY!')

if __name__ == "__main__":
    main()
