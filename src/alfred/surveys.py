import numpy as np
import alfred.surveys as surveys

import alfred.astrofit

from alfred.parameters import home_dir, base_dir
from alfred.utils import summon_emu
from alfred.emulator import kemu


telescopes ={
    'SO-LAT': {'fsky':0.4, 'fwhm':1.5, 'noise':6.0},
    'SO-SAT': {'fsky':0.1, 'fwhm':10.0, 'noise':2.5},
    'CMB-S4': {'fsky':0.6, 'fwhm':1.0, 'noise': 1.4142},
    'CMB-HD': {'fsky':0.6, 'fwhm':0.42, 'noise':0.7},
}

def sample_var(ls, dl, telescope, bin_width=100):
    if np.shape(ls) != np.shape(dl):
        raise ValueError('ls and dl must have the same shape.')
    dDl = dl * np.sqrt(2./telescope['fsky']/(2.*ls+1.))
    return dDl


def noise(ls, telescope, pol=False, is_cl=False):
    ls = np.atleast_1d(ls)
    sig0 = telescope['noise'] / 60.0 * np.pi / 180.0 # arcmin to rad
    fwhm = telescope['fwhm'] / 60.0 * np.pi / 180.0 # arcmin to rad
    nl = sig0**2 / 2. * np.exp(ls*(ls+1.)*fwhm**2/8./np.log(2.))
    if pol:
        nl *= 2.
    if not is_cl:
        nl *= ls*(ls+1.)/2./np.pi
    return nl

def emu_error(ells, file=f'{base_dir}/emulators/setrandomseed3/emulator_std.npy', verbose=True):
    if verbose:
        print(f"Loading emulator error from {file}...")
    residuals = np.load(file)
    err = np.std(residuals, axis=0)
    # ells_emu = alfred.astrofit.ells
    # emu_err = (np.maximum(np.abs(err[0]), err[1]))**2
    # emu_err = 10.0 * np.interp(ells, ells_emu, emu_err)

    return err

# print(f"cosmic variance: {surveys.sample_var(mcmc_run.ells, mcmc_run.datapoints, telescopes['CMB-HD'])**2}")
# print(f"noise: {err[mcmc_run.lmask]**2}")
# print(f"emu: {surveys.emu_error(ells[indices])[mcmc_run.lmask]}**2")

def error_cov(ells, datapoints, telescope, verbose=False, sn='12958',
            include_samplevar=True, include_noise=True, include_emulator=True,
            emuerr_file=f'{base_dir}/emulators/reduced_dataset/emuv5.0_run0/emu_err.npy'):
    delta_ell = np.diff(ells).mean()
    errors = []

    if include_samplevar:
        sample_var = surveys.sample_var(ells, datapoints, telescope)**2.0
        if verbose:
            print(f"sample variance: {sample_var}")
        errors.append(sample_var)
    if include_noise:
        noise = (surveys.noise(ells, telescope, pol=False)/np.sqrt(delta_ell))**2.0
        if verbose:
            print(f"noise: {noise}")
        errors.append(noise)
    if include_emulator:
        emu_err = surveys.emu_error(ells, file=emuerr_file, verbose=verbose)**2.0
        # emu = summon_emu('v5.0_err')
        # emu_err = kemu(alfred.astrofit.df.loc[sn].to_numpy(), **emu)
        if verbose:
            print(f"emulator error: {emu_err}")

        errors.append(emu_err)

    errors = np.sum(errors, axis=0)

    return np.diag(errors)

