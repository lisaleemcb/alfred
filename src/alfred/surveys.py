import numpy as np
import alfred.surveys as surveys

import alfred.astrofit

from alfred.parameters import home_dir, base_dir


telescopes ={
    'SO-LAT': {'fsky':0.4, 'fwhm':1.5, 'noise':6.0},
    'SO-SAT': {'fsky':0.1, 'fwhm':10.0, 'noise':2.5},
    'CMB-S4': {'fsky':0.6, 'fwhm':1.0, 'noise': 1.4142},
    'CMB-HD': {'fsky':0.5, 'fwhm':0.5, 'noise':2.7},
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

def error_cov(ells, datapoints, telescope,
            include_samplevar=True, include_noise=True, include_emulator=True,
            emuerr_file=f'{base_dir}/emulators/LoReLi_settings/NN_LoReLi_errors.npy'):
    delta_ell = np.diff(ells).mean()
    errors = []

    if include_samplevar:
        sample_var = surveys.sample_var(ells, datapoints, telescope)**2
        errors.append(sample_var)
    if include_noise:
        noise = (surveys.noise(ells, telescope, pol=False)/np.sqrt(delta_ell))**2
        errors.append(noise)
    if include_emulator:
        err = np.load(emuerr_file)
        ells_emu = alfred.astrofit.ells
        emu_err = (np.maximum(np.abs(err[0]), err[1]))**2
        emu_err = np.interp(ells, ells_emu, emu_err)

        errors.append(emu_err)

        print(errors)

    errors = np.sum(errors, axis=0)

    return np.diag(errors)

