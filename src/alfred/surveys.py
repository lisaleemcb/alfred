import numpy as np

import alfred.astrofit
import alfred.surveys as surveys
from alfred.emulator import kemu
from alfred.parameters import base_dir, home_dir
from alfred.utils import summon_emu

telescopes = {
    "SO-LAT": {"fsky": 0.4, "fwhm": 1.5, "noise": 6.0, "fg_bump": 1.0},
    "SO-SAT": {"fsky": 0.1, "fwhm": 10.0, "noise": 2.5, "fg_bump": 1.0},
    "CMB-S4": {"fsky": 0.6, "fwhm": 1.0, "noise": 1.4142, "fg_bump": 1.0},
    "CMB-HD": {"fsky": 0.6, "fwhm": 0.42, "noise": 0.7, "fg_bump": 1.0},
    "cosmic": {"fsky": 1.0, "fwhm": 0.42, "noise": 0.7, "fg_bump": 1.0},
}


def modes(ells, telescope):
    return np.sqrt(2.0 / telescope["fsky"] / (2.0 * ells + 1.0))


def sample_var(ells, Dl, telescope):
    if np.shape(ells) != np.shape(Dl):
        raise ValueError("ekls and Dl must have the same shape.")
    dDl = Dl * modes(ells, telescope)
    return dDl


def noise(ls, telescope, pol=False, is_cl=False):
    ls = np.atleast_1d(ls)
    sig0 = telescope["noise"] / 60.0 * np.pi / 180.0  # arcmin to rad
    fwhm = telescope["fwhm"] / 60.0 * np.pi / 180.0  # arcmin to rad
    nl = sig0**2 / 2.0 * np.exp(ls * (ls + 1.0) * fwhm**2 / 8.0 / np.log(2.0))
    if pol:
        nl *= 2.0
    if not is_cl:
        nl *= ls * (ls + 1.0) / 2.0 / np.pi
    return nl


def emu_error(
    ells,
    file=f"{base_dir}/emulators/setrandomseed3/ensemble_error_v5.1.npy",
    verbose=True,
):
    if verbose:
        print(f"Loading emulator error from {file}...")
    # std
    err = np.load(file)

    # ells_emu = alfred.astrofit.ells
    # emu_err = (np.maximum(np.abs(err[0]), err[1]))**2
    # emu_err = 10.0 * np.interp(ells, ells_emu, emu_err)

    return err


# print(f"cosmic variance: {surveys.sample_var(mcmc_run.ells, mcmc_run.datapoints, telescopes['CMB-HD'])**2}")
# print(f"noise: {err[mcmc_run.lmask]**2}")
# print(f"emu: {surveys.emu_error(ells[indices])[mcmc_run.lmask]}**2")


def error_cov(
    ells,
    datapoints,
    telescope,
    verbose=False,
    sn="12958",
    include_samplevar=False,
    include_noise=False,
    include_emulator=False,
    include_fgresiduals=False,
    binning=True,
    emuerr_file=f"{base_dir}/emulators/setrandomseed3/ensemble_error_v5.1.npy",
    fgres_file=f"{base_dir}/metadata/cmbhd_fgs_coadded_noksz_Dl.txt",
):
    errors = []

    if binning:
        if verbose:
            print(f"binning in ell...")
        delta_ell = ells[1:] - ells[:-1]
        delta_ell = [
            *delta_ell,
            delta_ell[-1],
        ]  # assumes the last bin is same as second to last

    else:
        delta_ell = 1.0

    sigma_CV = np.zeros_like(datapoints)
    if include_samplevar:
        sigma_CV += surveys.sample_var(ells, datapoints, telescope) / np.sqrt(delta_ell)
        if verbose:
            print(f"sample variance: {sigma_CV}")

    sigma_noise = np.zeros_like(datapoints)
    if include_noise:
        if binning:
            if verbose:
                print(f"binning noise in ell...")
            sigma_noise += (
                modes(ells, telescope) * surveys.noise(ells, telescope, pol=False)
            ) / np.sqrt(delta_ell)
        else:
            sigma_noise += surveys.noise(ells, telescope, pol=False)
        if verbose:
            print(f"noise: {sigma_noise}")

    sigma_residuals = np.zeros_like(datapoints)
    if include_fgresiduals:
        #  bump_fg_fraction = telescope['fg_bump']
        fg_residuals = np.genfromtxt(fgres_file).T
        if binning:
            if verbose:
                print(f"binning foreground residuals in ell...")
            sigma_residuals += (
                modes(ells, telescope) * np.interp(ells, fg_residuals[0], fg_residuals[1])
            ) / np.sqrt(delta_ell)

            sigma_residuals *= telescope["fg_bump"]
        else:
            sigma_residuals += np.interp(ells, fg_residuals[0], fg_residuals[1]) * telescope[
                "fg_bump"
            ]

        if verbose:
            print(f"foreground residuals: {sigma_residuals}")
            print(f"foregrounds bumped by: {telescope['fg_bump']}")

    errors.append((sigma_CV + sigma_noise + sigma_residuals) ** 2.0)

    if include_emulator:
        var_emu = surveys.emu_error(ells, file=emuerr_file, verbose=verbose) ** 2.0
        # emu = summon_emu('v5.0_err')
        # emu_err = kemu(alfred.astrofit.df.loc[sn].to_numpy(), **emu)
        if verbose:
            print(f"emulator error: {var_emu}")

        errors.append(var_emu)

    errors = np.sum(errors, axis=0)

    return np.diag(errors)
