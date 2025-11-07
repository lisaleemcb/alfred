import os, re
import argparse
import logging
import sys
import numpy as np
import copy as cp
import h5py

from scipy.interpolate import CubicSpline, interp1d
from alfred.parameters import modelparams_Gorce2022, base_dir


__author__ = "Lisa McBride"
__copyright__ = "Lisa McBride"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from ksz.skeleton import fib`,
# when using this Python module as a library.


def dimless(k, P):
    return (k**3.0 * P) / (2 * np.pi**2)


def tau(n):
    """Calculates tau given an ionisation history xe(z)

    Args:
        z  (array_like): redshift
        xe (array_like): ionisation history

    Returns:
        z  (array_like): redshift
        tau (array_like): tau (cumulative integral)
    """
    z_unity = 5.0  # redshift at which hydrogen is has an ionisation fraction x_HII=1
    z_HeII = 3.5  # redshift at which helium doubly ionises

    if z.min() <= z_unity:
        z_extra = np.linspace(0.0, z.min(), endpoint=False)
    else:
        z_extra = np.linspace(0.0, z_unity)

    xe_lowz = np.ones_like(z_extra)

    spl = CubicSpline(x, y)

    return z, tau


def xe_allz(z, xe):
    """Calculates tau given an ionisation history xe(z)

    Args:
        z  (array_like): redshift
        xe (array_like): ionisation history

    Returns:
        z  (array_like): redshift
        xe (array_like): extrapolated up to z=0
    """
    xe_recomb = 1.7e-4
    Yp = 0.2453
    not4 = 3.9715  # eta
    fHe = Yp / (not4 * (1 - Yp))

    z_unity = 5.0  # redshift at which hydrogen is has an ionisation fraction x_HII=1
    z_HeII = 3.5  # redshift at which helium doubly ionises

    if z.min() <= z_unity:
        z_extra = np.linspace(0.0, z.min(), endpoint=False)
    else:
        z_extra = np.linspace(0.0, z_unity)

    xe_lowz = np.ones_like(z_extra) * 1.08 + xe_recomb

    z2interpl = np.concatenate((z_extra, np.sort(z)))
    xe2interpl = np.concatenate((xe_lowz, xe[::-1]))

    xe_all = CubicSpline(z2interpl, xe2interpl)

    z_all = np.linspace(0.0, z.max(), 1000)
    add_He = (z_all < 3.5).astype(float) * fHe

    return z_all, np.minimum(xe_all(z_all), (1.08 + xe_recomb)) + add_He


def unpack_data(spectra_dict):
    data = np.zeros((len(spectra_dict), spectra_dict[0]["P_k"].size))

    # if isinstance(zrange, int):
    #     data = spectra[zrange][key][krange[0]:krange[1]]
    for i in range(len(spectra_dict)):
        data[i] = spectra_dict[i]["P_k"]

    return data


def pack_params(pvals, pfit):
    params = cp.deepcopy(modelparams_Gorce2022)
    for i, key in enumerate(pfit):
        params[key] = pvals[i]

    return params


def unpack_params(params, pfit):
    return np.asarray([params[key] for key in pfit])


def find_index(arr):
    for i in range(arr.size - 1):
        a = arr[i:]
        # print(f'array looks like {a}')
        if np.all(a[:-1] < a[1:]):
            return i

    print(
        "No monotonically increasing part of this function. Are you sure this is correct?"
    )
    return np.nan


def get_sims(dir=f"spectra/kSZ/LoReLi/nells30", base_dir=base_dir):
    sims = []

    path = f"{base_dir}/{dir}"

    print(f"parsing {path} ...")

    for filename in os.listdir(path):
        # files_LoReLi.append(filename)
        # print(repr(filename))
        match = re.search(r"\d{5}", filename)
        # print(match.group())
        if match.group() is not None:
            sims.append(match.group())
        else:
            print(f"filename {filename} has no match")

    print(f"{len(sims)} sims available")

    return sims


def spectra(
    sn,
    dir="nells30_v5",
    key="Dell",
    basedir=base_dir,
    prefix="kSZ_LoReLi",
    verbose=True,
):
    fn = f"{base_dir}/spectra/kSZ/LoReLi/{dir}/{prefix}_simu{sn}.npz"
    f = np.load(fn, allow_pickle=True)

    return f[key]


def smooth_Pee(sim):
    k = []
    Pee = []
    for i in range(0, sim.k.size - 1, 2):
        k.append((sim.k[i] + sim.k[i + 1]) / 2.0)
        Pee.append((sim.Pee[:, i] + sim.Pee[:, i + 1]) / 2.0)

    Pee = np.asarray(Pee).T

    return k, Pee


def load_samples(dir, verbose=True, flatten=True):
    if verbose:
        print(f"Loading samples from {dir}...")
    with h5py.File(f"{dir}/saved_chains.h5", "r") as hf:
        samples = np.copy(hf["samples"])
        lp = np.copy(hf["logprob"])

        if flatten:
            samples = samples.reshape(
                (samples.shape[0] * samples.shape[1], samples.shape[2])
            )
            lp = lp.flatten()

    return samples, lp


def summon_emu(dir, base=f"{base_dir}/emulators", verbose=False):
    import joblib
    import keras
    from alfred.emulator import WeightedMSELoss

    path = f"{base}/{dir}"

    scalerX = joblib.load(f"{path}/scalerX.pkl")
    scalerY = joblib.load(f"{path}/scalerY.pkl")
    model = keras.models.load_model(f"{path}/model.keras")

    emu = {"scalerX": scalerX, "scalerY": scalerY, "model": model}

    if verbose:
        print(f"SUMMONING THE EMU!!! From {path}...")

    return emu


def plot_vlines(values, axes, **kwargs):
    ndim = axes.shape[0]

    for j in range(ndim):
        ax = axes[j, j]
        ax.axvline(values[j], **kwargs)

    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(values[xi], **kwargs)
            ax.axhline(values[yi], **kwargs)
            ax.plot(values[xi], **kwargs)


# import matplotlib as m
# cmap = m.cm.get_cmap('Blues')
# norm = m.colors.Normalize(vmin=min_chi2-10, vmax=min_chi2+20.)
# lvs = [min_chi2+2.30,min_chi2+6.17,min_chi2+11.8]
# labels=[r'$68\%$',r'$95\%$',r'$99.7\%$']

# plt.figure()
# CS = plt.contour(kappas,alphas,chi2,levels=lvs,colors=[cmap(norm(lvs[0])),cmap(norm(lvs[1])),cmap(norm(lvs[2]))])#,linewidths=.8,colors='white')
# plt.xlabel(r'$\kappa$ [Mpc$^{-1}$]',fontsize=13)
# # plt.xlim(0.07,0.085)
# plt.ylabel(r'log$\alpha_0$',fontsize=13)
# # plt.ylim(4.05,4.2)
# ax = plt.gca()
# fmt={}
# for l,s in zip(lvs, labels):
#     fmt[l]=s
# ax.clabel(CS,CS.levels,fmt=fmt,inline=True)
# plt.scatter(kappa,alpha0,color='k',marker='+',s=100,lw=1.)

# kappas_sims = np.array([0.093,0.094,0.098,0.100,0.089,0.093])
# alphas_sims = np.array([3.86,3.85,3.80,3.78,3.91,3.87])
# ax.scatter(kappas_sims,alphas_sims,color='navy',label='Our simulations',marker='+', s=80,zorder=10)

# plt.tight_layout()


def xe(z, z_data, xe_data, helium=True, helium2=True, just_H=False):
    """
    From alfred.KSZ.py but for use without class
    Computes model's reionisation history.

    The redshift-asymmetric parameterisation of xe(z) in Douspis+2015
    and the class parameters are used.

    Parameters
    ----------
        z: (array of) float(s)
            Redshift range used to compute the ionisation history.
    """

    from alfred.parameters import (
        fHe,
        xe_recomb,
        helium_fullreion_redshift,
        helium_fullreion_deltaredshift,
    )

    z_data = np.sort(z_data)
    xe_data = np.sort(xe_data)[::-1]
    xe_spline = interp1d(
        z_data, xe_data, axis=0, fill_value="extrapolate"
    )  # CubicSpline(z, xe, axis=0)

    frac = 1.0  # - self.xe_recomb)
    xe_He = 0
    if helium:
        frac = 1.0 + fHe - xe_recomb
        # add second He reionisation
        if helium2:
            assert helium, (
                "Need to set both He reionisation to True, cannot have HeII without HeI"
            )
            a = np.divide(1, z + 1.0)
            deltayHe2 = (
                1.5
                * np.sqrt(1 + helium_fullreion_redshift)
                * helium_fullreion_deltaredshift
            )
            VarMid2 = (1.0 + helium_fullreion_redshift) ** 1.5
            xod2 = (VarMid2 - 1.0 / a**1.5) / deltayHe2
            tgh2 = np.tanh(xod2)  # check if xod<100
            xe_He += (fHe - xe_recomb) * (tgh2 + 1.0) / 2.0
            #  self.xe_He = np.where(z < self.z_early, self.xe_He, 0.0)

        xe_early = np.where(z > z_data.max(), xe_recomb, 0.0)
        xe_reion = frac * np.where(
            (z <= z_data.max()) & (z >= z_data.min()), xe_spline(z), 0.0
        )
        xe_late = np.where(z < z_data.min(), frac, 0.0)

        # print(frac)
        # print(f'early {self.xe_early}')
        # print(f'reion {self.xe_reion}')
        # print(f'late {self.xe_late}')
        # print(f'He {self.xe_He}')

        if just_H:
            return xe_early + (xe_reion + xe_late) / frac

        xe = xe_early + xe_reion + xe_late + xe_He
        # the -1 below is totally ad hoc to make sure it doesn't unnecessarily ruin He reion
        if helium:
            xe = np.where(
                (z < helium_fullreion_redshift - 1)
                & (xe <= (1.0 + 2 * fHe - xe_recomb)),
                (1.0 + 2 * fHe - xe_recomb),
                xe,
            )

    return xe
