import os, re, time
import copy as cp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
import corner
import zeus
from scipy.integrate import cumulative_trapezoid
from astropy import cosmology, units, constants

import joblib
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import alfred.emulator as emulator
import alfred.surveys as surveys
import alfred.keras_xe_emul as keras_xe_emul
from alfred.parameters import *
from alfred.utils import get_sims, summon_emu


telescopes ={
    'SO-LAT': {'fsky':0.4, 'fwhm':1.5, 'noise':6.0},
    'SO-SAT': {'fsky':0.1, 'fwhm':10.0, 'noise':2.5},
    'CMB-S4': {'fsky':0.6, 'fwhm':1.0, 'noise': 1.4142},
    'CMB-HD': {'fsky':0.5, 'fwhm':0.5, 'noise':2.7},
}

delta_ell = np.mean(np.diff(ells))
indices = list(np.concatenate([np.arange(ells.size)[3:13], np.arange(ells.size)[13::2]]))
Planck =0.0576 #  0.054
Planck_err = 0.0060 # 0.007
priors2d = np.load(f'{base_dir}/inference/priors/2dpriors.npz')

df = pd.read_pickle(f"{base_dir}/metadata/LoReLi_database_loggedparams.pkl")
df = df.loc[df.index.intersection(get_sims(dir='spectra/kSZ/LoReLi/nells30_v5'))]
# failed = np.load(f'{base_dir}/metadata/sims_failed.npy', allow_pickle=True)
# df = df.drop(failed, errors='ignore')

xe_histories = np.load(f'{base_dir}/metadata/ion_histories_full.npz', allow_pickle=True)
xe_histories = xe_histories['arr_0'].item()

labels = df.columns
priors = np.stack([df.to_numpy().min(axis=0), df.to_numpy().max(axis=0)]).T
sims = cp.deepcopy(df.index.to_numpy())
features = cp.deepcopy(df.to_numpy())

pass_prior = np.load(f'{base_dir}/inference/priors/pass_prior.npy')
pass_Planck = np.load(f'{base_dir}/inference/priors/pass_Planckprior.npy')

emu = summon_emu('v5.0')

p_from_dict = lambda pdict: np.asarray(list(pdict.values()))
p_from_npz = lambda file: np.asarray(list(file['truths'].item().values()))

ztau = np.linspace(0,30,1000)

def xe2tau(z, xe):
        """
        Computes redshift evolution of the model's optical depth.

        Parameters
        ----------
            z: (array of) float(s)
                Redshift range used to compute the optical depth.
        """
        cos = cosmology.FlatLambdaCDM(
            H0=h * 100, Tcmb0=T_cmb, Ob0=Ob_0, Om0=Om_0
        )
        z = np.sort(z)
       # xe = np.sort(xe)[::-1]

        integ = constants.c.value * constants.sigma_T.value * nh * xe / cos.H(z).si.value * (1+z)**2
        # tofz = cumulative_trapezoid(integ[::-1], z, initial=0)[::-1]
        tofz = cumulative_trapezoid(integ, z, initial=0)

        return tofz
    
def whatthetau(params):
    xemu_Planck = keras_xe_emul.xe_emul_array(ztau, params, plot=False)
    tau = xe2tau(ztau, xemu_Planck)[:,-1]

    return tau

def addtau2chains(chain, truths, which_params):
    whatthetau(fill(chain, truths, df.columns[2:]))

    return chain


def lnprior(theta, truths, priors,
            priors2d=priors2d, add2d=True, only3=False,
            Planck=Planck, Planck_err=Planck_err, addPlanck=True,
            verbose=False, debug=False):
        
    if theta.ndim == 1:
        theta = theta[None,:]

    index = 0
    if only3:
        index = 2

    pass1d = np.all((priors[index:,0] <= theta) & (priors[index:,1] >= theta), axis=1)
    pass2d = np.ones_like(pass1d, dtype=bool)
    passPlanck = np.zeros_like(pass1d)

    if add2d:
       # Mmin = theta[:,3] # CAREFUL! currently hardcoded (also log10)
       # tau_SR = theta[:,2]  # CAREFUL! currently hardcoded (also log10)

        Mmin = theta[:,3-index] # CAREFUL! currently hardcoded (also log10)
        tau_SR = theta[:,2-index]  # CAREFUL! currently hardcoded (also log10)

        below = priors2d['m'] * Mmin + priors2d['b_below']
        above = priors2d['m'] * Mmin + priors2d['b_above']

        pass2d =  (below <= tau_SR) & (tau_SR <= above)

    
    if addPlanck:
        if verbose:
            print('I am checking the tau prior!')
        xemu = keras_xe_emul.xe_emul_array(ztau, theta, plot=False)
        tau = xe2tau(ztau, xemu)[:,-1]

        passPlanck = -.5 * (Planck - tau)**2.0 / Planck_err**2.0

        if debug:
            print(f"tau={tau}")
            print(f"pass_Planck={passPlanck}")

    passes = pass1d & pass2d
    passes = np.where(passes, 0, -np.inf)
    passes += passPlanck

    return passes

def draws(ndraws, priors=priors, truths=dict(zip(df.columns, df.mean().to_numpy())),
           add2d=True, addPlanck=False, verbose=False):
    # Extract low and high, reshape to (5, 1) so broadcasting works
    if verbose:
        print(f"drawing {ndraws} samples")
    low = priors[:, 0][:, np.newaxis]   # shape (5, 1)
    high = priors[:, 1][:, np.newaxis]  # shape (5, 1)

    # Generate uniform random numbers of shape (5, 100)
    draws = np.random.uniform(low=low, high=high, size=(5, ndraws)).T

    if add2d:
        if verbose:
            print(f"running samples through 2d prior")
        passes = lnprior(draws, truths, priors, add2d=True, addPlanck=False)

        while np.any(np.isneginf(passes)):
            idx2replace = np.where(np.isneginf(passes))[0]
            moredraws = np.random.uniform(low=low, high=high, size=(5, len(idx2replace) * 2)).T
            morepasses = lnprior(moredraws, truths, priors, add2d=True, addPlanck=False)
            idx2keep = np.where(np.isfinite(morepasses))[0]

            for i, idx in enumerate(idx2keep):
                if i < len(idx2replace):
                  #  print(f"Replacing index {idx2replace[i]} with index {idx}")
                    draws[idx2replace[i]] = moredraws[idx]

            passes = lnprior(draws, truths, priors, add2d=True, addPlanck=False)

    return draws

def fill(guess, truths, which_params):
    if guess.ndim == 1:
            guess = guess[None,:]
        
    theta_dict = cp.deepcopy(truths)
    # this seems complicated but it works even when the listed params are out of order

    if np.any(which_params == 'all'):
        which_params = list(truths.keys())

    theta = np.zeros((guess.shape[0], len(truths)))

    for i, key in enumerate(truths.keys()):
        if key in which_params:
            theta[:,i] = guess[:,list(which_params).index(key)]
        else:
            theta[:,i] = np.ones(guess.shape[0])
            theta[:,i] *= truths[key]

    return theta

def lnprob(guess, model, data, err, truths, priors, Aprior=None,
            which_params='all', vectorize=False,
            priors2d=priors2d, add2d=True,
            Planck=Planck, Planck_err=Planck_err, addPlanck=False,
            debug=False):

    theta = fill(guess, truths, which_params)
    # theta = guess
    
    lp = lnprior(theta, truths, priors,
                priors2d=priors2d, add2d=add2d,
                Planck=Planck, Planck_err=Planck_err,
                addPlanck=addPlanck, debug=debug)
    if np.any(Aprior is not None):
        lp += Aprior

    # if not np.isfinite(lp):
    #     return -np.inf#, 0.
    ln = lnlike(theta, model, data, err)

    if not vectorize:
        return (lp + ln)[0]

    if np.any(np.isnan(lp + ln)):
        print(f"Guess values {theta} are causing a NaN!")
    return lp + ln #, model

def lnlike(theta, model, data, err):
    test = model(theta)

    return -0.5 * ((data - test) ** 2.0 / err**2.0).sum(axis=1)


def chi2_contribution(theta, test, data, err,truths=None, which_params='all'):
   # theta = fill(guess, truths, which_params)
   # test = model(theta)
    return ((data - test) ** 2.0 / err**2.0)

def gelman_rubin_rhat(chains):
    """
    Compute the Gelman-Rubin R-hat statistic for convergence diagnostics.
    
    Parameters:
        chains (ndarray): Shape (n_walkers, n_samples, n_params), where:
            - n_walkers: Number of MCMC chains (walkers)
            - n_samples: Number of samples per chain
            - n_params: Number of parameters

    Returns:
        rhat (ndarray): R-hat values for each parameter.
    """
    chains = np.array(chains)  # Ensure it's an ndarray
    n_walkers, n_samples, n_params = chains.shape

    # Compute the mean of each chain (shape: n_walkers, n_params)
    chain_means = np.mean(chains, axis=1)

    # Compute the variance of each chain (shape: n_walkers, n_params)
    chain_variances = np.var(chains, axis=1, ddof=1)

    # Compute the between-chain variance (B)
    B = np.var(chain_means, axis=0, ddof=1) * n_samples  # Shape: (n_params,)

    # Compute the within-chain variance (W)
    W = np.mean(chain_variances, axis=0)  # Shape: (n_params,)

    # Compute the estimated variance of the target distribution
    var_hat = (W * (n_samples - 1) / n_samples) + (B / n_samples)

    # Compute the Gelman-Rubin R-hat statistic
    rhat = np.sqrt(var_hat / W)

    return rhat

class MCMC:
    def __init__(self,
                    config,
                    ells=None,
                    lmask=None,
                    datapoints=None,
                    p0=None,
                    A=None,
                    Astats=[0.0, .1],
                    emu=None,
                    priors=priors,
                    priors2d=priors2d,
                    Planck=Planck,
                    Planck_err=Planck_err,
                    base_dir=base_dir,
                    dir=None,
                    showfigs=False,
                    dryrun=False,
                    verbose=False):
        
        self.config = config
        self.verbose = verbose
        if self.verbose:
            print(f"initialising instance of MCMC class...")
        self.title = self.config.title
        if dir is None:
            self.mcmc_dir = f"{base_dir}/inference/mcmc_runs/{self.title}"
        elif dir is not None:
            self.mcmc_dir = f"{dir}/{self.title}"
        if self.verbose:
            print(f"placing mcmc results in directory {self.mcmc_dir}")

        self.showfigs = showfigs
        self.dryrun = dryrun
        if self.verbose:
            if self.dryrun:
                print(f"doing a dry run...")
        if not self.dryrun:
            os.makedirs(self.mcmc_dir)

        self.telescope = self.config.survey
        self.ells = ells
        self.datapoints = datapoints

        if lmask is not None:
            self.lmask = lmask
        else:
            self.lmin = self.config.lmin
            self.lmax = self.config.lmax
            self.lmask = np.where((self.lmin <= self.ells) & (self.ells <= self.lmax))[0]

        self.p0 = p0
        self.A = A
        if self.A is not None:
            self.p0 = np.concatenate([self.p0, self.A[:,None]], axis=1)
            if self.verbose:
                print(f"Running with nuisance parameter, A...")
        self.Astats = Astats

        if emu is not None:
            self.emu = emu
        # elif self.config.emu == 'v2':
        #     print(f"using emu: {self.config.emu}")
        #     self.emu = emu_v2
        # elif self.config.emu == 'v3':
        #     print(f"using emu: {self.config.emu}")
        #     self.emu = emu_v3
        # elif self.config.emu == 'v3.1':
        #     print(f"using emu: {self.config.emu}")
        #     self.emu = emu_v3p1
        # elif self.config.emu == 'v4':
        #     print(f"using emu: {self.config.emu}")
        #     self.emu = emu_v4
        # elif self.config.emu == 'input_emu':
        #     print(f"using emu: {self.config.emu}")

        self.addnoise = self.config.addnoise

        self.which_params = self.config.which_params_to_fit
        self.priors = priors
        self.add2d = self.config.add2d
        if self.add2d:
            self.priors2d = priors2d
        elif not self.add2d:
            self.priors2d = None
        self.addPlanck = self.config.addPlanck
        if self.addPlanck:
            self.Planck = Planck
            self.Planck_err = Planck_err
        else:
            self.Planck = None
            self.Planck_err = None
        self.vectorize = self.config.vectorize

        self.nwalkers = self.config.nwalkers
        self.burnin = self.config.burnin
        self.nsteps = self.config.nsteps
        self.ndim = len(self.which_params) + int(np.any(self.A is not None))
        self.rescale_cov = self.config.rescale_cov

        print(f"initialisation complete!")

    def init_data(self):
        if self.verbose:
            print(f'-----------------------------------------')
            print(f"Now initialising data for mcmc run")
            print(f'-----------------------------------------')
        if not self.config.use_data:
            self.truths = {}
            for pname in df.columns:
                self.truths[pname] = getattr(self.config, pname)

        elif self.config.use_data:
            self.truths = df.loc[self.config.sn].to_dict()

        self.theta_true = np.asarray(list(self.truths.values()))

        if self.datapoints is None:
            if self.verbose:
                print('Using emulated data for datapoints')

            self.datapoints = emulator.kemu(self.theta_true, **self.emu)
        else:
            if self.verbose:
                print('Using input data for datapoints')

        self.err_cov = surveys.error_cov(self.ells, self.datapoints, surveys.telescopes[self.telescope])
        self.err =np.sqrt(np.diag(self.err_cov))

        if self.addnoise:
            if self.verbose:
                print(f"adding noise to simulated data assuming {self.telescope} specifications...")

            self.datapoints = self.datapoints + np.random.normal(scale=self.err)

        self.ells = self.ells[self.lmask]
        self.datapoints = self.datapoints[self.lmask]
        self.err = self.err[self.lmask]

        if self.verbose:
            print(f"simulating data with the parameter values:")
            print(self.truths)
            print(f"using emulator version {self.config.emu}...")


    def init_run(self, list_emus=None, savefig=True):
        if self.verbose:
            print(f"Now running initialisation of MCMC run")
            print(f'-----------------------------------------')
            print(f"putting mcmc chains in {self.mcmc_dir}...")
            print()
            print(f"Running the mcmc for {self.title} on the params:")
            print(f"\t{self.which_params}...")
            print(f"True values are {self.truths} for simulation {self.config.sn}")
        
        if list_emus:
            print(f"passed list of emulators...")
            def model(p, lmask=self.lmask, emu=self.emu):
                if p.ndim == 1:
                    return emulator.mechkemu(p, list_emus)[lmask]
                elif p.ndim == 2:
                    return emulator.mechkemu(p, list_emus)[:, lmask]

        elif not list_emus:
            print(f"single emulator mode...")
            def model(p, A=None, lmask=self.lmask, emu=self.emu):
                if A is None:
                    A = np.array([1])
                #     mp = p
                # elif self.A is not None:
                #     A = p[-1]
                #     mp = p[:-1] 

         #       print(f"Inside model A={A}")
                if p.ndim == 1:
                    return A * emulator.kemu(p, **emu)[lmask]
                elif p.ndim == 2:
                    return A[:,None] * emulator.kemu(p, **emu)[:, lmask]
            
        self.model = model
        if not self.dryrun:
            if self.verbose:
                print(f"saving data to ...{self.mcmc_dir}/data")
            np.savez(f"{self.mcmc_dir}/data",
                        p0=self.p0,
                        lmask=self.lmask,
                        truths=self.truths,
                        ells=self.ells,
                        model=self.model(np.asarray(list(self.truths.values()))),
                        datapoints=self.datapoints,
                        err=self.err)


        def lnprob_prepped(params):
            if np.any(self.A is not None):
                model_params = params[:,:-1]
                self.A = params[:,-1] 
                self.Amean = self.Astats[0] * np.ones_like(self.A)
                self.Asigma = self.Astats[1] * np.ones_like(self.A)

                passA = -.5 * (self.A - self.Amean)**2.0 / self.Asigma**2.0
                # print(f"model_params: {model_params}")
                # print(f"resetting A to  A={self.A}")
                # print(f"in keeping with the lp of A: {passA}")
                # print('')
            elif np.any(self.A is None):
                model_params = params
                passA = None

            return lnprob(model_params, lambda p: self.model(p, A=self.A), self.datapoints,
                        self.err, self.truths, self.priors, Aprior=passA,
                        which_params=self.which_params, priors2d=self.priors2d, add2d=self.add2d,
                        Planck=self.Planck, Planck_err=self.Planck_err, addPlanck=self.addPlanck,
                        vectorize=self.vectorize)
            
        self.lnprob_prepped = lnprob_prepped

        fig, ax = plt.subplots()
        if self.verbose:
            print(f"ells: {self.ells}")
            print(f"datapoints: {self.datapoints.shape}")
            print(f"err: {self.err.shape}")

            if self.showfigs or savefig:
                ax.plot(self.ells, self.datapoints, color='green', alpha=.3, label='data')
                ax.plot(self.ells, self.model(self.theta_true, A=None), label='model truth', color='deeppink')
                ax.errorbar(self.ells, self.datapoints, color='green', marker='.', ls='', yerr=self.err, label='observations')
                ax.set_xlabel('ell')
                ax.set_ylabel('Dell')

                fig.legend()


        if savefig:
            fig.savefig(f"{self.mcmc_dir}/data.png")

    def start_run(self, save=True):
        if self.verbose:
            print(f"Now starting MCMC run")
            print(f'-----------------------------------------')

        chains_fn = f"{self.mcmc_dir}/saved_chains.h5"
        save_progress = zeus.callbacks.SaveProgressCallback(chains_fn, ncheck=100)
        autocorr_check = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=50, discard=0.5)
        R_check = zeus.callbacks.SplitRCallback(ncheck=100, epsilon=0.01, nsplits=2, discard=0.5)
        miniter_check = zeus.callbacks.MinIterCallback(nmin=500)

        if self.p0 is None:
            _p0 = pass_Planck[:12]
            # p0 = pass_prior[50:62]
            # p0[6] = pass_prior[52]
            #p0 = pass_prior[100:112]
            _p0 = _p0[:,np.where(np.isin(labels, self.which_params))[0]]
                
        elif self.p0 is not None:
            _p0 = self.p0

        if self.verbose:
            print(f"fitting tau prior is {self.addPlanck}...")
            print(f"evaluating likelihood function in vector mode is {self.vectorize}...")

            print('========================================================')

            print('Okay, here we go!')

        start_time = time.time()

        sampler = zeus.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob_prepped,
                                        vectorize=self.vectorize)

        sampler.run_mcmc(_p0, self.burnin)
        burnin_samples = sampler.get_chain()
        start = burnin_samples[-1] # Get the burnin samples

        end_time = time.time()

        if self.verbose:
            print('--------------------------------------------------------')
            print(f'Burn in phase took {(end_time - start_time) / 60:.3f} minutes...')
            print(f'Starting proper run...')
            print('--------------------------------------------------------')

        start_time = time.time()

        self.sampler = zeus.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob_prepped,
                #      args=[datapoints, err, theta_true, lmask, which_params],
                        moves=zeus.moves.GlobalMove(self.rescale_cov), vectorize=self.vectorize)
        
        if self.dryrun:
            callbacks = None
        
        elif not self.dryrun:
            callbacks = [save_progress, autocorr_check, R_check, miniter_check]
        
        self.sampler.run_mcmc(start, self.nsteps, callbacks=callbacks)
        end_time = time.time()

        if self.verbose:
            print(f'finished MCMC in {(end_time - start_time) / (60 * 60):.2} hours')
            if save:
                print(f'saving files in {self.mcmc_dir}...')

        if save:
            np.save(f'{self.mcmc_dir}/burnin', burnin_samples)
            np.save(f'{self.mcmc_dir}/tau', autocorr_check.estimates)
            np.save(f'{self.mcmc_dir}/R', R_check.estimates)

            fig = corner.corner(self.sampler.get_chain(flat=True)[:,:3],
                                 truths=list(self.truths.values())[5-len(self.which_params):])

            # Extract the axes
            ndim = len(list(self.truths.values())[5-len(self.which_params):])
            axes = np.array(fig.axes).reshape((ndim, ndim))

            # Loop over the diagonal
            for i in range(ndim):
                ax = axes[i, i]
                ax.axvline(self.sampler.get_chain(flat=True)[:,i].mean(), color="deeppink")
            fig.savefig(f'{self.mcmc_dir}/corner.png')

        if self.verbose:
            print('Done, YAY!')

        return self.sampler

