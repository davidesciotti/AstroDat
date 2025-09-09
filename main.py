import numpy as np
import matplotlib.pyplot as plt
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from scipy.stats import qmc


def update_cosmo_pars_dict(fid_pars, varied_pars, nreal):
    """update the cosmo_pars dict with the sampled values"""
    cosmo_pars = {}
    for i, par in enumerate(varied_pars):
        cosmo_pars[par] = sample[:, i]
    for par in fid_pars:
        if par not in varied_pars:
            cosmo_pars[par] = np.repeat(fid_pars[par], nreal)
    return cosmo_pars


# ! ====================== SETTINGS ====================== !
nreal = 4200
ndim = 5
# ! ====================== SETTINGS ====================== !

# define fiducials and prior bounds
fid_pars = {
    'omega_b': np.array([0.025]),
    'omega_cdm': np.array([0.11]),
    'h': np.array([0.68]),
    'n_s': np.array([0.97]),
    'ln10^{10}A_s': np.array([3.1]),
    'eta_0': np.array([0.7]),
    'cmin': np.array([2.6]),
    'z': np.array([0.0]),
}
prior_bounds = {
    'omega_b': np.array([fid_pars['omega_b'][0] - 0.01, fid_pars['omega_b'][0] + 0.01]),
    'omega_cdm': np.array(
        [fid_pars['omega_cdm'][0] - 0.1, fid_pars['omega_cdm'][0] + 0.1]
    ),
    'h': np.array([fid_pars['h'][0] - 0.1, fid_pars['h'][0] + 0.1]),
    'n_s': np.array([fid_pars['n_s'][0] - 0.1, fid_pars['n_s'][0] + 0.1]),
    'ln10^{10}A_s': np.array(
        [fid_pars['ln10^{10}A_s'][0] - 0.5, fid_pars['ln10^{10}A_s'][0] + 0.5]
    ),
}

# the parameters to vary
varied_pars = list(fid_pars.keys())[:ndim]


sampler = qmc.LatinHypercube(d=ndim)
sample = sampler.random(n=nreal)

plt.scatter(sample[:, 0], sample[:, 1])
plt.xlabel('omega_b', fontsize=12)
plt.ylabel('omega_cdm', fontsize=12)
plt.show()


# shift and stretch the hypercube values
for i, par in enumerate(varied_pars):
    sample[:, i] = (
        sample[:, i] * (prior_bounds[par][1] - prior_bounds[par][0])
        + prior_bounds[par][0]
    )

# take a look at the omega_b, omega_cdm slice
plt.scatter(sample[:, 0], sample[:, 1])
plt.axvspan(
    prior_bounds['omega_b'][0],
    prior_bounds['omega_b'][1],
    facecolor='lightgray',
    alpha=0.5,
)
plt.axhspan(
    prior_bounds['omega_cdm'][0],
    prior_bounds['omega_cdm'][1],
    facecolor='lightgray',
    alpha=0.5,
)
plt.xlabel('omega_b', fontsize=12)
plt.ylabel('omega_cdm', fontsize=12)
plt.show()


cosmo_pars = update_cosmo_pars_dict(fid_pars, varied_pars, nreal)

emulator = CPJ(probe='mpk_nonlin')
emulator_predictions = emulator.predict(cosmo_pars)

# and compare
for i in range(nreal)[::10]:
    plt.loglog(emulator.modes, emulator_predictions[i, :])
plt.xlabel('$k$ [Mpc$^{-1}]$', fontsize=12)
plt.ylabel('$P_{\mathrm{NL}}(k) [\mathrm{Mpc}^3]$', fontsize=12)
plt.legend(fontsize=12, frameon=False)


