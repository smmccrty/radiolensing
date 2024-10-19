import numpy as np
from numba import njit, float64
from scipy.integrate import quad, dblquad
from concurrent.futures import ProcessPoolExecutor
import time
import pandas as pd
import astropy.units as u
import astropy.constants as const
import lmfit
from lmfit import Parameters
import logging
import json
from astropy.cosmology import FlatLambdaCDM
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('radiolensing')
logger.setLevel(logging.DEBUG)

# Suppress Numba's debug messages
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# The cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# constants
cluster_alpha = float64(1.2) 
group_alpha = float64(1.6)
SIGMA_CR_FACTOR = (const.c**2 / (4*np.pi*const.G) ).to(u.M_sun/u.Mpc).value
CRITICAL_DENSITY_CONVERSION = (u.g / u.cm**3).to(u.M_sun / u.Mpc**3)

# For output

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
        return float(obj) if isinstance(obj, (np.float64, np.float32)) else int(obj)
    elif hasattr(obj, 'nominal_value'):  # Check for AffineScalarFunc
        return {
            'type': 'AffineScalarFunc',
            'nominal_value': float(obj.nominal_value),
            'std_dev': float(obj.std_dev)
        }
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)  # Convert any other types to strings

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return convert_to_serializable(obj)

def save_results_to_json(filename, **kwargs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(filename, 'w') as f:
        json.dump(kwargs, f, cls=NumpyEncoder, indent=2)

        
##################################
### OPTICAL DEPTH CALCULATIONS ###
##################################

# Root finding algorithm
@njit
def find_root(f, left_bound, right_bound, args=(), epsilon=1e-6, max_iterations=500):
    # Ensure correct ordering of bounds
    if left_bound > right_bound:
        left_bound, right_bound = right_bound, left_bound
    
    a, b = left_bound, right_bound
    fa, fb = f(a, *args), f(b, *args)
    
    # Check if root is at bounds
    if abs(fa) < epsilon:
        return a
    if abs(fb) < epsilon:
        return b
    
    # Check if there's a root in the interval
    if fa * fb >= 0:
        return np.nan
    
    # Initial guess using weighted average
    c = a + (b - a) * abs(fa) / (abs(fa) + abs(fb))
    fc = f(c, *args)
    
    for _ in range(max_iterations):
        if abs(fc) < epsilon or abs(b - a) < epsilon * abs(b + a):
            return c
        
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
            # Illinois algorithm modification
            fb /= 2
        
        # New estimate
        c_new = a + (b - a) * abs(fa) / (abs(fa) + abs(fb))
        
        # If new estimate is too close to a or b, use bisection
        if abs(c_new - a) < epsilon or abs(c_new - b) < epsilon:
            c_new = (a + b) / 2
        
        if abs(c_new - c) < epsilon * abs(c):
            return c_new
        
        c, fc = c_new, f(c_new, *args)
    
    return np.nan

    
# Define general MF
@njit
def schechter_mf(mass, phi_star, a, m_star, b):
    return phi_star * (mass / m_star)**(-a) * np.exp(-(mass / m_star)**b)

#fit the MF
def fit_cosmoDC2(data,num_bins,cosmo):
    hist_data = []
    fitted_params = []
    for i,d in enumerate(data):
        n, binss = np.histogram(d['halo_mass+sum(stellar_mass)'], bins = num_bins)
        n = n * u.steradian.to(u.degree**2) / 440 / cosmo.differential_comoving_volume(i*0.5+.25).value / 0.5
        binss = np.array(binss) + (binss[1]-binss[0])/2
        n = n / (binss[1]-binss[0])
        hist_data.append(n)

        model = lmfit.Model(schechter_mf)
        params = model.make_params()
        params['phi_star'].set(value=1.231e-5, vary=True)
        params['a'].set(value=0.922, vary=True)
        params['m_star'].set(value=2e14, vary=True)
        params['b'].set(value=0.629, vary=True)
        if i > 3 or i ==0:
            fit = model.fit(n[1:], params, mass=binss[1:-1])
        else:
            fit = model.fit(n, params, mass=binss[:-1])
        new_params = fit.params
        fitted_params.append(new_params)
    return fitted_params

#Create grid for interpolation
def create_mf_grid(fitted_params, mass_range, z_range):
    masses = np.logspace(np.log10(mass_range[0]), np.log10(mass_range[1]), 600)
    redshifts = np.linspace(0, z_range[-1], 50)
    mf_values = np.zeros((len(redshifts), len(masses)))
    
    for i, z in enumerate(redshifts):
        params = fitted_params[min(int(z/0.5), len(fitted_params)-1)]
        phi_star = params['phi_star'].value
        a = params['a'].value
        m_star = params['m_star'].value
        b = params['b'].value
        mf_values[i] = schechter_mf(masses, phi_star, a, m_star, b)

    masses = np.array(masses, dtype=np.float64)
    redshifts = np.array(redshifts, dtype=np.float64)
    mf_values = np.array(mf_values, dtype=np.float64)
    return masses, redshifts, mf_values

# Func to do MF interpolation
@njit
def interpolate_mf(mass: float, z: float, masses, redshifts, mf_values):
    i = np.searchsorted(masses, mass) - 1
    j = np.searchsorted(redshifts, z) - 1
    
    i = max(0, min(i, len(masses) - 2))
    j = max(0, min(j, len(redshifts) - 2))
    
    t = (mass - masses[i]) / (masses[i+1] - masses[i])
    u = (z - redshifts[j]) / (redshifts[j+1] - redshifts[j])
    
    return (
        mf_values[j, i] * (1-t) * (1-u) +
        mf_values[j, i+1] * t * (1-u) +
        mf_values[j+1, i] * (1-t) * u +
        mf_values[j+1, i+1] * t * u
    )
    

# NFW Setup
# Following Li & Ostriker (2002)

@njit
def sigma_cr(zl, zs, Dl, Ds, Dls):
    # Dl, Ds, and Dls are assumed to be in Mpc
    # Returns sigma_cr in M_sun/Mpc^2
    return SIGMA_CR_FACTOR * Ds / (Dl * Dls)

@njit
def mu_s(zl, zs, rs, ps, Dl, Ds, Dls):
    # rs is in Mpc, ps is in M_sun/Mpc^3
    # Returns dimensionless mu_s
    return 4 * ps * rs / sigma_cr(zl, zs, Dl, Ds, Dls)

@njit
def gx_integrand(x_prime, z, alpha):
    return x_prime / ((np.sqrt(x_prime**2 + z**2))**alpha * (1 + np.sqrt(x_prime**2 + z**2))**(3-alpha))

@njit
def calculate_gx_grid(alpha, x_min, x_max, num_points):
    log_x_min, log_x_max = np.log10(x_min), np.log10(x_max)
    log_x_values = np.linspace(log_x_min, log_x_max, num_points)
    x_values = 10**log_x_values
    gx_values = np.zeros_like(x_values)
    
    n_x, n_z = 10, 500  # number of points for integration
    z_max = 2.0  # upper limit for z integration (approximating infinity)
    
    for i, x in enumerate(x_values):
        dx = x / n_x
        dz = z_max / n_z
        
        result = 0.0
        for j in range(n_x):
            x_prime = (j + 0.5) * dx
            for k in range(n_z):
                z = (k + 0.5) * dz
                result += gx_integrand(x_prime, z, alpha) * dx * dz
        
        gx_values[i] = result
    
    return x_values, gx_values

# Pre-calculate the grids
x_grid_group, gx_grid_group = calculate_gx_grid(group_alpha, 1e-6, 0.9, 4000)
x_grid_cluster, gx_grid_cluster = calculate_gx_grid(cluster_alpha, 1e-6, 0.9, 4000)

@njit
def gx(x, alpha):
    if alpha == 1:
        if x <= 0:
            return 0.0
        elif x == 1:
            return np.log(0.5) + 1
        elif x < 1:
            return np.log(x/2) + np.arctanh(np.sqrt(1-x**2)) / np.sqrt(1-x**2)
        else:
            return np.log(x/2) + np.arctan(np.sqrt(x**2-1)) / np.sqrt(x**2-1)
    elif alpha == 2:
        if x <= 0:
            return 0.0
        elif x == 1:
            return np.log(0.5) + np.pi * 0.5 + 1
        elif x < 1:
            return np.log(x/2) + np.pi * x /2 + np.arctanh(np.sqrt(1-x**2)) * np.sqrt(1-x**2)
        else:
            return np.log(x/2) + np.pi * x /2 - np.arctan(np.sqrt(x**2-1)) * np.sqrt(x**2-1)
    # Log space interpolation of the gx grids
    elif alpha == group_alpha:
        if x <= x_grid_group[0]:
            return gx_grid_group[0]
        if x >= x_grid_group[-1]:
            return gx_grid_group[-1]
        
        idx = np.searchsorted(x_grid_group, x) - 1
        x0, x1 = x_grid_group[idx], x_grid_group[idx + 1]
        y0, y1 = gx_grid_group[idx], gx_grid_group[idx + 1]
        
        log_x, log_x0, log_x1 = np.log(x), np.log(x0), np.log(x1)
        log_y0, log_y1 = np.log(y0), np.log(y1)
        
        log_y = log_y0 + (log_x - log_x0) * (log_y1 - log_y0) / (log_x1 - log_x0)
        return np.exp(log_y)
    elif alpha == cluster_alpha:
        if x <= x_grid_cluster[0]:
            return gx_grid_cluster[0]
        if x >= x_grid_cluster[-1]:
            return gx_grid_cluster[-1]
        
        idx = np.searchsorted(x_grid_cluster, x) - 1
        x0, x1 = x_grid_cluster[idx], x_grid_cluster[idx + 1]
        y0, y1 = gx_grid_cluster[idx], gx_grid_cluster[idx + 1]
        
        log_x, log_x0, log_x1 = np.log(x), np.log(x0), np.log(x1)
        log_y0, log_y1 = np.log(y0), np.log(y1)
        
        log_y = log_y0 + (log_x - log_x0) * (log_y1 - log_y0) / (log_x1 - log_x0)
        return np.exp(log_y)
        
@njit
def lens_eqn(x, mus, alpha):
    gx_val = gx(x, alpha)
    return x - mus * gx_val / x

@njit
def gx_x_prime(x, alpha):
    if alpha == 1:
        term1 = (x * np.arctanh(np.sqrt(1-x**2)) / (1-x**2)**(3/2) + 1/x - 1 / (x * (1-x**2))) / x
        term2 = (np.log(x/2) + np.arctanh(np.sqrt(1-x**2))/ np.sqrt(1-x**2)) / x**2
        return (term1 - term2)
    elif alpha == 2:
        term1 = ( -x * np.arctanh(np.sqrt(1-x**2)) / np.sqrt(1-x**2) + np.pi/2) / x
        term2 = (np.log(x/2) + np.pi * x /2 + np.arctanh(np.sqrt(1-x**2)) * np.sqrt(1-x**2)) / x**2
        return -1 * (term1 - term2)
    else:
        # Numerical approximation of the derivative
        h = 1e-7
        return (gx(x + h, alpha)/(x+h) - gx(x - h, alpha)/(x-h)) / (2 * h)

@njit
def lens_prime(x, mus, alpha):
    return 1 - mus * gx_x_prime(x, alpha)

# From Dutton and Maccio (2014)
@njit
def c1_func(zl, mass, alpha):
    b = -0.101 + 0.026 * zl
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * zl**1.21)
    return 10**(a + b * np.log10(mass/1e12)) * (2-alpha)

c_simga=0.11
c_interval = 2

@njit
def c_scatter(c, c_bar):
    return (1 / (c * c_simga * np.sqrt(2 * np.pi) * np.log(10))) * np.exp(-(np.log10(c) - np.log10(c_bar))**2 / (2 * c_simga**2))

@njit
def calculate_fc1_grid(alpha, max_c, step):
    grid_size = int(max_c / step) + 1
    c_values = np.linspace(0, max_c, grid_size)
    fc1_values = np.zeros_like(c_values)
    for i, c in enumerate(c_values):
        if c == 0:
            fc1_values[i] = 0
        else:
            range = np.arange(step, c + step, step)
            total = 0
            for c_inner in range:
                total += c_inner**2 / c_inner**alpha / (1+c_inner)**(3-alpha) * step
            fc1_values[i] = total
    return c_values, fc1_values

c_values_group, fc1_values_group = calculate_fc1_grid(group_alpha, 20, 0.001)
c_values_cluster, fc1_values_cluster = calculate_fc1_grid(cluster_alpha, 20, 0.001)

@njit
def fc1_func(c1, alpha):
    if alpha == 1:
        return np.log1p(c1) - c1/(1+c1)
    elif alpha == 2:
        return np.log1p(c1)
    elif alpha == group_alpha: #and 0 <= c1 <= 20:
        # interpolate
        idx = int(c1 * 1000)
        x0, x1 = c_values_group[idx], c_values_group[idx + 1]
        y0, y1 = fc1_values_group[idx], fc1_values_group[idx + 1]
        return y0 + (c1 - x0) * (y1 - y0) / (x1 - x0)
    elif alpha == cluster_alpha: #and 0 <= c1 <= 20:
        # interpolate
        idx = int(c1 * 1000)
        x0, x1 = c_values_cluster[idx], c_values_cluster[idx + 1]
        y0, y1 = fc1_values_cluster[idx], fc1_values_cluster[idx + 1]
        return y0 + (c1 - x0) * (y1 - y0) / (x1 - x0)
    else:
        # Do the actual integral
        range_values = np.arange(0.05, c1+0.05, 0.05)
        integrand = range_values**2 / range_values**alpha / (1+range_values)**(3-alpha)
        return np.sum(integrand) * 0.05

@njit
def ps_func(zl, c1, critical_density_cgs, alpha):
    critical_density = critical_density_cgs * CRITICAL_DENSITY_CONVERSION
    return critical_density * 200/3 * c1**3 / fc1_func(c1, alpha)

@njit
def rs_func(zl, M, c1, critical_density_cgs):
    critical_density = critical_density_cgs * CRITICAL_DENSITY_CONVERSION
    return (3 * M / (800 * np.pi * critical_density))**(1/3) / c1

@njit
def x0(mus, rs, Dl, alpha):
    root = find_root(lens_eqn, 0.9, 5e-4, args=(mus, alpha), epsilon=1e-7, max_iterations=1000)
    return root * rs / Dl if root > 0 and np.isfinite(root) else 0.0

@njit
def inverse_nfw_theta_E_objective(mass, thetaE, zl, zs, Dl, Ds, Dls, critical_density_cgs, alpha):
    computed_thetaE = nfw_theta_E_func(mass, zl, zs, Dl, Ds, Dls, critical_density_cgs, alpha)
    return computed_thetaE - thetaE

@njit
def inverse_nfw_theta_E_func(thetaE, m_lower, m_upper, zl, zs, Dl, Ds, Dls, critical_density_cgs, alpha):
    root = find_root(inverse_nfw_theta_E_objective, m_upper, m_lower, 
                     args=(thetaE, zl, zs, Dl, Ds, Dls, critical_density_cgs, alpha), 
                     epsilon=1e-8, max_iterations=2000)
    return root if root is not None and root > 0 else 0.0

@njit
def nfw_theta_E_func(mass, c, zl, zs, Dl, Ds, Dls, critical_density_cgs, alpha):
    rs = rs_func(zl, mass, c, critical_density_cgs)
    ps = ps_func(zl, c, critical_density_cgs, alpha)
    mus = mu_s(zl, zs, rs, ps, Dl, Ds, Dls)
    x = x0(mus, rs, Dl, alpha)
    return x

@njit
def nfw_cross_section(mus, rs, Dl, alpha):
    x_cr = find_root(lens_prime, 0.9, 5e-4, args=(mus, alpha))
    y_cr = lens_eqn(x_cr, mus, alpha)
    result = -1 * y_cr * rs / Dl 
    return np.pi * result**2 if result is not None and result > 0 else 0.0
    
# Galaxy/SIS setup
# Following Yue et al. (2022)

@njit
def galaxy_vdf(sigma, z):
    a, b = -0.15, 2.35
    sigma0 = 172.2 * (1+z)**0.18
    phis_z0 = 5.86e-3 * np.e / np.log(10)
    phis = phis_z0 * (1+z)**-1.18
    return phis * (sigma/sigma0)**a * np.exp(-(sigma/sigma0)**b) * np.log(10) * np.exp(-(sigma/300)**10)

@njit
def galaxy_vdf_integral(sigma, z):
    return galaxy_vdf(sigma, z) / (np.log(10) * sigma)

@njit
def sis_theta_E_func(sigma, zl, zs, Ds, Dls):
    return 4 * np.pi * (sigma/3e5)**2 * Dls / Ds

@njit
def inverse_sis_theta_E_func(theta_E, zl, zs, Ds, Dls):
    sc2 = theta_E / 4 / np.pi * Ds / Dls
    return np.sqrt(sc2) * 3e5

# Calculating optical depths

@njit
def tau_integral_nfw(mass, z, zs, masses, redshifts, mf_values, thetaE_min, Dl, Ds, Dls, dV, critical_density_cgs, alpha):
    if z >= zs:
        return 0.0
        
    c_bar = c1_func(z, mass, alpha)
    c_range = np.linspace(c_bar/c_interval, c_bar*c_interval, 300)
    rs = rs_func(z, mass, c_range, critical_density_cgs)
    ps = ps_func(z, c_range, critical_density_cgs, alpha)
    mus = mu_s(z, zs, rs, ps, Dl, Ds, Dls)
    
    mfterm = interpolate_mf(mass, z, masses, redshifts, mf_values)
    
    result = 0.0
    for i, c in enumerate(c_range):
        if thetaE_min <= 0 or x0(mus[i], rs[i], Dl, alpha) >= thetaE_min:    
            cross_section = nfw_cross_section(mus[i], rs[i], Dl, alpha)
            c_scatter_value = c_scatter(c_range[i], c_bar)
            result += cross_section * c_scatter_value
    
    return result * (c_range[1] - c_range[0]) * dV * mfterm

@njit
def tau_integral_sis(sigma, z, zs, thetaE_min, Ds, Dls, dV):
    if z >= zs:
        return 0.0
    theta_E = sis_theta_E_func(sigma, z, zs, Ds, Dls)
    if theta_E < thetaE_min:
        return 0.0
    vdfterm = galaxy_vdf_integral(sigma, z)
    area = np.pi * theta_E**2 
    return vdfterm * dV * area

def taum_nfw(zs, m_lower, m_upper, masses, redshifts, mf_values, thetaE_min, cosmo, alpha):
    Ds = cosmo.angular_diameter_distance(zs).value
    
    # Define number of steps for mass and redshift
    n_mass = 1000
    n_redshift = 100
    
    # Calculate step sizes
    d_mass = (m_upper - m_lower) / n_mass
    d_z = zs / n_redshift
    
    mass_array = np.linspace(m_lower + 0.5*d_mass, m_upper - 0.5*d_mass, n_mass)
    z_array = np.linspace(0.5*d_z, zs - 0.5*d_z, n_redshift)
    
    result = np.zeros((n_mass, n_redshift))
    for i in range(n_mass):
        for j in range(n_redshift):
            z = z_array[j]
            
            Dls = cosmo.angular_diameter_distance_z1z2(z, zs).value
            Dl = cosmo.angular_diameter_distance(z).value
            dV = cosmo.differential_comoving_volume(z).value
            critical_density_cgs = cosmo.critical_density(z).value
            
            result[i, j] = tau_integral_nfw(mass_array[i], z, zs, masses, redshifts, mf_values, 
                                            thetaE_min, Dl, Ds, Dls, dV, critical_density_cgs, alpha)
    
    return np.trapz(np.trapz(result, z_array), mass_array)
    
def taum_sis(zs, thetaE_min, cosmo):
    Ds = cosmo.angular_diameter_distance(zs).value
    
    def integrand(z, sigma):
        Dls = cosmo.angular_diameter_distance_z1z2(z, zs).value
        Ds = cosmo.angular_diameter_distance(zs).value
        dV = cosmo.differential_comoving_volume(z).value
        return tau_integral_sis(sigma, z, zs, thetaE_min, Ds, Dls, dV)
    
    result, _ = dblquad(integrand, 0, 850, 0, zs, epsabs=1e-6, epsrel=1e-6) #assuming 850km/s is 1e13M_sun for galaxies by the scaling relation of Yue 2022
    return result

def calc_tau_for_z(z, thetaE_min, masses, redshifts, mf_values, cosmo):
    galaxy_tau = taum_sis(z, 0, cosmo)
    galaxy_tau_observable = taum_sis(z, thetaE_min, cosmo)
        
    cluster_tau = taum_nfw(z, 1e14, 1e16, masses, redshifts, mf_values, 0, cosmo, alpha=cluster_alpha)
    cluster_tau_observable = taum_nfw(z, 1e14, 1e16, masses, redshifts, mf_values, thetaE_min, cosmo, alpha=cluster_alpha)
        
    group_tau = taum_nfw(z, 1e13, 1e14, masses, redshifts, mf_values, 0, cosmo, alpha=group_alpha)
    group_tau_observable = taum_nfw(z, 1e13, 1e14, masses, redshifts, mf_values, thetaE_min, cosmo, alpha=group_alpha)
    print(z)
    
    return (galaxy_tau, galaxy_tau_observable, group_tau, group_tau_observable, cluster_tau, cluster_tau_observable)

def calc_tau_chunk(z_chunk, thetaE_min, masses, redshifts, mf_values, cosmo):
    return [calc_tau_for_z(z, thetaE_min, masses, redshifts, mf_values, cosmo) for z in z_chunk]

def calc_tau_parallel(zs, thetaE_min, masses, redshifts, mf_values, cosmo, chunk_size=3):
    with ProcessPoolExecutor(max_workers=max_proccessors) as executor:
        futures = [executor.submit(calc_tau_chunk, zs[i:i+chunk_size], thetaE_min, masses, redshifts, mf_values, cosmo) 
                   for i in range(0, len(zs), chunk_size)]
        results = [item for future in futures for item in future.result()]

    lists = list(map(list, zip(*results)))  # Transpose results
    save_results_to_json(f'{save_dir}taus', tau_lists=lists)
    return lists


####################
### SOURCE DISTS ###
####################

# Constants
c = 299792458  # Speed of light in m/s
h = 6.62607015e-34  # Planck constant in J*s
k_B = 1.380649e-23  # Boltzmann constant in J/K

# Global variables
sne_alpha = -0.7
sne_mu = 25.5
sne_sigma = 1.5
sne_cutoff = 30
fsrq_alpha = -0.1
bllac_alpha = -0.1
ssagn_alpha = -0.8
sfg_alpha = -0.7

class Source:
    def __init__(self, name, lf, lf_alpha, L_max, freq, L_units, gamma_func, gamma_args, lf_args=None):
        self.name = name
        self.lf = lf
        self.lf_alpha = lf_alpha
        self.lf_args = lf_args
        self.L_max = L_max
        self.freq = freq
        self.L_units = L_units
        self.gamma_func = gamma_func
        self.gamma_args = gamma_args
    
    def make_lf_args(self, z):
        if self.lf_args is not None:
            return self.lf_args
        else:
            return (z,)

    def gamma(self, z):
        return self.gamma_func(z, *self.gamma_args)

# SNe
# From Lien et al. (2011) and Bietenholz et al. (2021)

# Porportional to SFR
@njit
def ccSNe_rate(z):
    a = 3.4
    b = -0.3
    g = -3.5
    z1 = 1
    z2 = 4
    B = pow(1+z1,1-a/b)
    C = pow(1+z1,(b-a)/g) * pow(1+z2,1-b/g)
    n = -10
    return 0.007 * 0.0178 * pow(pow(1+z,a*n) + pow((1+z)/B,b*n) + pow((1+z)/C, g*n), 1/n)

@njit
def norm_pdf_cutoff(x, mu, sigma, cutoff):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(10**(x-cutoff)))

@njit
def sne_lf(L, mu, sigma):
    return norm_pdf_cutoff(L, mu, sigma, sne_cutoff)

# Adjusted for 5GHz data
def L_peak_min_sn(z, alpha):
    return 4 * np.pi * cosmo.luminosity_distance(z).value**2 / (1+z)**(1+alpha) * single_epoch_flux_lim * u.Jansky.to(u.erg / u.second / u.Hz / u.Mpc**2) * (6/1.4)**alpha

def f_survey(z, mu, sigma, alpha):
    result, _ = quad(sne_lf, np.log10(L_peak_min_sn(z, alpha)), 40, args=(mu, sigma), limit=200, epsrel=1e-5) #Luminosity limit of 10^30 for sne
    return result if result > 0 and result is not None else 0

def gamma_sn(z, mu, sigma, alpha):
    return f_survey(z, mu, sigma, alpha) * ccSNe_rate(z) * cosmo.differential_comoving_volume(z).value / 3282.8 * survey_coverage / 3 #steradians to degrees and cadence

# AGN
# Bonaldi et al. 2018

@njit
def double_plaw(L, n0, a, b, l_star):
    return 10**n0 / ((10**(L-l_star))**a + (10**(L-l_star))**b)

@njit
def z_top(L, z_top0, dz_top, l_star_0):
    return z_top0 + dz_top / (1 + 10**(l_star_0 - L))

@njit
def l_star_func(l_star_0, k, z, z_top, m_ev):
    return l_star_0 + (k * z * (2 * z_top - 2 * z**m_ev * z_top**(1-m_ev) / (1 + m_ev)))

#to handle the three agn types
@njit
def lf_helper(L, z, params):
    a, b, n_0, l_star_0, k, z_top0, dz_top, m_ev = params
    ztop = z_top(L, z_top0, dz_top, l_star_0)
    lstar = l_star_func(l_star_0, k, z, ztop, m_ev)
    dL0dL = 1 + 2*k*dz_top*np.log(10)*10**(l_star_0-L)/(1+10**(2*(l_star_0-L))) - \
            2*np.log(10)*(1-m_ev)/(1+m_ev)*dz_top*10**(l_star_0-L) * \
            (z_top0 + dz_top/(1+10**(l_star_0-L)))**(-m_ev) / (1+ 10**(2*(l_star_0-L)))
    return double_plaw(L, n_0, a, b, lstar) * dL0dL

@njit
def fsrq_lf(L, z):
    params = [0.776, 2.669, -8.319, 33.268, 1.234, 2.062, 0.559, 0.136]
    return lf_helper(L, z, params)

@njit
def bllac_lf(L, z):
    params = [0.723, 1.918, -7.165, 32.282, 0.206, 1.262, 0, 1]
    return lf_helper(L, z, params)

@njit
def ssagn_lf(L, z):
    params = [0.508, 2.545, -5.973, 32.560, 1.349, 1.116, 0.705, 0.253]
    return lf_helper(L, z, params)

def gamma_agn(z, alpha, lf):
    L_min = 4 * np.pi * cosmo.luminosity_distance(z).value**2 / (1+z)**(1+alpha) * survey_flux_lim * u.Jansky.to(u.erg / u.second / u.Hz / u.Mpc**2)
    n_den = quad(lf, np.log10(L_min), 40, args=(z,), limit=200, epsrel=1e-5)[0]
    return n_den * cosmo.differential_comoving_volume(z).value / 3282.8 * survey_coverage

# SFRG
# Bonaldi et al. 2018

@njit
def sfr_funcs(z):
    x = np.log10(1+z)
    return {
        'alpha': 1.2 + 0.5*x - 0.5*x**2 + 0.2*x**3,
        'omega': 0.7 - 0.15*x + 0.16*x**2 + 0.01*x**3,
        'star': 10**(1.1 + 3.2*x - 1.4*x**2 - 2.1*x**3),
        'phi': 10**(-2.4 -2.3*x + 6.2*x**2 -4.9*x**3)
    }

@njit
def phi_sfr(sfr, z):
    sf = sfr_funcs(z)
    return sf['phi'] * (sfr / sf['star'])**(1-sf['alpha']) * np.exp(-(sfr/sf['star'])**(sf['omega']))

@njit
def gaunt(nu, T):
    return np.log(np.exp(5.960 - np.sqrt(3) / np.pi * np.log(nu * (T/1e4)**(-1.5))) + np.exp(1))

@njit
def Lff(nu, sfr, T):
    return 3.75e19 * sfr * (T/1e4)**0.3 * gaunt(nu, T) * np.exp(-h * nu * 1e9 / k_B / T)
    
@njit
def Lsyncbar(nu, sfr):
    return 1.9e21 * sfr * nu**(-0.85) * (1 + (nu / 20)**0.5)**(-1)

@njit
def Lsync(nu, sfr, z):
    Lsyncstar = 0.886 * Lsyncbar(nu, 1)
    return Lsyncstar / ((Lsyncstar / Lsyncbar(nu, sfr))**3 + (Lsyncstar / Lsyncbar(nu, sfr))) * 10**(2.35*(1-(1+z)**(-0.12)))

#workaround for determining SFR for given L instead of vice versa
@njit
def sfg_lf_func(sfr, L, z, nu):
    return Lsync(nu, sfr, z) + Lff(nu, sfr, 1e4) - 10**L

@njit
def sfg_lf(L, z):
    L = float(L)
    nu = 1.4
    SFR = find_root(sfg_lf_func, 1e-5, 1e11, args=(L, z, nu), epsilon=1e-6, max_iterations=100)
    return phi_sfr(SFR, z) if SFR is not None else 0

def gamma_sfg(z):
    L_min = 4 * np.pi * cosmo.luminosity_distance(z).value**2 / (1+z)**0.3 * survey_flux_lim * u.Jansky.to(u.Watt / u.Hz / u.Mpc**2)
    n_den = quad(sfg_lf, np.log10(L_min), 27, args=(z,), limit=200, epsrel=1e-5)[0]
    return n_den * cosmo.differential_comoving_volume(z).value / 3282.8 * survey_coverage

# For total count
def gamma_total(z):
    return (gamma_sfg(z) + 
            gamma_agn(z, fsrq_alpha, fsrq_lf) + 
            gamma_agn(z, bllac_alpha, bllac_lf) + 
            gamma_agn(z, ssagn_alpha, ssagn_lf) + 
            gamma_sn(z, sne_mu, sne_sigma, sne_alpha))


sne_class = Source("sne", sne_lf, sne_alpha, 40, 6, u.erg / u.second / u.Hz, gamma_sn, (sne_mu, sne_sigma, sne_alpha), lf_args=(sne_mu, sne_sigma))
fsrq_class = Source("fsrq", fsrq_lf, fsrq_alpha, 40, 1.4, u.erg / u.second / u.Hz, gamma_agn, (fsrq_alpha, fsrq_lf))
bllac_class = Source("bllac", bllac_lf, bllac_alpha, 40, 1.4, u.erg / u.second / u.Hz, gamma_agn, (bllac_alpha, bllac_lf))
ssagn_class = Source("ssagn", ssagn_lf, ssagn_alpha, 40, 1.4, u.erg / u.second / u.Hz, gamma_agn, (ssagn_alpha, ssagn_lf))
sfg_class = Source("sfg", sfg_lf, sfg_alpha, 27, 1.4, u.Watt / u.Hz, gamma_sfg, ())
source_classes = [sfg_class, fsrq_class, bllac_class, ssagn_class, sne_class]

# Assemble all source dists into dict
def calc_source_dists(zs):
    results = {source.name: np.array([source.gamma(z) for z in zs]) for source in source_classes}
    results['agn'] = results['fsrq'] + results['bllac'] + results['ssagn']
    results['total'] = sum(results[key] for key in results if key != 'total')
    
    save_results_to_json(f'{save_dir}source_dists', source_dists=results)


    
################
### MAG BIAS ###
################
    

# NFW magnification bias functions
# Li & Ostriker (2002) Yue et al. (2022) Wyithe et al. (2001)

@njit
def Pmu_nfw(mu, mu_min):
    return 2 * mu_min**2 / mu**3
    
@njit
def Pmu_total(mu):
    return 8 / mu**3

@njit
def Pmu_bright(mu):
    return 2/(mu-1)**3

@njit
def Pmu_faint(mu):
    return 2/(mu+1)**3

def N_Llim(L_min, lf, L_max, args=None, points=None, limit=200, epsrel=1e-5):
    return quad(lf, np.log10(L_min), L_max, args=args, points=points, limit=limit, epsrel=epsrel)[0]

@njit
def nfw_mu_min(z, zs, mass, c, critical_density_cgs, Dl, Ds, Dls, alpha):
    rs = rs_func(z, mass, c, critical_density_cgs)
    ps = ps_func(z, c, critical_density_cgs, alpha)
    mu = mu_s(z, zs, rs, ps, Dl, Ds, Dls)
    cross_section = nfw_cross_section(mu, rs, Dl, alpha)
    x = x0(mu, rs, Dl, alpha)
    y_cr = np.sqrt(cross_section / np.pi)
    if y_cr > 0 and np.isfinite(x):
        return 2 * x / y_cr, cross_section
    else:
        return 0, 0

@njit
def N_Llim_numba(L_min, lf, L_max, lf_args=(), N_steps=100):
    L = np.linspace(np.log10(L_min), L_max, N_steps + 1)
    L_mid = np.sqrt(L[1:] * L[:-1])
    f_values = np.array([lf(L, *lf_args) for L in L_mid])
    integral = np.trapz(f_values, L_mid)
    return integral

def calc_galaxy_mag_bias(z, cosmo, source_class, survey_flux_lim):
    L_min = 4 * np.pi * pow(cosmo.luminosity_distance(z).value, 2) / (1+z)**(1+source_class.lf_alpha) * survey_flux_lim * u.Jansky.to(source_class.L_units / u.Mpc**2) * (source_class.freq/1.4)**source_class.lf_alpha

    def intfunc(mu):
        return N_Llim(L_min/mu, source_class.lf, source_class.L_max, args=source_class.make_lf_args(z)) * Pmu_total(mu)

    def intfunc_faint(mu):
        return N_Llim(L_min/mu, source_class.lf, source_class.L_max, args=source_class.make_lf_args(z)) * Pmu_faint(mu)

    def intfunc_bright(mu):
        return N_Llim(L_min/mu, source_class.lf, source_class.L_max, args=source_class.make_lf_args(z)) * Pmu_bright(mu)

    N_unlensed = N_Llim(L_min, source_class.lf, source_class.L_max, args=source_class.make_lf_args(z))
    B = quad(intfunc, 2, np.inf, limit=200, epsrel=1e-5)[0] / N_unlensed
    B_faint = quad(intfunc_faint, 0, np.inf, limit=200, epsrel=1e-5)[0] / N_unlensed
    B_bright = quad(intfunc_bright, 2, np.inf, limit=200, epsrel=1e-5)[0] / N_unlensed

    return B, B_faint, B_bright

@njit
def calc_halo_mag_biases(z_lens, z, mass, c, critical_density_cgs, Dl, Ds, Dls, alpha, L_min, lf, L_max, lf_args, mu_min):
    N_unlensed = N_Llim_numba(L_min, lf, L_max, lf_args=lf_args)

    N_steps = 300
    mu_range = np.logspace(np.log10(mu_min), np.log10(mu_min + 200), N_steps)
    mu_mid = np.sqrt(mu_range[1:] * mu_range[:-1])
    values = Pmu_nfw(mu_mid, mu_min) * np.array([N_Llim_numba(L_min/mu, lf, L_max, lf_args=lf_args) for mu in mu_mid])
    B_integral = np.trapz(values, mu_mid)
    B = B_integral / N_unlensed

    mu_range_faint = np.logspace(np.log10(mu_min/2), np.log10(mu_min/2 + 200), N_steps)
    mu_mid_faint = np.sqrt(mu_range_faint[1:] * mu_range_faint[:-1])
    values_faint = Pmu_nfw(mu_mid_faint, mu_min/2) * np.array([N_Llim_numba(L_min/mu, lf, L_max, lf_args=lf_args) for mu in mu_mid_faint])
    B_integral_faint = np.trapz(values_faint, mu_mid_faint)
    B_faint = B_integral_faint / N_unlensed
    
    return B, B_faint

def calc_mag_biases_for_z(args):
    z, cosmo, masses, redshifts, mf_values = args
    t0 = time.time()
    results = {}

    # Calculate galaxy magnification biases
    for source_class in source_classes:
        results[source_class.name] = list(calc_galaxy_mag_bias(z, cosmo, source_class, survey_flux_lim))

    # Precompute cosmological values
    z_min = 1e-3
    n_steps = 10
    z_range = np.linspace(z_min, z - (z - z_min) / n_steps, n_steps)
    dz = (z - z_min) / n_steps
    Dl_values = cosmo.angular_diameter_distance(z_range).value
    Ds = cosmo.angular_diameter_distance(z).value
    Dls_values = cosmo.angular_diameter_distance_z1z2(z_range, z).value
    critical_density_cgs_values = cosmo.critical_density(z_range).value
    dV_values = cosmo.differential_comoving_volume(z_range).value

    # Calculate group and cluster magnification biases
    for halo_type in ['group', 'cluster']:
        m_lower = 1e13 if halo_type == 'group' else 1e14
        m_upper = 1e14 if halo_type == 'group' else 1e16
        alpha = group_alpha if halo_type == 'group' else cluster_alpha

        m_steps = 100
        mass_range = np.linspace(m_lower, m_upper, m_steps)
        dm = (m_upper - m_lower) / m_steps

        weighted_B = {source_class.name: [0.0, 0.0] for source_class in source_classes}
        total_weight = 0.0

        for i, z_lens in enumerate(z_range):
            for mass in mass_range:
                c_bar = c1_func(z_lens, mass, alpha)
                mfterm = interpolate_mf(mass, z_lens, masses, redshifts, mf_values)

                c_range = np.linspace(c_bar/c_interval, c_bar*c_interval, 10)
                for c in c_range:
                    mu_min, cross_section = nfw_mu_min(z_lens, z, mass, c, critical_density_cgs_values[i], Dl_values[i], Ds, Dls_values[i], alpha)

                    if mu_min == 0 or cross_section == 0:
                        continue

                    c_scatter_value = c_scatter(c, c_bar)
                    weight = cross_section * c_scatter_value * mfterm * dV_values[i] * dm * dz * (c_range[1]-c_range[0])

                    for source_class in source_classes:
                        L_min = 4 * np.pi * pow(cosmo.luminosity_distance(z).value, 2) / (1+z)**(1+source_class.lf_alpha) * survey_flux_lim * u.Jansky.to(source_class.L_units / u.Mpc**2) * (source_class.freq/1.4)**source_class.lf_alpha
                        
                        B, B_faint = calc_halo_mag_biases(z_lens, z, mass, c, critical_density_cgs_values[i], 
                                                                Dl_values[i], Ds, Dls_values[i], alpha, 
                                                                L_min, source_class.lf, source_class.L_max, source_class.make_lf_args(z), mu_min)
                        weighted_B[source_class.name][0] += B * weight
                        weighted_B[source_class.name][1] += B_faint * weight

                    total_weight += weight

        for source_class in source_classes:
            results[source_class.name].extend([b / total_weight if total_weight > 0 else 0.0 for b in weighted_B[source_class.name]])

    t1 = time.time()
    logger.debug(f"Completed calculations for z={z} in {t1-t0:.2f} seconds \nResults:{results}")
    return results

def calc_mag_biases_parallel(zs, cosmo, masses, redshifts, mf_values, max_workers):
    args = [(z, cosmo, masses, redshifts, mf_values) for z in zs]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(calc_mag_biases_for_z, args))

    # Reorganize results
    mag_biases = {k: [r[k] for r in results] for k in results[0].keys()}
    save_results_to_json(f'{save_dir}mag_biases', mag_biases=mag_biases)
    return






if __name__ == "__main__":

    total_time0 = time.time()
    # Maximum number of processors for multiprocessing. On 10 processors entire simulation time is several hours
    max_proccessors = 10
    
    save_dir = './SKA_simulation/VLASS/12_16_10sigma_newvdf/'
    
    # The source redshifts we are using
    dz = 0.5
    bins = np.arange(0, 9 + dz, dz)
    zs = [x+dz/2 for x in bins[:-1]]

    #survey specs
    survey_theta_E_lim = 1.1 * u.arcsec.to(u.radian) # Minimum Einsteiin radius. 1/3 PSF size
    survey_flux_lim = 0.000010 # Jy
    single_epoch_flux_lim = 0.000020 # Jy, For transients 
    survey_coverage = 30000 #deg
    
    # Read in queried cosmoDC2 data, this is all halos where 1e13<(mass halo + mass of member galaxies)<1e15, split into 6 redshift bins
    z0to05 = pd.read_csv('m13to15z0to0.5.dat')
    z05to1 = pd.read_csv('m13to15z0.5to1.dat')
    z1to15 = pd.read_csv('m13to15z1to1.5.dat')
    z15to2 = pd.read_csv('m13to15z1to1.5.dat')
    z2to25 = pd.read_csv('m13to15z2to2.5.dat')
    z25to3 = pd.read_csv('m13to15z2.5to3.dat')
    cosmo_data = [z0to05, z05to1, z1to15, z15to2, z2to25, z25to3]
    
    # Fit CosmoDC2 data and create the mass function grid for interpolation
    num_bins = 100
    fitted_params = fit_cosmoDC2(cosmo_data, num_bins, cosmo)
    mass_range = (1e13, 1e16)
    z_range = (0, bins[-1])
    masses, redshifts, mf_values = create_mf_grid(fitted_params, mass_range, z_range)

    #save settings
    save_results_to_json(f'{save_dir}settings', save_dir=save_dir,
    survey_specs= {'survey_theta_E_lim':survey_theta_E_lim, 
                   'survey_flux_lim':survey_flux_lim, 
                   'single_epoch_flux_lim':single_epoch_flux_lim, 
                   'survey_coverage':survey_coverage}, 
    bins=bins,
    dz=dz,
    zs=zs)

    
    logger.info("Starting tau calculations...")
    start_time = time.time()
    calc_tau_parallel(zs, survey_theta_E_lim, masses, redshifts, mf_values, cosmo)
    end_time = time.time()
    logger.info(f"Calculation time: {end_time - start_time:.2f} seconds")

    logger.info("Starting source calculations...")
    start_time = time.time()
    calc_source_dists(zs)
    end_time = time.time()
    logger.info(f"Calculation time: {end_time - start_time:.2f} seconds")

    logger.info("Starting magnification bias calculations...")
    start_time = time.time()
    calc_mag_biases_parallel(zs, cosmo, masses, redshifts, mf_values, max_proccessors)
    end_time = time.time()
    logger.info(f"Magnification bias calculation time: {end_time - start_time:.2f} seconds")
    
    total_time1 = time.time()
    logger.info(f"Total simulation time: {total_time1 - total_time0:.2f} seconds") 