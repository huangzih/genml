"""
BSD 3-Clause License

Copyright (c) 2017, Konrad Hinsen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


"""
functions to calculate the numerical value of Mittag Leffer function

Refs:
[1] R. Garrappa, Numerical evaluation of two and three parameter Mittag-Leffler functions, SIAM J. Numer. Anal. 53 (2015) 1350-1369.
[2] GitHub, mittag-leffler, https://github.com/khinsen/mittag-leffler, 2017.
"""

import numpy as np
from scipy.special import gamma

def ml(z, alpha, beta=1., gama=1.):
    """
    Mittag-Leffler function.

    Parameters:
    - z: Complex number or numpy array of complex numbers.
    - alpha: Real part of the exponent parameter.
    - beta: Real part of another parameter (default 1).
    - gama: Real part of the scale parameter (default 1).

    Returns the value of the Mittag-Leffler function.
    """
    eps = np.finfo(np.float64).eps
    if np.real(alpha) <= 0 or np.real(gama) <= 0 or np.imag(alpha) != 0. \
       or np.imag(beta) != 0. or np.imag(gama) != 0.:
        raise ValueError('ALPHA and GAMA must be real and positive. BETA must be real.')
    if np.abs(gama-1.) > eps:
        if alpha > 1.:
            raise ValueError('GAMMA != 1 requires 0 < ALPHA < 1')
        if (np.abs(np.angle(np.repeat(z, np.abs(z) > eps))) <= alpha*np.pi).any():
            raise ValueError('|Arg(z)| <= alpha*pi')
    return np.vectorize(ml_, [np.float64])(z, alpha, beta, gama)


def ml_(z, alpha, beta, gama):
    """
    Helper function for Mittag-Leffler function.

    Parameters:
    - z: Complex number.
    - alpha: Real part of the exponent parameter.
    - beta: Real part of another parameter.
    - gama: Real part of the scale parameter.

    Returns a more precise value of the Mittag-Leffler function for given parameters.
    """
    # Target precision 
    log_epsilon = np.log(1.e-15)
    # Inversion of the LT
    if np.abs(z) < 1.e-15:
        return 1/gamma(beta)
    else:
        return laplace_transform_inversion(1, z, alpha, beta, gama, log_epsilon)


def laplace_transform_inversion(time, lambda_val, alpha, beta, gamma, log_epsilon):
    """
    Inversion of Laplace Transform.

    Parameters:
    - time: Time value for the inversion.
    - lambda_val: Lambda value in the Laplace domain.
    - alpha, beta, gamma: Parameters for the Mittag-Leffler function.
    - log_epsilon: Logarithm of the precision epsilon.

    Returns the inverse Laplace transform at the specified time.
    """
    # Calculate poles
    theta = np.angle(lambda_val)
    k_min = np.ceil(-alpha / 2. - theta / (2 * np.pi))
    k_max = np.floor(alpha / 2. - theta / (2 * np.pi))
    k_vals = np.arange(k_min, k_max + 1)
    poles = np.abs(lambda_val)**(1. / alpha) * np.exp(1j * (theta + 2 * k_vals * np.pi) / alpha)

    # Calculate phi for each pole and sort
    phi_poles = (np.real(poles) + np.abs(poles)) / 2
    sorted_indices = np.argsort(phi_poles)
    poles, phi_poles = poles[sorted_indices], phi_poles[sorted_indices]

    # Filter out poles with phi near zero
    valid_indices = phi_poles > 1.0e-15
    poles, phi_poles = poles[valid_indices], phi_poles[valid_indices]

    # Include origin in singularities
    poles = np.hstack([[0], poles])
    phi_poles = np.hstack([[0], phi_poles])
    num_poles = len(poles) - 1

    # Calculate singularity strengths
    p = gamma * np.ones(len(poles))
    p[0] = max(0, -2 * (alpha * gamma - beta + 1))
    q = gamma * np.ones(len(poles))
    q[-1] = np.inf
    phi_poles = np.hstack([phi_poles, [np.inf]])

    # Find admissible regions considering round-off errors
    round_off_limit = (log_epsilon - np.log(np.finfo(np.float64).eps)) / time
    admissible_regions = np.where((phi_poles[:-1] < round_off_limit) & (phi_poles[:-1] < phi_poles[1:]))[0]

    # Initialize optimal parameters
    optimal_region = admissible_regions[-1]
    mu_vals, N_vals, h_vals = np.inf * np.ones(optimal_region + 1), np.inf * np.ones(optimal_region + 1), np.inf * np.ones(optimal_region + 1)

    # Find optimal parameters
    region_found = False
    while not region_found:
        for region in admissible_regions:
            if region < len(poles) - 1:
                mu, h, N = OptimalParam_RB(time, phi_poles[region], phi_poles[region + 1], p[region], q[region], log_epsilon)
            else:
                mu, h, N = OptimalParam_RU(time, phi_poles[region], p[region], log_epsilon)
            mu_vals[region], h_vals[region], N_vals[region] = mu, h, N

        if N_vals.min() > 200:
            log_epsilon += np.log(10)
        else:
            region_found = True

    # Select optimal integration region
    optimal_index = np.argmin(N_vals)
    N, mu, h = N_vals[optimal_index], mu_vals[optimal_index], h_vals[optimal_index]

    # Calculate inverse Laplace transform
    k_vals = np.arange(-N, N + 1)
    u_vals = h * k_vals
    z_vals = mu * (1j * u_vals + 1.)**2
    zd_vals = -2. * mu * u_vals + 2j * mu
    zexp_vals = np.exp(z_vals * time)
    F_vals = z_vals**(alpha * gamma - beta) / (z_vals**alpha - lambda_val)**gamma * zd_vals
    S_vals = zexp_vals * F_vals
    integral = h * np.sum(S_vals) / (2 * np.pi * 1j)

    # Calculate residues
    residues = np.sum(1. / alpha * poles[optimal_index + 1:]**(1 - beta) * np.exp(time * poles[optimal_index + 1:]))

    # Final value
    E = integral + residues
    if np.imag(lambda_val) == 0:
        E = np.real(E)
    return E


def OptimalParam_RB(t, phi_s_star_j, phi_s_star_j1, pj, qj, log_epsilon):
    """
    Optimal parameter calculation for RB (Region Bounded).

    Parameters:
    - t: Time value.
    - phi_s_star_j, phi_s_star_j1: Phi values for the calculation.
    - pj, qj: Parameters for the calculation.
    - log_epsilon: Logarithm of the precision epsilon.

    Returns optimal parameters for the region bounded inversion.
    """
    log_eps, fac = -36.043653389117154, 1.01
    f_max, threshold = np.exp(log_epsilon - log_eps), 2 * np.sqrt((log_epsilon - log_eps) / t)

    # Calculate square roots and check for admissible region
    sq_phi_star_j, sq_phi_star_j1 = np.sqrt(phi_s_star_j), min(np.sqrt(phi_s_star_j1), threshold - np.sqrt(phi_s_star_j))
    f_min = fac * ((sq_phi_star_j / (sq_phi_star_j1 - sq_phi_star_j)) ** max(pj, qj) if sq_phi_star_j > 0 else fac)
    f_bar = min(f_min + f_min / f_max * (f_max - f_min), f_max)

    if f_min < f_max:
        w = compute_w(phi_s_star_j1, t, log_epsilon, pj, qj)
        factor = f_bar ** (-1 / max(pj, qj))
        den = 2 + w - (1 + w) * factor + factor
        sq_phibar_star_j = ((2 + w + factor) * sq_phi_star_j + factor * sq_phi_star_j1) / den
        sq_phibar_star_j1 = ((2 + w - (1 + w) * factor) * sq_phi_star_j1 - (1 + w) * factor * sq_phi_star_j) / den
        muj = ((1 + w) * sq_phibar_star_j + sq_phibar_star_j1) ** 2 / (2 + w)
        hj = -2 * np.pi / log_epsilon * (sq_phibar_star_j1 - sq_phibar_star_j) / ((1 + w) * sq_phibar_star_j + sq_phibar_star_j1)
        Nj = np.ceil(np.sqrt(1 - log_epsilon / t / muj) / hj)
    else:
        muj, hj, Nj = 0.0, 0.0, np.inf
    return muj, hj, Nj


def compute_w(phi_s_star_j1, t, log_epsilon, pj, qj):
    """
    Compute w parameter for optimal parameter calculation.

    Parameters:
    - phi_s_star_j1: Phi value.
    - t: Time value.
    - log_epsilon: Logarithm of the precision epsilon.
    - pj, qj: Parameters for the calculation.

    Returns the computed w value.
    """
    return -phi_s_star_j1 * t / log_epsilon if pj >= 1.0e-14 and qj >= 1.0e-14 else -2 * phi_s_star_j1 * t / (log_epsilon - phi_s_star_j1 * t)


def OptimalParam_RU(time, phi_s_optimal_j, p_j, log_epsilon_limit):
    """
    Optimal parameter calculation for RU (Region Unbounded).

    Parameters:
    - time: Time value.
    - phi_s_optimal_j: Phi value for the optimal region.
    - p_j: Parameter for the calculation.
    - log_epsilon_limit: Logarithm of the precision epsilon limit.

    Returns optimal parameters for the region unbounded inversion.
    """
    # Calculate square roots of phi values
    sq_phi_s_optimal_j = np.sqrt(phi_s_optimal_j)
    phi_bar_optimal_j = phi_s_optimal_j * 1.01 if phi_s_optimal_j > 0 else 0.01
    sq_phi_bar_optimal_j = np.sqrt(phi_bar_optimal_j)
    # Constants
    f_min, f_max, f_target = 1, 10, 5
    # Iteratively find f_bar within the range [f_min, f_max]
    while True:
        phi_time = phi_bar_optimal_j * time
        log_epsilon_phi_time = log_epsilon_limit / phi_time
        N_j = np.ceil(phi_time / np.pi * (1. - 1.5 * log_epsilon_phi_time + np.sqrt(1 - 2 * log_epsilon_phi_time)))
        A_j = np.pi * N_j / phi_time
        sq_mu_j = sq_phi_bar_optimal_j * np.abs(4 - A_j) / np.abs(7 - np.sqrt(1 + 12 * A_j))
        f_bar = ((sq_phi_bar_optimal_j - sq_phi_s_optimal_j) / sq_mu_j) ** (-p_j)
        if p_j < 1.0e-14 or f_min < f_bar < f_max:
            break
        sq_phi_bar_optimal_j = f_target ** (-1. / p_j) * sq_mu_j + sq_phi_s_optimal_j
        phi_bar_optimal_j = sq_phi_bar_optimal_j ** 2

    mu_j = sq_mu_j ** 2
    h_j = (-3 * A_j - 2 + 2 * np.sqrt(1 + 12 * A_j)) / (4 - A_j) / N_j

    # Adjust integration parameters for round-off error control
    log_eps = np.log(np.finfo(np.float64).eps)
    threshold = (log_epsilon_limit - log_eps) / time
    if mu_j > threshold:
        Q = f_target ** (-1 / p_j) * np.sqrt(mu_j) if abs(p_j) >= 1.0e-14 else 0
        phi_bar_optimal_j = (Q + np.sqrt(phi_s_optimal_j)) ** 2

        if phi_bar_optimal_j < threshold:
            w = np.sqrt(log_eps / (log_eps - log_epsilon_limit))
            u = np.sqrt(-phi_bar_optimal_j * time / log_eps)
            mu_j = threshold
            N_j = np.ceil(w * log_epsilon_limit / (2 * np.pi * (u * w - 1)))
            h_j = np.sqrt(log_eps / (log_eps - log_epsilon_limit)) / N_j
        else:
            N_j = np.inf
            h_j = 0
    return mu_j, h_j, N_j


