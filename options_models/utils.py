import numpy as np
from math import exp, log, sqrt
from numba import njit

@njit
def normal_cdf(x):
    """
    Approximation of the cumulative distribution function (CDF) for a standard normal distribution.

    Parameters:
    - x (float): The input value.

    Returns:
    - float: The CDF value.
    """
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))

@njit
def leisen_reimer_ud_p(S, K, T, r, sigma, q, steps, option_type):
    """
    Calculate the up (u), down (d), and probability (p) factors for the Leisen-Reimer model.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price of the option.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate.
        sigma (float): Implied volatility.
        q (float): Continuous dividend yield.
        steps (int): Number of steps in the binomial tree.
        option_type (str): 'calls' or 'puts'.

    Returns:
        Tuple[float, float, float]: (u, d, p) - up factor, down factor, and probability.
    """
    dt = T / steps
    dx = sigma * sqrt(dt)
    
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == 'calls':
        p = _peizer_pratt_inverse(steps, d2)
    else:
        p = 1 - _peizer_pratt_inverse(steps, -d2)
    
    u = exp(dx)
    d = exp(-dx)
    
    return u, d, p

@njit
def _peizer_pratt_inverse(N, x):
    """
    Peizer-Pratt inversion formula for the cumulative binomial distribution.
    
    Parameters:
        N (int): Number of steps in the binomial tree.
        x (float): Adjustment parameter.

    Returns:
        float: Probability adjustment for the Leisen-Reimer model.
    """
    z = x / sqrt(N)
    a = 1.0 / (6.0 * N)
    b = z + a * (z ** 3 - z)
    p = 0.5 + 0.5 * b
    return max(0.0, min(1.0, p))

@njit
def _erf(x):
    """
    Approximation of the error function (erf) using a high-precision method.

    Parameters:
    - x (float): The input value.

    Returns:
    - float: The calculated error function value.
    """
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-x * x))

    return sign * y