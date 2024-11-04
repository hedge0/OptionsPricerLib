from math import log, sqrt, exp
from numba import njit
from options_models.utils import normal_cdf

class BlackScholes:
    """
    Class implementing the Black-Scholes model for European options pricing
    and Greeks calculations.
    """

    @staticmethod
    @njit
    def price(S, K, T, r, sigma, q=0.0, option_type='calls'):
        """
        Calculate the price of a European option using the Black-Scholes model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Implied volatility.
            q (float, optional): Continuous dividend yield. Defaults to 0.0.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
        
        Returns:
            float: The calculated option price.
        """
        return black_scholes_price_helper(S, K, T, r, sigma, q, option_type)

    @staticmethod
    @njit
    def calculate_implied_volatility(option_price, S, K, r, T, q=0.0, option_type='calls', max_iterations=100, tolerance=1e-8):
        """
        Calculate the implied volatility for a given European option price.

        Parameters:
            option_price (float): Observed option price (mid-price).
            S (float): Current stock price.
            K (float): Strike price of the option.
            r (float): Risk-free interest rate.
            T (float): Time to expiration in years.
            q (float, optional): Continuous dividend yield. Defaults to 0.0.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
            tolerance (float, optional): Convergence tolerance. Defaults to 1e-8.

        Returns:
            float: The implied volatility.
        """
        lower_vol = 1e-5
        upper_vol = 10.0

        for _ in range(max_iterations):
            mid_vol = (lower_vol + upper_vol) / 2
            price = black_scholes_price_helper(S, K, T, r, mid_vol, q, option_type)

            if abs(price - option_price) < tolerance:
                return mid_vol

            if price > option_price:
                upper_vol = mid_vol
            else:
                lower_vol = mid_vol

            if upper_vol - lower_vol < tolerance:
                break

        return mid_vol

    @staticmethod
    @njit
    def calculate_delta(S, K, T, r, sigma, q=0.0, option_type='calls'):
        """
        Calculate the delta of a European option.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float, optional): Continuous dividend yield.
            option_type (str, optional): 'calls' or 'puts'.
        
        Returns:
            float: The delta of the option.
        """
        d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        if option_type == 'calls':
            return exp(-q * T) * normal_cdf(d1)
        elif option_type == 'puts':
            return exp(-q * T) * (normal_cdf(d1) - 1)
        else:
            raise ValueError("option_type must be 'calls' or 'puts'.")

    @staticmethod
    @njit
    def calculate_gamma(S, K, T, r, sigma, q=0.0):
        """
        Calculate the gamma of a European option.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float, optional): Continuous dividend yield.
        
        Returns:
            float: The gamma of the option.
        """
        d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        return exp(-q * T) * normal_cdf(d1) / (S * sigma * sqrt(T))

    @staticmethod
    @njit
    def calculate_vega(S, K, T, r, sigma, q=0.0):
        """
        Calculate the vega of a European option.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float, optional): Continuous dividend yield.
        
        Returns:
            float: The vega of the option.
        """
        d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        return S * exp(-q * T) * sqrt(T) * normal_cdf(d1)

    @staticmethod
    @njit
    def calculate_theta(S, K, T, r, sigma, q=0.0, option_type='calls'):
        """
        Calculate the theta of a European option.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float, optional): Continuous dividend yield.
            option_type (str, optional): 'calls' or 'puts'.
        
        Returns:
            float: The theta of the option.
        """
        d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if option_type == 'calls':
            theta = (-S * sigma * exp(-q * T) * normal_cdf(d1) / (2 * sqrt(T))) - (r * K * exp(-r * T) * normal_cdf(d2)) + (q * S * exp(-q * T) * normal_cdf(d1))
        elif option_type == 'puts':
            theta = (-S * sigma * exp(-q * T) * normal_cdf(-d1) / (2 * sqrt(T))) + (r * K * exp(-r * T) * normal_cdf(-d2)) - (q * S * exp(-q * T) * normal_cdf(-d1))
        else:
            raise ValueError("option_type must be 'calls' or 'puts'.")
        return theta

    @staticmethod
    @njit
    def calculate_rho(S, K, T, r, sigma, q=0.0, option_type='calls'):
        """
        Calculate the rho of a European option.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float, optional): Continuous dividend yield.
            option_type (str, optional): 'calls' or 'puts'.
        
        Returns:
            float: The rho of the option.
        """
        d2 = (log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        if option_type == 'calls':
            return K * T * exp(-r * T) * normal_cdf(d2)
        elif option_type == 'puts':
            return -K * T * exp(-r * T) * normal_cdf(-d2)
        else:
            raise ValueError("option_type must be 'calls' or 'puts'.")

@njit
def black_scholes_price_helper(S, K, T, r, sigma, q=0.0, option_type='calls'):
    """
    Helper function to calculate the price of a European option using the Black-Scholes model.

    Parameters:
        S (float): Current stock price.
        K (float): Strike price of the option.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate.
        sigma (float): Implied volatility.
        q (float, optional): Continuous dividend yield. Defaults to 0.0.
        option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.

    Returns:
        float: The calculated option price.
    """
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == 'calls':
        return S * exp(-q * T) * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)
    elif option_type == 'puts':
        return K * exp(-r * T) * normal_cdf(-d2) - S * exp(-q * T) * normal_cdf(-d1)
    else:
        raise ValueError("option_type must be 'calls' or 'puts'.")
    