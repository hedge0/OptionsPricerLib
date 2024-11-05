from math import log, sqrt, exp
from numba import njit
from OptionsPricerLib.utils import normal_cdf

class BaroneAdesiWhaley:
    """
    Class implementing the Barone-Adesi Whaley model for American options pricing
    and Greeks calculations.
    """

    @staticmethod
    @njit
    def price(sigma, S, K, T, r, q=0.0, option_type='calls'):
        """
        Calculate the price of an American option using the Barone-Adesi Whaley model.
        
        Parameters:
            sigma (float): Implied volatility.
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            q (float, optional): Continuous dividend yield. Defaults to 0.0.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
        
        Returns:
            float: The calculated option price.
        """
        return barone_adesi_whaley_price_helper(sigma, S, K, T, r, q, option_type)

    @staticmethod
    @njit
    def calculate_implied_volatility(option_price, S, K, T, r, q=0.0, option_type='calls', max_iterations=100, tolerance=1e-8):
        """
        Calculate the implied volatility for a given option price.

        Parameters:
            option_price (float): Observed option price (mid-price).
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
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
            price = barone_adesi_whaley_price_helper(mid_vol, S, K, T, r, q, option_type)

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
    def calculate_delta(sigma, S, K, T, r, q=0.0, option_type='calls'):
        """
        Calculate the delta of an option using the Black-Scholes formula with dividend yield.
        
        Parameters:
            sigma (float): Volatility of the underlying asset.
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
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
    def calculate_gamma(sigma, S, K, T, r, q=0.0, option_type='calls'):
        """
        Calculate the gamma of an option.
        
        Parameters:
            sigma (float): Volatility of the underlying asset.
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            q (float, optional): Continuous dividend yield.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
        
        Returns:
            float: The gamma of the option.
        """
        d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        pdf_d1 = exp(-0.5 * d1**2) / sqrt(2 * 3.141592653589793)
        return exp(-q * T) * pdf_d1 / (S * sigma * sqrt(T))

    @staticmethod
    @njit
    def calculate_vega(sigma, S, K, T, r, q=0.0, option_type='calls'):
        """
        Calculate the vega of an option.
        
        Parameters:
            sigma (float): Volatility of the underlying asset.
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            q (float, optional): Continuous dividend yield.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
        
        Returns:
            float: The vega of the option.
        """
        d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        pdf_d1 = exp(-0.5 * d1**2) / sqrt(2 * 3.141592653589793)
        return S * exp(-q * T) * sqrt(T) * pdf_d1

    @staticmethod
    @njit
    def calculate_theta(sigma, S, K, T, r, q=0.0, option_type='calls'):
        """
        Calculate the theta of an option.
        
        Parameters:
            sigma (float): Volatility of the underlying asset.
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            q (float, optional): Continuous dividend yield.
            option_type (str, optional): 'calls' or 'puts'.
        
        Returns:
            float: The theta of the option.
        """
        d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        pdf_d1 = exp(-0.5 * d1 ** 2) / sqrt(2 * 3.141592653589793)
        
        if option_type == 'calls':
            theta = (-S * sigma * exp(-q * T) * pdf_d1 / (2 * sqrt(T))) \
                    - r * K * exp(-r * T) * normal_cdf(d2) \
                    + q * S * exp(-q * T) * normal_cdf(d1)
        elif option_type == 'puts':
            theta = (-S * sigma * exp(-q * T) * pdf_d1 / (2 * sqrt(T))) \
                    + r * K * exp(-r * T) * normal_cdf(-d2) \
                    - q * S * exp(-q * T) * normal_cdf(-d1)
        else:
            raise ValueError("option_type must be 'calls' or 'puts'.")
        
        return theta

    @staticmethod
    @njit
    def calculate_rho(sigma, S, K, T, r, q=0.0, option_type='calls'):
        """
        Calculate the rho of an option.
        
        Parameters:
            sigma (float): Volatility of the underlying asset.
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
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
def barone_adesi_whaley_price_helper(sigma, S, K, T, r, q=0.0, option_type='calls'):
    """
    Helper function to calculate the price of an American option using the Barone-Adesi Whaley model.

    Parameters:
        sigma (float): Implied volatility.
        S (float): Current stock price.
        K (float): Strike price of the option.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate.
        q (float, optional): Continuous dividend yield. Defaults to 0.0.
        option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.

    Returns:
        float: The calculated option price.
    """
    M = 2 * (r - q) / sigma**2
    n = 2 * (r - q - 0.5 * sigma**2) / sigma**2
    q2 = (-(n - 1) - sqrt((n - 1)**2 + 4 * M)) / 2

    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == 'calls':
        european_price = S * exp(-q * T) * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)
        if q >= r or q2 < 0:
            return european_price
        S_critical = K / (1 - 1 / q2)
        if S >= S_critical:
            return S - K
        else:
            A2 = (S_critical - K) * (S_critical**-q2)
            return european_price + A2 * (S / S_critical)**q2

    elif option_type == 'puts':
        european_price = K * exp(-r * T) * normal_cdf(-d2) - S * exp(-q * T) * normal_cdf(-d1)
        if q >= r or q2 < 0:
            return european_price
        S_critical = K / (1 + 1 / q2)
        if S <= S_critical:
            return K - S
        else:
            A2 = (K - S_critical) * (S_critical**-q2)
            return european_price + A2 * (S / S_critical)**q2

    else:
        raise ValueError("option_type must be 'calls' or 'puts'.")
