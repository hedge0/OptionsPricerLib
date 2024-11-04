from math import exp, log, sqrt
from numba import njit
from options_models.utils import normal_cdf

class LeisenReimer:
    """
    Class implementing the Leisen-Reimer binomial model for American options pricing
    and Greeks calculations.
    """

    @staticmethod
    @njit
    def price(S, K, T, r, sigma, q=0.0, option_type='calls', steps=100):
        """
        Calculate the price of an American option using the Leisen-Reimer binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Implied volatility.
            q (float, optional): Continuous dividend yield. Defaults to 0.0.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The calculated option price.
        """
        dt = T / steps
        discount = exp(-r * dt)
        u, d, p = LeisenReimer._calculate_ud_p(r, sigma, q, dt, steps, option_type)

        prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
        values = [max(price - K, 0) if option_type == 'calls' else max(K - price, 0) for price in prices]

        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                values[j] = discount * (p * values[j + 1] + (1 - p) * values[j])
                exercise_value = max((prices[j] - K) if option_type == 'calls' else (K - prices[j]), 0)
                values[j] = max(values[j], exercise_value)
                prices[j] = prices[j] * d

        return values[0]

    @staticmethod
    @njit
    def calculate_implied_volatility(option_price, S, K, r, T, q=0.0, option_type='calls', steps=100, max_iterations=100, tolerance=1e-8):
        """
        Calculate the implied volatility for a given American option price using the Leisen-Reimer model.
        
        Parameters:
            option_price (float): Observed option price (mid-price).
            S (float): Current stock price.
            K (float): Strike price of the option.
            r (float): Risk-free interest rate.
            T (float): Time to expiration in years.
            q (float, optional): Continuous dividend yield. Defaults to 0.0.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
            max_iterations (int, optional): Maximum number of iterations for the bisection method. Defaults to 100.
            tolerance (float, optional): Convergence tolerance. Defaults to 1e-8.
        
        Returns:
            float: The implied volatility.
        """
        lower_vol = 1e-5
        upper_vol = 10.0

        for _ in range(max_iterations):
            mid_vol = (lower_vol + upper_vol) / 2
            price = LeisenReimer.price(S, K, T, r, mid_vol, q, option_type, steps)

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
    def calculate_delta(S, K, T, r, sigma, q=0.0, option_type='calls', steps=100):
        """
        Calculate the delta of an option using the Leisen-Reimer binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Implied volatility.
            q (float, optional): Continuous dividend yield. Defaults to 0.0.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The delta of the option.
        """
        dt = T / steps
        u, d, p = LeisenReimer._calculate_ud_p(r, sigma, dt, steps, option_type)

        price_up = LeisenReimer.price(S * u, K, T, r, sigma, q, option_type, steps)
        price_down = LeisenReimer.price(S * d, K, T, r, sigma, q, option_type, steps)

        return (price_up - price_down) / (S * (u - d))

    @staticmethod
    @njit
    def calculate_gamma(S, K, T, r, sigma, q=0.0, option_type='calls', steps=100):
        """
        Calculate the gamma of an option using the Leisen-Reimer binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Implied volatility.
            q (float, optional): Continuous dividend yield. Defaults to 0.0.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The gamma of the option.
        """
        dt = T / steps
        u, d, p = LeisenReimer._calculate_ud_p(r, sigma, dt, steps, option_type)

        price_up = LeisenReimer.price(S * u, K, T, r, sigma, q, option_type, steps)
        price_down = LeisenReimer.price(S * d, K, T, r, sigma, q, option_type, steps)
        price = LeisenReimer.price(S, K, T, r, sigma, q, option_type, steps)

        return (price_up - 2 * price + price_down) / (S ** 2 * (u - d) ** 2)

    @staticmethod
    @njit
    def calculate_vega(S, K, T, r, sigma, q=0.0, option_type='calls', steps=100):
        """
        Calculate the vega of an option using the Leisen-Reimer binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Implied volatility.
            q (float, optional): Continuous dividend yield. Defaults to 0.0.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The vega of the option.
        """
        epsilon = 1e-5
        price_up = LeisenReimer.price(S, K, T, r, sigma + epsilon, q, option_type, steps)
        price_down = LeisenReimer.price(S, K, T, r, sigma - epsilon, q, option_type, steps)

        return (price_up - price_down) / (2 * epsilon)

    @staticmethod
    @njit
    def calculate_theta(S, K, T, r, sigma, q=0.0, option_type='calls', steps=100):
        """
        Calculate the theta of an option using the Leisen-Reimer binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Implied volatility.
            q (float, optional): Continuous dividend yield. Defaults to 0.0.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The theta of the option.
        """
        epsilon = 1e-5
        price = LeisenReimer.price(S, K, T, r, sigma, q, option_type, steps)
        price_epsilon = LeisenReimer.price(S, K, T - epsilon, r, sigma, q, option_type, steps)

        return (price_epsilon - price) / epsilon

    @staticmethod
    @njit
    def calculate_rho(S, K, T, r, sigma, q=0.0, option_type='calls', steps=100):
        """
        Calculate the rho of an option using the Leisen-Reimer binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Implied volatility.
            q (float, optional): Continuous dividend yield. Defaults to 0.0.
            option_type (str, optional): 'calls' or 'puts'. Defaults to 'calls'.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The rho of the option.
        """
        epsilon = 1e-5
        price_up = LeisenReimer.price(S, K, T, r + epsilon, sigma, q, option_type, steps)
        price_down = LeisenReimer.price(S, K, T, r - epsilon, sigma, q, option_type, steps)

        return (price_up - price_down) / (2 * epsilon)

    @staticmethod
    @njit
    def _calculate_ud_p(r, sigma, q, dt, steps, option_type):
        """
        Private method to calculate the up (u), down (d), and probability (p) factors for the Leisen-Reimer model,
        accounting for the dividend yield.

        Parameters:
            r (float): Risk-free interest rate.
            sigma (float): Implied volatility.
            q (float): Continuous dividend yield.
            dt (float): Time increment per step.
            steps (int): Number of steps in the binomial tree.
            option_type (str): 'calls' or 'puts'.

        Returns:
            Tuple[float, float, float]: (u, d, p) - up factor, down factor, and probability.
        """
        u = exp(sigma * sqrt(dt))
        d = 1 / u
        m = (steps + 1) / 2.0

        if option_type == 'calls':
            p = normal_cdf((log(m) + (r - q - 0.5 * sigma ** 2) * dt) / (sigma * sqrt(dt)))
        else:
            p = 1 - normal_cdf((log(m) + (r - q - 0.5 * sigma ** 2) * dt) / (sigma * sqrt(dt)))

        return u, d, p
