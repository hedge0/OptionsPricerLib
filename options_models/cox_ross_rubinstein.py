from math import exp, sqrt
from numba import njit

class CoxRossRubinstein:
    """
    Class implementing the Cox-Ross-Rubinstein (CRR) binomial model for American options pricing
    and Greeks calculations.
    """

    @staticmethod
    @njit
    def price(S, K, T, r, sigma, q=0.0, option_type='calls', steps=100):
        """
        Calculate the price of an American option using the Cox-Ross-Rubinstein binomial model.
        
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
        u = exp(sigma * sqrt(dt))
        d = 1 / u
        p = (exp((r - q) * dt) - d) / (u - d)

        prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
        values = [max(price - K, 0) if option_type == 'calls' else max(K - price, 0) for price in prices]

        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                price = S * (u ** j) * (d ** (i - j))
                exercise_value = max(price - K, 0) if option_type == 'calls' else max(K - price, 0)
                continuation_value = discount * (p * values[j + 1] + (1 - p) * values[j])
                values[j] = max(exercise_value, continuation_value)

        return values[0]

    @staticmethod
    @njit
    def calculate_implied_volatility(option_price, S, K, r, T, q=0.0, option_type='calls', steps=100, max_iterations=100, tolerance=1e-8):
        """
        Calculate the implied volatility for a given American option price using the Cox-Ross-Rubinstein model.

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
            price = CoxRossRubinstein.price(S, K, T, r, mid_vol, q, option_type, steps)

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
    def calculate_delta(S, K, T, r, sigma, q=0.0, steps=100):
        """
        Calculate the delta of an option using the CRR binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float, optional): Continuous dividend yield.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The delta of the option.
        """
        dt = T / steps
        u = exp(sigma * sqrt(dt))
        d = 1 / u

        price_up = CoxRossRubinstein.price(S * u, K, T, r, sigma, q, option_type='calls', steps=steps)
        price_down = CoxRossRubinstein.price(S * d, K, T, r, sigma, q, option_type='calls', steps=steps)

        return (price_up - price_down) / (S * (u - d))

    @staticmethod
    @njit
    def calculate_gamma(S, K, T, r, sigma, q=0.0, steps=100):
        """
        Calculate the gamma of an option using the CRR binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float, optional): Continuous dividend yield.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The gamma of the option.
        """
        dt = T / steps
        u = exp(sigma * sqrt(dt))
        d = 1 / u

        price_up = CoxRossRubinstein.price(S * u, K, T, r, sigma, q, option_type='calls', steps=steps)
        price_down = CoxRossRubinstein.price(S * d, K, T, r, sigma, q, option_type='calls', steps=steps)
        price = CoxRossRubinstein.price(S, K, T, r, sigma, q, option_type='calls', steps=steps)

        return (price_up - 2 * price + price_down) / (S ** 2 * (u - d) ** 2)

    @staticmethod
    @njit
    def calculate_vega(S, K, T, r, sigma, q=0.0, steps=100):
        """
        Calculate the vega of an option using the CRR binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float, optional): Continuous dividend yield.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The vega of the option.
        """
        epsilon = 1e-5
        price_up = CoxRossRubinstein.price(S, K, T, r, sigma + epsilon, q, option_type='calls', steps=steps)
        price_down = CoxRossRubinstein.price(S, K, T, r, sigma - epsilon, q, option_type='calls', steps=steps)
        
        return (price_up - price_down) / (2 * epsilon)

    @staticmethod
    @njit
    def calculate_theta(S, K, T, r, sigma, q=0.0, option_type='calls', steps=100):
        """
        Calculate the theta of an option using the CRR binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float, optional): Continuous dividend yield.
            option_type (str, optional): 'calls' or 'puts'.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The theta of the option.
        """
        epsilon = 1e-5
        price = CoxRossRubinstein.price(S, K, T, r, sigma, q, option_type, steps)
        price_epsilon = CoxRossRubinstein.price(S, K, T - epsilon, r, sigma, q, option_type, steps)
        
        return (price_epsilon - price) / epsilon

    @staticmethod
    @njit
    def calculate_rho(S, K, T, r, sigma, q=0.0, option_type='calls', steps=100):
        """
        Calculate the rho of an option using the CRR binomial model.
        
        Parameters:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float, optional): Continuous dividend yield.
            option_type (str, optional): 'calls' or 'puts'.
            steps (int, optional): Number of steps in the binomial tree. Defaults to 100.
        
        Returns:
            float: The rho of the option.
        """
        epsilon = 1e-5
        price_up = CoxRossRubinstein.price(S, K, T, r + epsilon, sigma, q, option_type, steps)
        price_down = CoxRossRubinstein.price(S, K, T, r - epsilon, sigma, q, option_type, steps)
        
        return (price_up - price_down) / (2 * epsilon)
