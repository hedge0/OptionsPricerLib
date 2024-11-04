import unittest
from options_models.barone_adesi_whaley import BaroneAdesiWhaley

class TestBaroneAdesiWhaley(unittest.TestCase):

    def setUp(self):
        # Sample parameters for testing
        self.S = 100  # Stock price
        self.K = 100  # Strike price
        self.T = 1    # Time to maturity (1 year)
        self.r = 0.05  # Risk-free rate
        self.sigma = 0.2  # Volatility
        self.q = 0.01  # Dividend yield

    def test_price_call(self):
        price = BaroneAdesiWhaley.price(self.S, self.K, self.T, self.r, self.sigma, self.q, option_type='calls')
        self.assertAlmostEqual(price, 10.45, places=2)  # Replace with expected value

    def test_price_put(self):
        price = BaroneAdesiWhaley.price(self.S, self.K, self.T, self.r, self.sigma, self.q, option_type='puts')
        self.assertAlmostEqual(price, 5.75, places=2)  # Replace with expected value

    def test_delta_call(self):
        delta = BaroneAdesiWhaley.calculate_delta(self.S, self.K, self.T, self.r, self.sigma, self.q, option_type='calls')
        self.assertAlmostEqual(delta, 0.6, places=2)  # Replace with expected value

    def test_gamma(self):
        gamma = BaroneAdesiWhaley.calculate_gamma(self.S, self.K, self.T, self.r, self.sigma, self.q)
        self.assertAlmostEqual(gamma, 0.02, places=2)  # Replace with expected value

    def test_vega(self):
        vega = BaroneAdesiWhaley.calculate_vega(self.S, self.K, self.T, self.r, self.sigma, self.q)
        self.assertAlmostEqual(vega, 12.34, places=2)  # Replace with expected value

    def test_theta_call(self):
        theta = BaroneAdesiWhaley.calculate_theta(self.S, self.K, self.T, self.r, self.sigma, self.q, option_type='calls')
        self.assertAlmostEqual(theta, -0.04, places=2)  # Replace with expected value

    def test_rho_call(self):
        rho = BaroneAdesiWhaley.calculate_rho(self.S, self.K, self.T, self.r, self.sigma, self.q, option_type='calls')
        self.assertAlmostEqual(rho, 20.23, places=2)  # Replace with expected value

    def test_implied_volatility(self):
        option_price = 10  # Example observed option price
        implied_vol = BaroneAdesiWhaley.calculate_implied_volatility(option_price, self.S, self.K, self.r, self.T, self.q, option_type='calls')
        self.assertAlmostEqual(implied_vol, 0.2, places=2)  # Replace with expected value

if __name__ == "__main__":
    unittest.main()
