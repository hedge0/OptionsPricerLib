import unittest
from OptionsPricerLib.barone_adesi_whaley import BaroneAdesiWhaley
from OptionsPricerLib.black_scholes import BlackScholes
from OptionsPricerLib.leisen_reimer import LeisenReimer
from OptionsPricerLib.jarrow_rudd import JarrowRudd
from OptionsPricerLib.cox_ross_rubinstein import CoxRossRubinstein

class TestOptionsModels(unittest.TestCase):

    def setUp(self):
        self.S = 100
        self.K = 100
        self.T = 1
        self.r = 0.05
        self.sigma = 0.2
        self.q = 0.01
        self.option_types = ['calls', 'puts']
        self.steps = 100

    def check_model(self, model, expected_results):
        for option_type, results in zip(self.option_types, expected_results):
            # Price
            price = model.price(self.sigma, self.S, self.K, self.T, self.r, self.q, option_type)
            self.assertAlmostEqual(price, results['price'], places=2)
            
            # Implied Volatility
            iv = model.calculate_implied_volatility(price, self.S, self.K, self.T, self.r, self.q, option_type)
            self.assertAlmostEqual(iv, results['iv'], places=2)
            
            # Greeks
            delta = model.calculate_delta(self.sigma, self.S, self.K, self.T, self.r, self.q, option_type)
            self.assertAlmostEqual(delta, results['delta'], places=4)
            gamma = model.calculate_gamma(self.sigma, self.S, self.K, self.T, self.r, self.q, option_type)
            self.assertAlmostEqual(gamma, results['gamma'], places=4)
            vega = model.calculate_vega(self.sigma, self.S, self.K, self.T, self.r, self.q, option_type)
            self.assertAlmostEqual(vega, results['vega'], places=4)
            theta = model.calculate_theta(self.sigma, self.S, self.K, self.T, self.r, self.q, option_type)
            self.assertAlmostEqual(theta, results['theta'], places=4)
            rho = model.calculate_rho(self.sigma, self.S, self.K, self.T, self.r, self.q, option_type)
            self.assertAlmostEqual(rho, results['rho'], places=4)

    def test_barone_adesi_whaley(self):
        expected_results = [
            {  # Calls
                'price': 9.83, 'iv': 0.2, 'delta': 0.6118, 'gamma': 0.0189, 
                'vega': 37.7593, 'theta': -5.7317, 'rho': 51.3500
            },
            {  # Puts
                'price': 5.94, 'iv': 0.2, 'delta': -0.3783, 'gamma': 0.0189, 
                'vega': 37.7593, 'theta': -1.9656, 'rho': -43.7729
            }
        ]
        self.check_model(BaroneAdesiWhaley, expected_results)

    def test_black_scholes(self):
        expected_results = [
            {  # Calls
                'price': 9.83, 'iv': 0.2, 'delta': 0.6118, 'gamma': 0.0189, 
                'vega': 37.7593, 'theta': -5.7317, 'rho': 51.3500
            },
            {  # Puts
                'price': 5.94, 'iv': 0.2, 'delta': -0.3783, 'gamma': 0.0189, 
                'vega': 37.7593, 'theta': -1.9656, 'rho': -43.7729
            }
        ]
        self.check_model(BlackScholes, expected_results)

    def test_leisen_reimer(self):
        expected_results = [
            {  # Calls
                'price': 9.80, 'iv': 0.2, 'delta': 0.6112, 'gamma': 0.0189, 
                'vega': 37.6868, 'theta': -5.7191, 'rho': 51.2122
            },
            {  # Puts
                'price': 6.36, 'iv': 0.2, 'delta': -0.3515, 'gamma': 0.0189, 
                'vega': 37.7298, 'theta': -2.4543, 'rho': -32.0665
            }
        ]
        self.check_model(LeisenReimer, expected_results)

    def test_jarrow_rudd(self):
        expected_results = [
            {  # Calls
                'price': 9.84, 'iv': 0.2, 'delta': 0.6123, 'gamma': 0.0189, 
                'vega': 37.8446, 'theta': -5.7399, 'rho': 51.3469
            },
            {  # Puts
                'price': 6.38, 'iv': 0.2, 'delta': -0.4166, 'gamma': 0.0189, 
                'vega': 37.8534, 'theta': -2.4607, 'rho': -32.1845
            }
        ]
        self.check_model(JarrowRudd, expected_results)

    def test_cox_ross_rubinstein(self):
        expected_results = [
            {  # Calls
                'price': 9.81, 'iv': 0.2, 'delta': 0.6119, 'gamma': 0.0189, 
                'vega': 37.6649, 'theta': -5.7221, 'rho': 51.3418
            },
            {  # Puts
                'price': 6.36, 'iv': 0.2, 'delta': -0.4166, 'gamma': 0.0189, 
                'vega': 37.7392, 'theta': -2.4531, 'rho': -32.1017
            }
        ]
        self.check_model(CoxRossRubinstein, expected_results)

if __name__ == "__main__":
    unittest.main()
