# OptionsPricerLib

**OptionsPricerLib** is a Python library for pricing financial options using various european and american models. The library provides options pricing, implied volatility calculation, and the Greeks for options, covering models such as Barone-Adesi Whaley, Black-Scholes, Leisen-Reimer, Jarrow-Rudd, and Cox-Ross-Rubinstein.

You can find the library on [PyPI](https://pypi.org/project/OptionsPricerLib/) and on [GitHub](https://github.com/hedge0/OptionsPricerLib)

## Installation

```bash
pip install OptionsPricerLib
```

Or, install directly from the GitHub repository:

```bash
pip install git+https://github.com/hedge0/OptionsPricerLib.git
```

## Usage

After installation, you can import and use any of the models. Here's a quick example:

```python
from OptionsPricerLib import BaroneAdesiWhaley, BlackScholes, CoxRossRubinstein, LeisenReimer, JarrowRudd

# Define parameters
S = 100                 # Current stock price
K = 100                 # Strike price
T = 1                   # Time to maturity (in years)
r = 0.05                # Risk-free interest rate
q = 0.01                # Dividend yield
sigma = 0.2             # Volatility
option_type = 'calls'   # Option type ('calls' or 'puts')

# Barone-Adesi Whaley
price = BaroneAdesiWhaley.price(sigma, S, K, T, r, q, option_type)
delta = BaroneAdesiWhaley.calculate_delta(sigma, S, K, T, r, q, option_type)
gamma = BaroneAdesiWhaley.calculate_gamma(sigma, S, K, T, r, q, option_type)
vega = BaroneAdesiWhaley.calculate_vega(sigma, S, K, T, r, q, option_type)
theta = BaroneAdesiWhaley.calculate_theta(sigma, S, K, T, r, q, option_type)
rho = BaroneAdesiWhaley.calculate_rho(sigma, S, K, T, r, q, option_type)
sigma = BaroneAdesiWhaley.calculate_implied_volatility(price, S, K, T, r, q, option_type)
print(f"Barone-Adesi Whaley {option_type}: Price={price:.2f}, Delta={delta:.4f}, Gamma={gamma:.4f}, Vega={vega:.4f}, Theta={theta:.4f}, Rho={rho:.4f}, Sigma={sigma:.4f}")

# Black-Scholes
price = BlackScholes.price(sigma, S, K, T, r, q, option_type)
delta = BlackScholes.calculate_delta(sigma, S, K, T, r, q, option_type)
gamma = BlackScholes.calculate_gamma(sigma, S, K, T, r, q, option_type)
vega = BlackScholes.calculate_vega(sigma, S, K, T, r, q, option_type)
theta = BlackScholes.calculate_theta(sigma, S, K, T, r, q, option_type)
rho = BlackScholes.calculate_rho(sigma, S, K, T, r, q, option_type)
sigma = BlackScholes.calculate_implied_volatility(price, S, K, T, r, q, option_type)
print(f"Black-Scholes {option_type}: Price={price:.2f}, Delta={delta:.4f}, Gamma={gamma:.4f}, Vega={vega:.4f}, Theta={theta:.4f}, Rho={rho:.4f}, Sigma={sigma:.4f}")

# Leisen-Reimer
price = LeisenReimer.price(sigma, S, K, T, r, q, option_type, steps=100)
delta = LeisenReimer.calculate_delta(sigma, S, K, T, r, q, option_type, steps=100)
gamma = LeisenReimer.calculate_gamma(sigma, S, K, T, r, q, option_type, steps=100)
vega = LeisenReimer.calculate_vega(sigma, S, K, T, r, q, option_type, steps=100)
theta = LeisenReimer.calculate_theta(sigma, S, K, T, r, q, option_type, steps=100)
rho = LeisenReimer.calculate_rho(sigma, S, K, T, r, q, option_type, steps=100)
sigma = LeisenReimer.calculate_implied_volatility(price, S, K, T, r, q, option_type, steps=100)
print(f"Leisen-Reimer {option_type}: Price={price:.2f}, Delta={delta:.4f}, Gamma={gamma:.4f}, Vega={vega:.4f}, Theta={theta:.4f}, Rho={rho:.4f}, Sigma={sigma:.4f}")

# Jarrow-Rudd
price = JarrowRudd.price(sigma, S, K, T, r, q, option_type, steps=100)
delta = JarrowRudd.calculate_delta(sigma, S, K, T, r, q, option_type, steps=100)
gamma = JarrowRudd.calculate_gamma(sigma, S, K, T, r, q, option_type, steps=100)
vega = JarrowRudd.calculate_vega(sigma, S, K, T, r, q, option_type, steps=100)
theta = JarrowRudd.calculate_theta(sigma, S, K, T, r, q, option_type, steps=100)
rho = JarrowRudd.calculate_rho(sigma, S, K, T, r, q, option_type, steps=100)
sigma = JarrowRudd.calculate_implied_volatility(price, S, K, T, r, q, option_type, steps=100)
print(f"Jarrow-Rudd {option_type}: Price={price:.2f}, Delta={delta:.4f}, Gamma={gamma:.4f}, Vega={vega:.4f}, Theta={theta:.4f}, Rho={rho:.4f}, Sigma={sigma:.4f}")

# Cox-Ross-Rubinstein
price = CoxRossRubinstein.price(sigma, S, K, T, r, q, option_type, steps=100)
delta = CoxRossRubinstein.calculate_delta(sigma, S, K, T, r, q, option_type, steps=100)
gamma = CoxRossRubinstein.calculate_gamma(sigma, S, K, T, r, q, option_type, steps=100)
vega = CoxRossRubinstein.calculate_vega(sigma, S, K, T, r, q, option_type, steps=100)
theta = CoxRossRubinstein.calculate_theta(sigma, S, K, T, r, q, option_type, steps=100)
rho = CoxRossRubinstein.calculate_rho(sigma, S, K, T, r, q, option_type, steps=100)
sigma = CoxRossRubinstein.calculate_implied_volatility(price, S, K, T, r, q, option_type, steps=100)
print(f"Cox-Ross-Rubinstein {option_type}: Price={price:.2f}, Delta={delta:.4f}, Gamma={gamma:.4f}, Vega={vega:.4f}, Theta={theta:.4f}, Rho={rho:.4f}, Sigma={sigma:.4f}")
```

## Available Models

1. **Barone-Adesi Whaley**: American, approximation model.
2. **Black-Scholes**: European, analytical model.
3. **Leisen-Reimer**: American, binomial model.
4. **Jarrow-Rudd**: American, binomial model.
5. **Cox-Ross-Rubinstein**: American, binomial model.

## Running Tests

The package includes unit tests in the `tests/` folder. You can run them using `unittest`:

```bash
python -m unittest discover -s tests -p "test_options_models.py"
```

This command will execute the test suite and verify the functionality of each pricing model. If all tests pass, it confirms that each model in `OptionsPricerLib` is performing as expected.

## Contributing

Contributions to **OptionsPricerLib** are welcome! If you find a bug or have suggestions for improvements, please open an issue or submit a pull request. Make sure to follow these guidelines:

1. Fork the repository and clone it locally.
2. Create a new branch for your feature or fix.
3. Add your changes and include tests for any new functionality.
4. Run the test suite to ensure all tests pass.
5. Submit a pull request describing your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or feedback, feel free to reach out to the author via GitHub: [hedge0](https://github.com/hedge0).
