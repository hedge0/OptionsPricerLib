# OptionsPricerLib

**OptionsPricerLib** is a Python library for pricing financial options using various european and american models. The library provides options pricing, implied volatility calculation, and the Greeks for options, covering models such as Barone-Adesi Whaley, Black-Scholes, Leisen-Reimer, Jarrow-Rudd, and Cox-Ross-Rubinstein.

## Installation

```bash
pip install OptionsPricerLib
```

Or, install directly from the GitHub repository:

```bash
pip install git+https://github.com/hedge0/OptionsPricerLib.git
```

## Usage

After installation, you can import and use any of the models. Here’s a quick example using the **Barone-Adesi Whaley** model.

```python
from options_models.barone_adesi_whaley import BaroneAdesiWhaley

# Define parameters
S = 100        # Current stock price
K = 100        # Strike price
T = 1          # Time to maturity (in years)
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility
q = 0.01       # Dividend yield
option_type = 'calls'  # Option type ('calls' or 'puts')

# Calculate option price
price = BaroneAdesiWhaley.price(sigma, S, K, T, r, q, option_type)
print(f"Option Price (Barone-Adesi Whaley, {option_type}): {price:.2f}")

# Calculate Greeks
delta = BaroneAdesiWhaley.calculate_delta(sigma, S, K, T, r, q, option_type)
gamma = BaroneAdesiWhaley.calculate_gamma(sigma, S, K, T, r, q, option_type)
vega = BaroneAdesiWhaley.calculate_vega(sigma, S, K, T, r, q, option_type)
theta = BaroneAdesiWhaley.calculate_theta(sigma, S, K, T, r, q, option_type)
rho = BaroneAdesiWhaley.calculate_rho(sigma, S, K, T, r, q, option_type)

print(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: {vega:.4f}, Theta: {theta:.4f}, Rho: {rho:.4f}")
```

## Available Models

1. **Barone-Adesi Whaley**: American, approximation model.
2. **Black-Scholes**: European.
3. **Leisen-Reimer**: American, binomial model.
4. **Jarrow-Rudd**: European, binomial model.
5. **Cox-Ross-Rubinstein (CRR)**: American, binomial model.

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
