from options_models.barone_adesi_whaley import BaroneAdesiWhaley
from options_models.black_scholes import BlackScholes
from options_models.leisen_reimer import LeisenReimer
from options_models.jarrow_rudd import JarrowRudd
from options_models.cox_ross_rubinstein import CoxRossRubinstein

def main():
    # Define option parameters
    S = 100      # Current stock price
    K = 100      # Strike price
    T = 1        # Time to maturity in years
    r = 0.05     # Risk-free interest rate
    sigma = 0.2  # Implied volatility
    q = 0.01     # Dividend yield

    # Loop through both call and put option types
    for option_type in ['calls', 'puts']:
        print(f"\nCalculating prices, implied volatilities, and Greeks for {option_type}:")

        # Calculate option price and Greeks using Barone-Adesi Whaley model
        baw_price = BaroneAdesiWhaley.price(sigma, S, K, T, r, q, option_type)
        print(f"The price of the {option_type} option using Barone-Adesi Whaley model is: {baw_price:.2f}")
        baw_iv = BaroneAdesiWhaley.calculate_implied_volatility(baw_price, S, K, T, r, q, option_type)
        print(f"The implied volatility using Barone-Adesi Whaley model is: {baw_iv:.2%}")
        baw_delta = BaroneAdesiWhaley.calculate_delta(sigma, S, K, T, r, q, option_type)
        print(f"The delta using Barone-Adesi Whaley model is: {baw_delta:.4f}")
        baw_gamma = BaroneAdesiWhaley.calculate_gamma(sigma, S, K, T, r, q, option_type)
        print(f"The gamma using Barone-Adesi Whaley model is: {baw_gamma:.4f}")
        baw_vega = BaroneAdesiWhaley.calculate_vega(sigma, S, K, T, r, q, option_type)
        print(f"The vega using Barone-Adesi Whaley model is: {baw_vega:.4f}")
        baw_theta = BaroneAdesiWhaley.calculate_theta(sigma, S, K, T, r, q, option_type)
        print(f"The theta using Barone-Adesi Whaley model is: {baw_theta:.4f}")
        baw_rho = BaroneAdesiWhaley.calculate_rho(sigma, S, K, T, r, q, option_type)
        print(f"The rho using Barone-Adesi Whaley model is: {baw_rho:.4f}")
        print()  # Space for readability

        # Calculate option price and Greeks using Black-Scholes model
        bs_price = BlackScholes.price(sigma, S, K, T, r, q, option_type)
        print(f"The price of the {option_type} option using Black-Scholes model is: {bs_price:.2f}")
        bs_iv = BlackScholes.calculate_implied_volatility(bs_price, S, K, T, r, q, option_type)
        print(f"The implied volatility using Black-Scholes model is: {bs_iv:.2%}")
        bs_delta = BlackScholes.calculate_delta(sigma, S, K, T, r, q, option_type)
        print(f"The delta using Black-Scholes model is: {bs_delta:.4f}")
        bs_gamma = BlackScholes.calculate_gamma(sigma, S, K, T, r, q, option_type)
        print(f"The gamma using Black-Scholes model is: {bs_gamma:.4f}")
        bs_vega = BlackScholes.calculate_vega(sigma, S, K, T, r, q, option_type)
        print(f"The vega using Black-Scholes model is: {bs_vega:.4f}")
        bs_theta = BlackScholes.calculate_theta(sigma, S, K, T, r, q, option_type)
        print(f"The theta using Black-Scholes model is: {bs_theta:.4f}")
        bs_rho = BlackScholes.calculate_rho(sigma, S, K, T, r, q, option_type)
        print(f"The rho using Black-Scholes model is: {bs_rho:.4f}")
        print()

        # Calculate option price and Greeks using Leisen-Reimer model
        lr_price = LeisenReimer.price(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The price of the {option_type} option using Leisen-Reimer model is: {lr_price:.2f}")
        lr_iv = LeisenReimer.calculate_implied_volatility(lr_price, S, K, T, r, q, option_type, steps=100)
        print(f"The implied volatility using Leisen-Reimer model is: {lr_iv:.2%}")
        lr_delta = LeisenReimer.calculate_delta(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The delta using Leisen-Reimer model is: {lr_delta:.4f}")
        lr_gamma = LeisenReimer.calculate_gamma(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The gamma using Leisen-Reimer model is: {lr_gamma:.4f}")
        lr_vega = LeisenReimer.calculate_vega(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The vega using Leisen-Reimer model is: {lr_vega:.4f}")
        lr_theta = LeisenReimer.calculate_theta(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The theta using Leisen-Reimer model is: {lr_theta:.4f}")
        lr_rho = LeisenReimer.calculate_rho(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The rho using Leisen-Reimer model is: {lr_rho:.4f}")
        print()

        # Calculate option price and Greeks using Jarrow-Rudd model
        jr_price = JarrowRudd.price(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The price of the {option_type} option using Jarrow-Rudd model is: {jr_price:.2f}")
        jr_iv = JarrowRudd.calculate_implied_volatility(jr_price, S, K, T, r, q, option_type, steps=100)
        print(f"The implied volatility using Jarrow-Rudd model is: {jr_iv:.2%}")
        jr_delta = JarrowRudd.calculate_delta(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The delta using Jarrow-Rudd model is: {jr_delta:.4f}")
        jr_gamma = JarrowRudd.calculate_gamma(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The gamma using Jarrow-Rudd model is: {jr_gamma:.4f}")
        jr_vega = JarrowRudd.calculate_vega(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The vega using Jarrow-Rudd model is: {jr_vega:.4f}")
        jr_theta = JarrowRudd.calculate_theta(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The theta using Jarrow-Rudd model is: {jr_theta:.4f}")
        jr_rho = JarrowRudd.calculate_rho(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The rho using Jarrow-Rudd model is: {jr_rho:.4f}")
        print()

        # Calculate option price and Greeks using Cox-Ross-Rubinstein model
        crr_price = CoxRossRubinstein.price(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The price of the {option_type} option using Cox-Ross-Rubinstein model is: {crr_price:.2f}")
        crr_iv = CoxRossRubinstein.calculate_implied_volatility(crr_price, S, K, T, r, q, option_type, steps=100)
        print(f"The implied volatility using Cox-Ross-Rubinstein model is: {crr_iv:.2%}")
        crr_delta = CoxRossRubinstein.calculate_delta(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The delta using Cox-Ross-Rubinstein model is: {crr_delta:.4f}")
        crr_gamma = CoxRossRubinstein.calculate_gamma(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The gamma using Cox-Ross-Rubinstein model is: {crr_gamma:.4f}")
        crr_vega = CoxRossRubinstein.calculate_vega(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The vega using Cox-Ross-Rubinstein model is: {crr_vega:.4f}")
        crr_theta = CoxRossRubinstein.calculate_theta(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The theta using Cox-Ross-Rubinstein model is: {crr_theta:.4f}")
        crr_rho = CoxRossRubinstein.calculate_rho(sigma, S, K, T, r, q, option_type, steps=100)
        print(f"The rho using Cox-Ross-Rubinstein model is: {crr_rho:.4f}")
        print()

if __name__ == "__main__":
    main()
