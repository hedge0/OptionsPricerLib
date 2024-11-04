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
        print(f"\nCalculating prices and implied volatilities for {option_type}:")

        # Calculate option price using Barone-Adesi Whaley model
        baw_price = BaroneAdesiWhaley.price(S, K, T, r, sigma, q, option_type)
        print(f"The price of the {option_type} option using Barone-Adesi Whaley model is: {baw_price:.2f}")
        baw_iv = BaroneAdesiWhaley.calculate_implied_volatility(baw_price, S, K, r, T, q, option_type)
        print(f"The implied volatility using Barone-Adesi Whaley model is: {baw_iv:.2%}")

        # Calculate option price using Black-Scholes model
        bs_price = BlackScholes.price(S, K, T, r, sigma, q, option_type)
        print(f"The price of the {option_type} option using Black-Scholes model is: {bs_price:.2f}")
        bs_iv = BlackScholes.calculate_implied_volatility(bs_price, S, K, r, T, q, option_type)
        print(f"The implied volatility using Black-Scholes model is: {bs_iv:.2%}")

        # Calculate option price using Leisen-Reimer binomial model
        lr_price = LeisenReimer.price(S, K, T, r, sigma, q, option_type, steps=100)
        print(f"The price of the {option_type} option using Leisen-Reimer model is: {lr_price:.2f}")
        lr_iv = LeisenReimer.calculate_implied_volatility(lr_price, S, K, r, T, q, option_type, steps=100)
        print(f"The implied volatility using Leisen-Reimer model is: {lr_iv:.2%}")

        # Calculate option price using Jarrow-Rudd binomial model
        jr_price = JarrowRudd.price(S, K, T, r, sigma, q, option_type, steps=100)
        print(f"The price of the {option_type} option using Jarrow-Rudd model is: {jr_price:.2f}")
        jr_iv = JarrowRudd.calculate_implied_volatility(jr_price, S, K, r, T, q, option_type, steps=100)
        print(f"The implied volatility using Jarrow-Rudd model is: {jr_iv:.2%}")

        # Calculate option price using Cox-Ross-Rubinstein binomial model
        crr_price = CoxRossRubinstein.price(S, K, T, r, sigma, q, option_type, steps=100)
        print(f"The price of the {option_type} option using Cox-Ross-Rubinstein model is: {crr_price:.2f}")
        crr_iv = CoxRossRubinstein.calculate_implied_volatility(crr_price, S, K, r, T, q, option_type, steps=100)
        print(f"The implied volatility using Cox-Ross-Rubinstein model is: {crr_iv:.2%}")

if __name__ == "__main__":
    main()