import math

def bs_price(S, K, T, r=0.0, q=0.0, sigma=0.2, option_type="C"):
    """Prix d'un call/put selon Black-Scholes."""
    # TODO: à implémenter ensuite
    pass

def vega(S, K, T, r=0.0, q=0.0, sigma=0.2):
    """Dérivée du prix par rapport à la volatilité (dV/dsigma)."""
    # TODO: à implémenter ensuite
    pass

def implied_vol(price_mkt, S, K, T, r=0.0, q=0.0, option_type="C",
                init_sigma=0.2, tol=1e-6, max_iter=100):
    """Calcule la volatilité implicite via inversion de Black-Scholes."""
    # TODO: à implémenter ensuite (Newton-Raphson + fallback bissection)
    pass