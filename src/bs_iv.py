import math

def _N(x): 
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _n(x): 
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def bs_price(S, K, T, r=0.0, q=0.0, sigma=0.2, option_type="C"):

    option_type = option_type.upper()
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "C" else (K - S))

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "C":
        return S * math.exp(-q * T) * _N(d1) - K * math.exp(-r * T) * _N(d2)
    else:
        return K * math.exp(-r * T) * _N(-d2) - S * math.exp(-q * T) * _N(-d1)
    

def vega(S, K, T, r=0.0, q=0.0, sigma=0.2):
    """dPrix/dsigma (non annualisé)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * math.exp(-q * T) * _n(d1) * sqrtT

def implied_vol(price_mkt, S, K, T, r=0.0, q=0.0, option_type="C",
                init_sigma=0.2, tol=1e-6, max_iter=100):
    """
    Inversion de BS : Newton-Raphson + repli bissection si vega faible.
    Retourne Non si échec.
    """
    if price_mkt <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float("non")

    # bornes larges mais raisonnables
    lo, hi = 1e-6, 5.0
    sigma = max(lo, min(init_sigma, hi))

    # Newton-raphson
    for _ in range(max_iter):
        model = bs_price(S, K, T, r, q, sigma, option_type)
        diff = model - price_mkt
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, q, sigma)
        if v < 1e-8:
            break
        sigma -= diff / v
        if sigma <= lo or sigma >= hi:
            break

    # Bissection
    a, b = lo, hi
    fa = bs_price(S, K, T, r, q, a, option_type) - price_mkt
    fb = bs_price(S, K, T, r, q, b, option_type) - price_mkt
    # élargir si nécessaire
    tries = 0
    while fa * fb > 0 and tries < 12:
        b *= 2
        fb = bs_price(S, K, T, r, q, b, option_type) - price_mkt
        tries += 1
        if b > 50:  # garde-fou
            return float("nan")

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = bs_price(S, K, T, r, q, m, option_type) - price_mkt
        if abs(fm) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)
