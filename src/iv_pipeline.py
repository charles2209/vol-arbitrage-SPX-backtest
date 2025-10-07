# src/iv_pipeline.py
import math
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# 1) Loading & cleaning
# ---------------------------

def _mid_price(bid, ask, last):
    bid = pd.to_numeric(bid, errors="coerce")
    ask = pd.to_numeric(ask, errors="coerce")
    last = pd.to_numeric(last, errors="coerce")
    mid = np.where((bid > 0) & (ask > 0), 0.5 * (bid + ask), last)
    return pd.to_numeric(mid, errors="coerce")

def load_calls_puts_xlsx(
    path: str,
    maturity_days: int,
    calls_cols: Optional[Tuple[int,int,int,int,int]] = None,
    puts_cols: Optional[Tuple[int,int,int,int,int]] = None,
    header_row: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lit un fichier Excel Bloomberg 'Calls | Puts' sur UNE enveloppe de maturité.
    Hypothèse par défaut (comme ton screenshot): 2 tableaux côte à côte:
      Calls: [Strike, Ticker, Bid, Ask, Dern, VIM]  -> colonnes 0..5
      Puts : [Strike, Ticker, Bid, Ask, Dern, VIM]  -> colonnes 6..11

    Si ton fichier n'a pas exactement cette structure, passe calls_cols/puts_cols
    comme tuples d'index (strike, bid, ask, last, vim).
    """
    df_full = pd.read_excel(path, header=header_row)

    # Détection simple si pas d'arguments: 1er bloc = calls, 2e bloc = puts
    if calls_cols is None:
        calls_cols = (0, 2, 3, 4, 5)  # strike,bid,ask,dern,vim dans les 6 premières colonnes
    if puts_cols is None:
        puts_cols = (6, 8, 9, 10, 11) # strike,bid,ask,dern,vim dans les 6 suivantes

    # Sélection & renommage
    c_strike, c_bid, c_ask, c_last, c_vim = calls_cols
    p_strike, p_bid, p_ask, p_last, p_vim = puts_cols

    calls_raw = df_full.iloc[:, [c_strike, c_bid, c_ask, c_last, c_vim]].copy()
    puts_raw  = df_full.iloc[:, [p_strike, p_bid, p_ask, p_last, p_vim]].copy()

    calls_raw.columns = ["strike", "bid", "ask", "last", "vim"]
    puts_raw.columns  = ["strike", "bid", "ask", "last", "vim"]

    # Nettoyage de base
    for df in (calls_raw, puts_raw):
        df["mid"] = _mid_price(df["bid"], df["ask"], df["last"])
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["vim"] = pd.to_numeric(df["vim"], errors="coerce")
        # Convertit IV en décimal si donnée en %
        if df["vim"].max(skipna=True) > 1.5:
            df["vim"] = df["vim"] / 100.0
        df.dropna(subset=["strike", "mid"], inplace=True)

    calls_raw["type"] = "call"
    puts_raw["type"]  = "put"
    calls_raw["maturity_days"] = maturity_days
    puts_raw["maturity_days"]  = maturity_days

    return calls_raw[["maturity_days","type","strike","bid","ask","last","mid","vim"]], \
           puts_raw[ ["maturity_days","type","strike","bid","ask","last","mid","vim"]]


# ---------------------------
# 2) Put-call parity (F & D)
# ---------------------------

def estimate_forward_discount_from_pairs(pairs: pd.DataFrame, T_years: float) -> Tuple[float, float]:
    """
    Parité: C - P = D * (F - K) = A - D*K, avec A = D*F.
    On ajuste y = a + b*K, b ≈ -D, a ≈ D*F  => F = a/(-b), D = -b.
    """
    d = pairs.dropna(subset=["mid_call", "mid_put", "strike"]).copy()
    if len(d) < 3:
        return math.nan, math.nan

    y = (d["mid_call"] - d["mid_put"]).to_numpy()
    K = d["strike"].to_numpy()
    A = np.vstack([np.ones_like(K), K]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    D = -b
    if D <= 0 or not np.isfinite(D):
        return math.nan, math.nan
    F = a / D
    return float(F), float(D)


# ---------------------------
# 3) Black-76 & IV solver
# ---------------------------

def _n_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black76_undiscounted(F: float, K: float, T: float, sigma: float, option: str) -> float:
    if T <= 0 or F <= 0 or K <= 0:
        return max((F - K) if option == "call" else (K - F), 0.0)
    if sigma <= 0:
        # limite sigma->0
        return max((F - K) if option == "call" else (K - F), 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if option == "call":
        return F * _n_cdf(d1) - K * _n_cdf(d2)
    else:
        return K * _n_cdf(-d2) - F * _n_cdf(-d1)

def implied_vol_black76(
    price: float, F: float, K: float, T: float, D: float, option: str,
    tol: float = 1e-7, sigma_lo: float = 1e-6, sigma_hi: float = 5.0, max_iter: int = 100
) -> float:
    if not all(np.isfinite([price, F, K, T, D])) or price <= 0 or D <= 0:
        return math.nan
    target = price / D
    lo, hi = sigma_lo, sigma_hi

    def f(sig: float) -> float:
        return black76_undiscounted(F, K, T, sig, option) - target

    flo, fhi = f(lo), f(hi)
    tries = 0
    while flo * fhi > 0 and tries < 6:
        hi *= 2.0
        fhi = f(hi)
        tries += 1
    if flo * fhi > 0:
        return math.nan

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < 1e-7:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return mid


# ---------------------------
# 4) Pairing & IV recompute
# ---------------------------

def build_pairs(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    """
    Regroupe par strike (au cas où il y ait des doublons) et merge.
    """
    c = calls.groupby("strike", as_index=False).agg(mid_call=("mid","median"), vim_call=("vim","median"))
    p = puts.groupby("strike", as_index=False).agg(mid_put =("mid","median"), vim_put =("vim","median"))
    pairs = pd.merge(c, p, on="strike", how="inner").sort_values("strike")
    return pairs

def recompute_ivs(pairs: pd.DataFrame, T_years: float) -> pd.DataFrame:
    F, D = estimate_forward_discount_from_pairs(pairs, T_years)
    out = pairs.copy()
    out["F_est"] = F
    out["D_est"] = D

    ivc, ivp = [], []
    for K, mc, mp in zip(out["strike"], out["mid_call"], out["mid_put"]):
        ivc.append(implied_vol_black76(mc, F, K, T_years, D, "call"))
        ivp.append(implied_vol_black76(mp, F, K, T_years, D, "put"))
    out["iv_call_recomputed"] = ivc
    out["iv_put_recomputed"]  = ivp
    return out


# ---------------------------
# 5) Plotting
# ---------------------------

def plot_smile(df_pairs: pd.DataFrame, label: str, use_calls: bool = True):
    y = "iv_call_recomputed" if use_calls else "iv_put_recomputed"
    d = df_pairs.dropna(subset=[y, "strike"]).copy()
    plt.figure()
    plt.plot(d["strike"], d[y])
    plt.xlabel("Strike")
    plt.ylabel("Implied Vol (decimal)")
    plt.title(f"IV Smile — {label}")
    plt.show()
