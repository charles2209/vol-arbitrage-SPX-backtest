import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

EXCEL_PATH = "data/CALL_PUT_SPX.xlsx"  # <-- ton fichier

def load_bloomberg_surface(path):
    df = pd.read_excel(path, header=None, engine="openpyxl")
    rows = []
    current_exp = None
    current_side = None

    for _, row in df.iterrows():
        first = str(row.iloc[0]).strip()

        # Ligne d'échéance (ex: "21-Nov-25 ...")
        if re.match(r"\d{1,2}-[A-Za-z]{3}-\d{2}", first):
            current_exp = first.split()[0]
            continue

        # Section Calls / Puts
        if "Calls" in first:
            current_side = "C"
            continue
        if "Put" in first:
            current_side = "P"
            continue

        # Lignes vides
        if pd.isna(row.iloc[0]):
            continue

        # Lignes de strikes (numériques)
        try:
            strike = float(row.iloc[0])
        except Exception:
            continue

        # Schéma Bloomberg observé: Strike, Ticker, Bid, Ask, Dern, VIM, Vol?
        strike, ticker, bid, ask, last, vim = row.iloc[:6]
        rows.append({
            "Expiration": current_exp,
            "Type": current_side,
            "Strike": float(strike),
            "Bid": pd.to_numeric(bid, errors="coerce"),
            "Ask": pd.to_numeric(ask, errors="coerce"),
            "Last": pd.to_numeric(last, errors="coerce"),
            "VIM": pd.to_numeric(vim, errors="coerce")  # en %
        })

    out = pd.DataFrame(rows).dropna(subset=["VIM"])
    out["Expiration"] = pd.to_datetime(out["Expiration"])
    return out

def estimate_spot_per_expiration(df):
    """
    Estime S par échéance comme le milieu des strikes observés (simple et robuste si S non fourni).
    """
    s_map = {}
    for exp, grp in df.groupby("Expiration"):
        k_min = grp["Strike"].min()
        k_max = grp["Strike"].max()
        s_est = 0.5 * (k_min + k_max)
        s_map[exp] = s_est
    return s_map

def plot_smiles_moneyness(df, m_range=(0.97, 1.03)):
    # Estime S par échéance
    s_map = estimate_spot_per_expiration(df)

    # Ajoute moneyness = K / S(expiration)
    df = df.copy()
    df["S_est"] = df["Expiration"].map(s_map)
    df["moneyness"] = df["Strike"] / df["S_est"]

    # Filtre zone proche d’ATM (lisible)
    lo, hi = m_range
    df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)].dropna(subset=["VIM"])

    # Tri par maturité
    df = df.sort_values(["Expiration", "moneyness"])

    # Plot: IV (%) vs moneyness (une courbe par échéance)
    plt.figure(figsize=(8,6))
    for exp, grp in df.groupby("Expiration"):
        # petit lissage par interpolation pour des courbes propres
        xs = np.linspace(grp["moneyness"].min(), grp["moneyness"].max(), 25)
        ys = np.interp(xs, grp["moneyness"], grp["VIM"])
        plt.plot(xs, ys, marker="o", markersize=2, label=exp.date())

    plt.title("IV smile (VIM %) — Normalisé en moneyness K/S, zone ATM")
    plt.xlabel("Moneyness K/S (≈ 1.00 à l’ATM)")
    plt.ylabel("Implied Vol (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

def main():
    df = load_bloomberg_surface(EXCEL_PATH)
    plot_smiles_moneyness(df, m_range=(0.97, 1.03))  # tu peux élargir à (0.95,1.05)

if __name__ == "__main__":
    main()
