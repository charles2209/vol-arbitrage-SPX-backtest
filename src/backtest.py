import pandas as pd

def simulate_straddle_pnl(spot: pd.Series, iv: pd.Series,
                          r=0.02, q=0.015, tenor_days=30) -> pd.Series:
    """
    Simule le PnL d'un straddle ATM 30j (mark-to-market, pricing BS jour par jour).
    """
    # TODO
    return pd.Series(index=iv.index, dtype=float)
