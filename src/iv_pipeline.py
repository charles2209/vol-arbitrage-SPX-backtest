import pandas as pd

def compute_mid(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute MID = (BID+ASK)/2 si BID/ASK existent."""
    # TODO
    return df

def add_time_to_maturity(df: pd.DataFrame, today=None) -> pd.DataFrame:
    """Ajoute T (années) à partir d'EXPIRATION et de 'today'."""
    # TODO
    return df

def compute_iv(df: pd.DataFrame, r=0.02, q=0.015) -> pd.DataFrame:
    """Calcule IV_calc pour chaque ligne à partir de MID, S, K, T."""
    # TODO
    return df

def select_atm_30d(df: pd.DataFrame) -> pd.Series | float:
    """Retourne l'IV ATM ~30 jours (snapshot ou Series si multi-dates)."""
    # TODO
    return pd.Series(dtype=float)
