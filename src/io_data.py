import pandas as pd

def load_options_csv(path: str) -> pd.DataFrame:
    """Charge un CSV options exporté depuis Bloomberg OMON."""
    df = pd.read_csv(path)
    return df

def load_spot_csv(path: str, date_col: str | None = None, price_col: str | None = None) -> pd.Series:
    """
    Charge un CSV de prix spot (ex: SPX historique).
    - date_col: nom de la colonne date (ex: 'Date'), sinon tentative auto
    - price_col: nom de la colonne prix (ex: 'PX_LAST'), sinon tentative auto
    Retourne une Series indexée par date.
    """
    df = pd.read_csv(path)
    if date_col is None:
        # essaye des noms courants
        for c in ["Date", "date", "DATES", "DATES_UTC"]:
            if c in df.columns: 
                date_col = c; break
    if price_col is None:
        for c in ["PX_LAST", "Close", "close", "Adj Close", "PRICE"]:
            if c in df.columns: 
                price_col = c; break
    if date_col is None or price_col is None:
        raise ValueError("Impossible d’identifier la colonne date/prix. Spécifie date_col et price_col.")
    df[date_col] = pd.to_datetime(df[date_col])
    s = df.set_index(date_col)[price_col].astype(float).sort_index()
    s.name = "SPOT"
    return s
