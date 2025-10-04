import pandas as pd

def signal_iv_vs_rv(iv: pd.Series, rv: pd.Series, theta=0.02) -> pd.Series:
    """
    +1 si IV < RV - theta (long vol), -1 si IV > RV + theta (short vol), 0 sinon.
    """
    # TODO
    return pd.Series(index=iv.index, dtype=int)
