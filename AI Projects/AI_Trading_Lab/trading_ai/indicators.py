"""
indicators.py - Technical Analysis Indicators
Computes popular technical indicators on OHLCV DataFrames.
Each function takes a DataFrame and returns it with new columns appended.
"""

import numpy as np
import pandas as pd


# ── Registry ────────────────────────────────────────────────────────────────

INDICATOR_REGISTRY: dict[str, dict] = {}


def register(name: str, category: str, params: dict):
    """Decorator that registers an indicator function."""
    def wrapper(fn):
        INDICATOR_REGISTRY[name] = {
            "function": fn,
            "category": category,
            "default_params": params,
            "description": fn.__doc__ or "",
        }
        return fn
    return wrapper


# ── Trend Indicators ────────────────────────────────────────────────────────

@register("SMA", "Trend", {"period": 20})
def sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Simple Moving Average"""
    col = f"SMA_{period}"
    df[col] = df["Close"].rolling(window=period).mean()
    return df


@register("EMA", "Trend", {"period": 20})
def ema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Exponential Moving Average"""
    col = f"EMA_{period}"
    df[col] = df["Close"].ewm(span=period, adjust=False).mean()
    return df


@register("DEMA", "Trend", {"period": 20})
def dema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Double Exponential Moving Average"""
    ema1 = df["Close"].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    df[f"DEMA_{period}"] = 2 * ema1 - ema2
    return df


@register("MACD", "Trend", {"fast": 12, "slow": 26, "signal": 9})
def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence"""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


@register("ADX", "Trend", {"period": 14})
def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index"""
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # Zero out where opposite is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    df[f"ADX_{period}"] = dx.rolling(period).mean()
    df[f"+DI_{period}"] = plus_di
    df[f"-DI_{period}"] = minus_di
    return df


@register("Ichimoku", "Trend", {"tenkan": 9, "kijun": 26, "senkou_b": 52})
def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> pd.DataFrame:
    """Ichimoku Cloud"""
    high, low = df["High"], df["Low"]
    df["Tenkan_Sen"] = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    df["Kijun_Sen"] = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    df["Senkou_A"] = ((df["Tenkan_Sen"] + df["Kijun_Sen"]) / 2).shift(kijun)
    df["Senkou_B"] = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    df["Chikou_Span"] = df["Close"].shift(-kijun)
    return df


@register("Parabolic_SAR", "Trend", {"af_start": 0.02, "af_max": 0.2})
def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    """Parabolic SAR"""
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    sar = np.zeros(n)
    trend = np.ones(n)  # 1 = up, -1 = down
    af = af_start
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if trend[i - 1] == 1:
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep
                af = af_start
                ep = low[i]
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_start, af_max)
        else:
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep
                af = af_start
                ep = high[i]
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_start, af_max)

    df["PSAR"] = sar
    df["PSAR_Trend"] = trend
    return df


# ── Volatility Indicators ──────────────────────────────────────────────────

@register("Bollinger_Bands", "Volatility", {"period": 20, "std_dev": 2.0})
def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands"""
    sma_val = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    df[f"BB_Mid_{period}"] = sma_val
    df[f"BB_Upper_{period}"] = sma_val + std_dev * std
    df[f"BB_Lower_{period}"] = sma_val - std_dev * std
    df[f"BB_Width_{period}"] = (df[f"BB_Upper_{period}"] - df[f"BB_Lower_{period}"]) / sma_val
    return df


@register("ATR", "Volatility", {"period": 14})
def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range"""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df[f"ATR_{period}"] = tr.rolling(period).mean()
    return df


@register("Keltner_Channel", "Volatility", {"period": 20, "atr_mult": 2.0})
def keltner_channel(df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0) -> pd.DataFrame:
    """Keltner Channel"""
    mid = df["Close"].ewm(span=period, adjust=False).mean()
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    df[f"KC_Mid_{period}"] = mid
    df[f"KC_Upper_{period}"] = mid + atr_mult * atr_val
    df[f"KC_Lower_{period}"] = mid - atr_mult * atr_val
    return df


# ── Momentum / Oscillator Indicators ───────────────────────────────────────

@register("RSI", "Momentum", {"period": 14})
def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index"""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


@register("Stochastic", "Momentum", {"k_period": 14, "d_period": 3})
def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D)"""
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    df[f"Stoch_K_{k_period}"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df[f"Stoch_D_{k_period}"] = df[f"Stoch_K_{k_period}"].rolling(d_period).mean()
    return df


@register("Williams_%R", "Momentum", {"period": 14})
def williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Williams %R"""
    high_max = df["High"].rolling(period).max()
    low_min = df["Low"].rolling(period).min()
    df[f"Williams_R_{period}"] = -100 * (high_max - df["Close"]) / (high_max - low_min)
    return df


@register("CCI", "Momentum", {"period": 20})
def cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Commodity Channel Index"""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df[f"CCI_{period}"] = (tp - sma_tp) / (0.015 * mad)
    return df


@register("ROC", "Momentum", {"period": 12})
def roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """Rate of Change"""
    df[f"ROC_{period}"] = df["Close"].pct_change(periods=period) * 100
    return df


@register("MFI", "Momentum", {"period": 14})
def mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Money Flow Index"""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos_mf = mf.where(tp > tp.shift(), 0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(), 0).rolling(period).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    df[f"MFI_{period}"] = 100 - (100 / (1 + mfr))
    return df


# ── Volume Indicators ──────────────────────────────────────────────────────

@register("OBV", "Volume", {})
def obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume"""
    sign = np.sign(df["Close"].diff())
    df["OBV"] = (sign * df["Volume"]).cumsum()
    return df


@register("VWAP", "Volume", {})
def vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-Weighted Average Price (rolling daily reset on daily data)"""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_vtp = (tp * df["Volume"]).cumsum()
    df["VWAP"] = cum_vtp / cum_vol.replace(0, np.nan)
    return df


@register("AD_Line", "Volume", {})
def ad_line(df: pd.DataFrame) -> pd.DataFrame:
    """Accumulation/Distribution Line"""
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    df["AD_Line"] = (clv * df["Volume"]).cumsum()
    return df


# ── Helper Functions ────────────────────────────────────────────────────────

def get_all_indicator_names() -> list[str]:
    return sorted(INDICATOR_REGISTRY.keys())


def get_indicators_by_category() -> dict[str, list[str]]:
    cats: dict[str, list[str]] = {}
    for name, info in INDICATOR_REGISTRY.items():
        cats.setdefault(info["category"], []).append(name)
    return {k: sorted(v) for k, v in cats.items()}


def apply_indicator(df: pd.DataFrame, name: str, **kwargs) -> pd.DataFrame:
    """Apply a single registered indicator by name."""
    if name not in INDICATOR_REGISTRY:
        raise ValueError(f"Unknown indicator: {name}")
    info = INDICATOR_REGISTRY[name]
    params = {**info["default_params"], **kwargs}
    return info["function"](df.copy() if df is not None else df, **params)


def apply_indicators(df: pd.DataFrame, indicator_configs: list[dict]) -> pd.DataFrame:
    """
    Apply multiple indicators.
    indicator_configs: list of {"name": str, "params": dict}
    """
    result = df.copy()
    for cfg in indicator_configs:
        name = cfg["name"]
        params = cfg.get("params", {})
        info = INDICATOR_REGISTRY[name]
        merged = {**info["default_params"], **params}
        result = info["function"](result, **merged)
    return result


def compute_feature_matrix(df: pd.DataFrame, indicator_names: list[str]) -> pd.DataFrame:
    """
    Apply all requested indicators and return only the feature columns
    (excludes OHLCV).  Rows with NaN are dropped.
    """
    result = df.copy()
    for name in indicator_names:
        info = INDICATOR_REGISTRY[name]
        result = info["function"](result, **info["default_params"])

    base_cols = {"Open", "High", "Low", "Close", "Volume"}
    feature_cols = [c for c in result.columns if c not in base_cols]
    return result[feature_cols].dropna()
