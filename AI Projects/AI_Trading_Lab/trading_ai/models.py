"""
models.py - AI Trading Models
Provides model training, evaluation, and prediction for trading strategies.
Supports: Random Forest, Gradient Boosting, SVM, and LSTM (deep learning).
"""

import json
import pickle
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)


# ── Data Preparation ───────────────────────────────────────────────────────

def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_horizon: int = 1,
    target_type: str = "direction",  # "direction" or "threshold"
    threshold_pct: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Build X (features) and y (labels) from a DataFrame that already has
    indicator columns computed.

    target_type:
        - "direction": 1 if Close rises over *target_horizon* bars, else 0
        - "threshold": 1 if return > threshold_pct, -1 if < -threshold_pct, else 0

    Returns (X, y, index) with NaN rows dropped.
    """
    tmp = df.copy()
    future_ret = tmp["Close"].pct_change(target_horizon).shift(-target_horizon)

    if target_type == "direction":
        tmp["_target"] = (future_ret > 0).astype(int)
    else:
        tmp["_target"] = 0
        tmp.loc[future_ret > threshold_pct / 100, "_target"] = 1
        tmp.loc[future_ret < -threshold_pct / 100, "_target"] = -1

    # Drop rows with NaN in features or target
    cols_needed = feature_cols + ["_target"]
    tmp.dropna(subset=cols_needed, inplace=True)

    X = tmp[feature_cols].values
    y = tmp["_target"].values
    idx = tmp.index
    return X, y, idx


def time_series_split(X, y, n_splits=5):
    """Yield train/test splits respecting time order."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ── Model Wrapper ──────────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    confusion: list = field(default_factory=list)
    classification_report: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class TradingModel:
    model_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    model_type: str = ""
    symbol: str = ""
    indicators: list = field(default_factory=list)
    feature_cols: list = field(default_factory=list)
    hyperparams: dict = field(default_factory=dict)
    metrics: Optional[ModelMetrics] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    trained: bool = False

    # Runtime objects (not serialized)
    _model: object = field(default=None, repr=False)
    _scaler: object = field(default=None, repr=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained or self._model is None:
            raise RuntimeError("Model not trained yet")
        X_scaled = self._scaler.transform(X) if self._scaler else X
        return self._model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self.trained or self._model is None:
            raise RuntimeError("Model not trained yet")
        X_scaled = self._scaler.transform(X) if self._scaler else X
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X_scaled)
        return None

    def save(self):
        path = MODELS_DIR / f"{self.model_id}"
        path.mkdir(exist_ok=True)
        # Save sklearn/model object
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self._model, f)
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f)
        # Save metadata
        meta = {
            "model_id": self.model_id,
            "name": self.name,
            "model_type": self.model_type,
            "symbol": self.symbol,
            "indicators": self.indicators,
            "feature_cols": self.feature_cols,
            "hyperparams": self.hyperparams,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "created_at": self.created_at,
            "trained": self.trained,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_id: str) -> "TradingModel":
        path = MODELS_DIR / model_id
        with open(path / "meta.json") as f:
            meta = json.load(f)
        with open(path / "model.pkl", "rb") as f:
            model_obj = pickle.load(f)
        with open(path / "scaler.pkl", "rb") as f:
            scaler_obj = pickle.load(f)

        metrics = ModelMetrics(**meta["metrics"]) if meta.get("metrics") else None
        tm = cls(
            model_id=meta["model_id"],
            name=meta["name"],
            model_type=meta["model_type"],
            symbol=meta["symbol"],
            indicators=meta["indicators"],
            feature_cols=meta["feature_cols"],
            hyperparams=meta["hyperparams"],
            metrics=metrics,
            created_at=meta["created_at"],
            trained=meta["trained"],
        )
        tm._model = model_obj
        tm._scaler = scaler_obj
        return tm


# ── Model Factory ──────────────────────────────────────────────────────────

MODEL_TYPES = {
    "Random Forest": {
        "class": RandomForestClassifier,
        "default_params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "Gradient Boosting": {
        "class": GradientBoostingClassifier,
        "default_params": {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        },
    },
    "SVM": {
        "class": SVC,
        "default_params": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
        },
    },
}


def get_model_types() -> list[str]:
    types = list(MODEL_TYPES.keys())
    try:
        import tensorflow  # noqa
        types.append("LSTM")
    except ImportError:
        pass
    return types


def _build_sklearn_model(model_type: str, params: dict):
    import inspect
    info = MODEL_TYPES[model_type]
    merged = {**info["default_params"], **params}
    # Only pass params that the model class actually accepts
    valid_keys = set(inspect.signature(info["class"].__init__).parameters.keys())
    filtered = {k: v for k, v in merged.items() if k in valid_keys}
    return info["class"](**filtered), filtered


def _build_lstm_model(input_shape: tuple, params: dict):
    """Build a Keras LSTM model for binary classification."""
    from tensorflow import keras
    from tensorflow.keras import layers

    units = params.get("lstm_units", 64)
    dropout = params.get("dropout", 0.3)
    lr = params.get("learning_rate", 0.001)

    model = keras.Sequential([
        layers.LSTM(units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(dropout),
        layers.LSTM(units // 2),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training Engine ─────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str = "Random Forest",
    name: str = "",
    symbol: str = "",
    indicators: list[str] = None,
    hyperparams: dict = None,
    target_horizon: int = 1,
    test_ratio: float = 0.2,
    lookback: int = 10,
    progress_callback=None,
) -> TradingModel:
    """
    Full training pipeline:
    1. Prepare features & labels
    2. Scale features
    3. Train model
    4. Evaluate on hold-out set
    5. Return a TradingModel object
    """
    if hyperparams is None:
        hyperparams = {}
    if indicators is None:
        indicators = []

    if progress_callback:
        progress_callback(10, "Preparing data...")

    X, y, idx = prepare_training_data(df, feature_cols, target_horizon=target_horizon)

    if len(X) < 50:
        raise ValueError(f"Not enough data rows ({len(X)}) after dropping NaN. Need ≥ 50.")

    # Train / test split (respecting time order)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if progress_callback:
        progress_callback(25, "Scaling features...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if progress_callback:
        progress_callback(40, f"Training {model_type}...")

    if model_type == "LSTM":
        # Reshape for LSTM: (samples, lookback, features)
        def make_sequences(X_arr, y_arr, lb):
            xs, ys = [], []
            for i in range(lb, len(X_arr)):
                xs.append(X_arr[i - lb:i])
                ys.append(y_arr[i])
            return np.array(xs), np.array(ys)

        X_train_seq, y_train_seq = make_sequences(X_train_s, y_train, lookback)
        X_test_seq, y_test_seq = make_sequences(X_test_s, y_test, lookback)

        model_obj = _build_lstm_model(
            input_shape=(lookback, X_train_s.shape[1]),
            params=hyperparams,
        )
        epochs = hyperparams.get("epochs", 50)
        batch_size = hyperparams.get("batch_size", 32)
        model_obj.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            verbose=0,
        )
        y_pred_raw = model_obj.predict(X_test_seq, verbose=0).flatten()
        y_pred = (y_pred_raw > 0.5).astype(int)
        y_eval = y_test_seq
        used_params = hyperparams
    else:
        model_obj, used_params = _build_sklearn_model(model_type, hyperparams)
        model_obj.fit(X_train_s, y_train)
        y_pred = model_obj.predict(X_test_s)
        y_eval = y_test

    if progress_callback:
        progress_callback(80, "Evaluating...")

    metrics = ModelMetrics(
        accuracy=round(accuracy_score(y_eval, y_pred), 4),
        precision=round(precision_score(y_eval, y_pred, average="weighted", zero_division=0), 4),
        recall=round(recall_score(y_eval, y_pred, average="weighted", zero_division=0), 4),
        f1=round(f1_score(y_eval, y_pred, average="weighted", zero_division=0), 4),
        confusion=confusion_matrix(y_eval, y_pred).tolist(),
        classification_report=classification_report(y_eval, y_pred, zero_division=0),
    )

    tm = TradingModel(
        name=name or f"{model_type}_{symbol}_{datetime.now().strftime('%H%M%S')}",
        model_type=model_type,
        symbol=symbol,
        indicators=indicators,
        feature_cols=feature_cols,
        hyperparams=used_params,
        metrics=metrics,
        trained=True,
    )
    tm._model = model_obj
    tm._scaler = scaler

    if progress_callback:
        progress_callback(100, "Done!")

    return tm


def list_saved_models() -> list[dict]:
    """Return metadata for all saved models."""
    results = []
    for d in MODELS_DIR.iterdir():
        meta_path = d / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                results.append(json.load(f))
    return results


def delete_model(model_id: str):
    import shutil
    path = MODELS_DIR / model_id
    if path.exists():
        shutil.rmtree(path)
