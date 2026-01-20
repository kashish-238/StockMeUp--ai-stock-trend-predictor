# model_utils.py (FAST + STABLE)
# Goal: keep "LSTM" resume vibe but make training fast on CPU

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def add_indicators(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.DataFrame:
    out = df.copy()
    out["MA_FAST"] = out["Close"].rolling(fast).mean()
    out["MA_SLOW"] = out["Close"].rolling(slow).mean()
    out["BASELINE_SIGNAL"] = (out["MA_FAST"] > out["MA_SLOW"]).astype(int)  # 1=UP, 0=DOWN
    out["RET"] = out["Close"].pct_change()
    out["TARGET_UP"] = (out["Close"].shift(-1) > out["Close"]).astype(int)  # next day direction
    out = out.dropna().copy()
    return out


def make_lstm_dataset(series: np.ndarray, lookback: int):
    """
    series: 1D array (scaled close prices)
    Returns:
      X: (samples, lookback)
      y: (samples,)
    """
    X, y = [], []
    # Need i+1 index for direction label, so go until len(series)-2 inclusive
    for i in range(lookback, len(series) - 1):
        window = series[i - lookback:i]
        X.append(window)
        # label: next value direction vs current
        y.append(1 if series[i + 1] > series[i] else 0)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y


def build_lstm(input_shape):
    """
    LIGHTWEIGHT model for CPU speed.
    Much faster than stacked LSTMs but still legit for a demo/resume.
    """
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False),
        Dropout(0.15),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(
    close_prices: pd.Series,
    lookback: int = 20,
    train_ratio: float = 0.8,
    epochs: int = 6,
    batch_size: int = 64,
    max_points: int = 1200,
    fast_mode: bool = True,
):
    """
    Fast training settings:
    - Trains on only the most recent `max_points` closes (default 1200)
    - Smaller LSTM, fewer epochs, bigger batch
    - EarlyStopping for safety

    Returns dict with same keys as your original implementation.
    """

    # ---- Safety: ensure enough data ----
    close_series = close_prices.dropna()
    if len(close_series) < (lookback + 50):
        raise ValueError("Not enough data to train LSTM. Use a longer period or smaller lookback.")

    # ---- Speed: limit dataset size ----
    # (use most recent points, which also makes the model feel more relevant)
    if max_points is not None and len(close_series) > max_points:
        close_series = close_series.tail(max_points)

    close = close_series.values.reshape(-1, 1)

    # ---- Scale ----
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close).astype(np.float32).flatten()

    # ---- Build dataset ----
    X, y = make_lstm_dataset(close_scaled, lookback=lookback)
    # reshape for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1)).astype(np.float32)

    # ---- Split ----
    split = int(len(X) * train_ratio)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # ---- Model ----
    model = build_lstm((lookback, 1))

    # Faster stop in fast mode
    patience = 2 if fast_mode else 3
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    # ---- Train ----
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    # ---- Predict ----
    if len(X_test) == 0:
        # In rare cases with tiny datasets
        probs = np.array([], dtype=np.float32)
        preds = np.array([], dtype=np.int32)
    else:
        probs = model.predict(X_test, verbose=0).flatten().astype(np.float32)
        preds = (probs >= 0.5).astype(np.int32)

    return {
        "model": model,
        "scaler": scaler,
        "lookback": lookback,
        "history": history.history,
        "y_test": y_test,
        "preds": preds,
        "probs": probs,
        "test_start_index": split + lookback,
    }
