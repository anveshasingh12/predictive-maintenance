import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load dataset
df = pd.read_csv("cmaps/train_FD001.txt", sep=' ', header=None)
df.dropna(axis=1, inplace=True)
df.columns = ["unit", "cycle"] + [f"op_{i}" for i in range(1, 4)] + [f"s_{i}" for i in range(1, 22)]

# Step 2: Create RUL
df["max_cycle"] = df.groupby("unit")["cycle"].transform("max")
df["RUL"] = df["max_cycle"] - df["cycle"]
df.drop("max_cycle", axis=1, inplace=True)

# Step 3: Normalize features
features = df.drop(["unit", "RUL"], axis=1)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

X = scaled_features
y = df["RUL"]

# Step 4: Random Forest Model
rf = RandomForestRegressor()
rf.fit(X, y)
joblib.dump(rf, "rf_model.pkl")

# Step 5: XGBoost Model
xgb = XGBRegressor()
xgb.fit(X, y)
joblib.dump(xgb, "xgb_model.pkl")

# Step 6: LSTM requires 3D input
X_lstm = []
y_lstm = []
sequence_length = 20

for i in range(sequence_length, len(X)):
    X_lstm.append(X[i-sequence_length:i])
    y_lstm.append(y.iloc[i])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Build LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_lstm, y_lstm, epochs=5, batch_size=64)

# Save LSTM
model.save("lstm_model.h5")
