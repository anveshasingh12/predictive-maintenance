import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load training data
train_df = pd.read_csv("cmaps/train_FD001.txt", sep=" ", header=None)
train_df.drop([26, 27], axis=1, inplace=True)
train_df.columns = ['unit_number', 'time_in_cycles'] + [f'operational_setting_{i}' for i in range(1, 4)] + [f'sensor_measurement_{i}' for i in range(1, 22)]

# Calculate RUL
def add_rul(df):
    max_cycle = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycle.columns = ['unit_number', 'max_cycle']
    df = df.merge(max_cycle, on='unit_number')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

train_df = add_rul(train_df)

# Feature selection
features = ['time_in_cycles'] + [f'sensor_measurement_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20]]

# Normalize features
scaler = MinMaxScaler()
train_df[features] = scaler.fit_transform(train_df[features])

# Train/test split
X = train_df[features]
y = train_df['RUL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# 1. Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"Random Forest RMSE: {rf_rmse:.2f}")

# ------------------------
# 2. XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
print(f"XGBoost RMSE: {xgb_rmse:.2f}")

# ------------------------
# 3. LSTM
# Create sequence data for LSTM
sequence_length = 30

def create_sequences(df, sequence_length, features):
    X, y = [], []
    for unit in df['unit_number'].unique():
        unit_df = df[df['unit_number'] == unit]
        for i in range(len(unit_df) - sequence_length):
            seq_x = unit_df[features].iloc[i:i+sequence_length].values
            seq_y = unit_df['RUL'].iloc[i+sequence_length]
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(train_df, sequence_length, features)

# Train/test split for LSTM
split = int(0.8 * len(X_lstm))
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

# Build LSTM model
model_lstm = Sequential([
    LSTM(100, input_shape=(sequence_length, len(features))),
    Dense(1)
])
model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=64, verbose=1)

# Predict with LSTM
y_pred_lstm = model_lstm.predict(X_test_lstm).flatten()
lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))
print(f"LSTM RMSE: {lstm_rmse:.2f}")

# ------------------------
# Final comparison
print("\nModel Comparison:")
print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"XGBoost RMSE: {xgb_rmse:.2f}")
print(f"LSTM RMSE: {lstm_rmse:.2f}")

from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7]
}

grid = GridSearchCV(XGBRegressor(random_state=42), params, cv=3, scoring='neg_root_mean_squared_error')
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
best_xgb = grid.best_estimator_
best_xgb_pred = best_xgb.predict(X_test)
best_xgb_rmse = np.sqrt(mean_squared_error(y_test, best_xgb_pred))
print(f"Tuned XGBoost RMSE: {best_xgb_rmse:.2f}")
import time

def simulate_streaming(model, test_unit_df, features, delay=1.0):
    print("Starting RUL prediction stream...")
    for i in range(len(test_unit_df)):
        input_data = test_unit_df[features].iloc[i:i+1]
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        print(f"Cycle {test_unit_df.iloc[i]['time_in_cycles']}: Predicted RUL = {prediction:.2f}")
        time.sleep(delay)  # Simulate real-time delay

# Example: Stream for unit 10
test_unit = train_df[train_df['unit_number'] == 10]
simulate_streaming(best_xgb, test_unit, features, delay=0.5)
import joblib

# Save models
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
model_lstm.save("lstm_model.h5")
joblib.dump(scaler, "scaler.pkl")


