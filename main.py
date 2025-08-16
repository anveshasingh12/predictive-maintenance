import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
os.environ['TCL_LIBRARY'] = r"C:\Users\tcl\tcl8.6"
os.environ['TK_LIBRARY'] = r'C:\Users\tcl\tk8.6'


# Path setup
data_folder = r'C:\Users\anvesha singh\PycharmProjects\dsproject1\CMaps'

# Column names
cols = ['unit_number', 'time_in_cycles'] + \
       ['op_setting_1', 'op_setting_2', 'op_setting_3'] + \
       [f'sensor_{i}' for i in range(1, 22)]

# Load train function
def load_train_data(filename):
    path = os.path.join(data_folder, filename)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = cols
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    df = df.merge(max_cycles, on='unit_number')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

# Load test function
def load_test_data(filename):
    path = os.path.join(data_folder, filename)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = cols
    return df

# Load RUL file
def load_rul_file(filename):
    return pd.read_csv(os.path.join(data_folder, filename), header=None)

# Load one dataset (FD001 for example)
train_df = load_train_data("train_FD001.txt")
test_df = load_test_data("test_FD001.txt")
true_rul = load_rul_file("RUL_FD001.txt")

# Select features (remove op_settings if not useful)
features = [col for col in train_df.columns if col.startswith('sensor_')]

# Normalize features
scaler = MinMaxScaler()
train_df[features] = scaler.fit_transform(train_df[features])
test_df[features] = scaler.transform(test_df[features])

# Get last cycle for each engine from test data
last_cycles = test_df.groupby('unit_number').last().reset_index()

# Add true RUL from file
last_cycles['RUL'] = true_rul.values

# Train data preparation
X_train = train_df[features]
y_train = train_df['RUL']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict RUL on test
X_test = last_cycles[features]
y_test = last_cycles['RUL']
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error on FD001 test set: {rmse:.2f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='True RUL', marker='o')
plt.plot(y_pred, label='Predicted RUL', marker='x')
plt.title('RUL Prediction on FD001')
plt.xlabel('Engine Number')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)
plt.show()



