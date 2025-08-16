# Predictive Maintenance using NASA C-MAPSS Dataset

This project implements a predictive maintenance system using the NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS). The goal is to predict the **Remaining Useful Life (RUL)** of engines using machine learning and deep learning models.

---

## 🚀 Project Overview
- Preprocessing of sensor data from the C-MAPSS FD001 dataset.
- Feature engineering and statistical analysis.
- Model training and evaluation with:
  - Random Forest
  - XGBoost
  - LSTM (Deep Learning)
- Model comparison using RMSE.
- Deployment-ready pipeline for real-time predictions.

---

## 📂 Project Structure
dsproject1-clean/
│-- data/ # Dataset files (train/test)
│-- notebooks/ # Jupyter notebooks for EDA & experimentation
│-- src/ # Source code
│ │-- preprocessing.py
│ │-- feature_engineering.py
│ │-- train_models.py
│ │-- evaluate.py
│ └-- utils.py
│-- models/ # Saved models (RandomForest, XGBoost, LSTM)
│-- requirements.txt # Dependencies
│-- README.md # Project documentation
│-- app.py # Streamlit/Flask deployment script (if added)


---

## ⚙️ Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/anveshasingh12/predictive-maintenance.git
   cd predictive-maintenance
   
Create and activate virtual environment:
python -m venv tfenv
# On Windows (PowerShell):
.\tfenv\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt

📊 Model Results

Random Forest: RMSE ≈ 36.29

XGBoost: RMSE ≈ 38.13

LSTM: RMSE ≈ 60.38

✨ Future Improvements

Hyperparameter tuning

Real-time sensor data integration

Advanced deep learning architectures (GRU, Transformer-based)

👩‍💻 Author

Anvesha Singh
3rd Year CSE Student, Bharati Vidyapeeth College of Engineering