# Appliance Energy Prediction — Deep Learning Assessment

Predicting household appliance energy consumption (Wh) using multivariate time-series 
data (temperature, humidity, weather, time-based features) with an LSTM deep learning model.

## Project Structure

├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Cleaned, scaled, feature-engineered datasets
├── notebooks/
│ ├── 01_EDA.ipynb # Exploratory Data Analysis & Preprocessing
│ ├── 02_Feature_Engineering.ipynb # Time-based, rolling, lag, interaction features
│ └── 03_Modeling.ipynb # Baseline models, LSTM, tuning, evaluation
├── models/
│ ├── trained_model.h5 # Final trained LSTM model
│ ├── feature_scaler.pkl # Scaler for input features
│ └── target_scaler.pkl # Scaler for the target variable (Appliances)
├── reports/
│ └── report.pdf # Full write-up (see Documentation section)
├── requirements.txt
└── README.md


## Setup Instructions

1. Clone the repository:
```bash
   git clone https://github.com/DhulakshanKannan/appliance-energy-prediction.git
   cd appliance-energy-prediction
```

2. Create and activate a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Run the notebooks in order:
```bash
   jupyter notebook
```
   Open and run `01_EDA.ipynb` → `02_Feature_Engineering.ipynb` → `03_Modeling.ipynb`

## Approach Summary

- **Preprocessing:** Checked for missing values (none found), detected outliers via IQR 
  (10.83% — kept, as they represent genuine high-usage events), applied Min-Max scaling.
- **Feature Engineering:** Extracted time-based features (hour, day of week, NSM), rolling 
  averages (1h/3h), lagged Appliances values (10min/30min/1h, justified via autocorrelation 
  analysis), and a temperature-humidity interaction term.
- **Modeling:** Compared Linear Regression, Random Forest, and an LSTM neural network. 
  Final LSTM: 2 stacked LSTM layers (64→32 units), dropout regularization, Adam optimizer.
- **Result:** Final LSTM achieved MAE ≈ 32 Wh on the test set (average consumption ≈ 95 Wh).

## Author
Dhulakshan Kannan