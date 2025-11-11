# 🍔 Food Delivery Time Prediction (Enhanced)

An enhanced **machine learning pipeline** to predict **food delivery times** across different Indian cities using contextual, temporal, and geospatial features such as **traffic**, **weather**, **distance**, and **festival periods**.

The project compares several regression algorithms — **Linear Regression**, **Random Forest**, **XGBoost**, **LightGBM**, and **Stacking Ensemble Models** — and demonstrates that ensemble-based approaches achieve the best accuracy.

---

## 📘 Project Overview

This project aims to accurately estimate the time required for a food delivery order to reach the customer after being placed.  
The system is built to help delivery platforms **optimize delivery logistics**, **reduce customer waiting time**, and **enhance user satisfaction**.

**Key Objectives**
- Predict actual delivery time using real-world factors.
- Compare multiple ML models for best predictive accuracy.
- Evaluate performance using RMSE and R² metrics.
- Build a reproducible pipeline with preprocessing, feature engineering, training, and evaluation.

---

## 📂 Repository Contents

| File/Folder | Description |
|--------------|-------------|
| `Enhanced_Delivery_Prediction_Enhanced (1).ipynb` | Main Jupyter Notebook containing full code, preprocessing, modeling, and evaluation. |
| `Paper.pdf` | Report describing dataset, methodology, results, and analysis. |
| `data/` *(optional)* | Folder to store dataset files locally. |

---

## 🧠 Model Summary

| Model | Description | RMSE | R² |
|--------|--------------|------|----|
| Linear Regression | Baseline model | 0.58 | 0.69 |
| Random Forest | Ensemble using bagging | 0.42 | 0.77 |
| XGBoost | Boosting algorithm | 0.37 | 0.79 |
| LightGBM | Gradient boosting with leaf-wise growth | 0.35 | 0.80 |
| **Stacking Ensemble (Best)** | Combines base learners + meta model | **0.33** | **0.81** |

---

## ⚙️ System Requirements

- Python 3.9 or higher  
- 8GB+ RAM recommended  
- OS: Windows / macOS / Linux  

**Install Required Libraries**
 ```bash 
pip install numpy pandas scikit-learn lightgbm xgboost matplotlib seaborn shap jupyterlab joblib
```
🚀 How to Run the Project (Step-by-Step)

1.Clone the Repository
```bash
git clone https://github.com/Venkat-esh-R/Food-Delivery-Time-Prediction.git
cd Food-Delivery-Time-Prediction
```
2.Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # For macOS/Linux
venv\Scripts\activate           # For Windows
```
3.Install Dependencies
```bash
pip install -r requirements.txt
```
4.Add Dataset

Place your dataset file (CSV) inside a folder named data/ or at the path used in the notebook.

The cleaned dataset used in the report contains ~41,368 records after removing null values.

5.Run the Notebook
```bash
jupyter notebook
```
or
```bash
jupyter lab
```

Open Enhanced_Delivery_Prediction_Enhanced (1).ipynb

Run all cells sequentially from top to bottom.

6.View Results

Performance metrics (RMSE, R²) will be displayed in the output cells.

Model comparison plots and SHAP feature importance graphs will be shown.

The final model (stacking ensemble) achieves the best results.

🔄 Workflow Summary

1.Data Loading → Read dataset and inspect columns

2.Preprocessing → Handle nulls, convert datatypes, encode categorical values

3.Feature Engineering →

4.Calculate Haversine distance

5.Extract day, hour, and time features

6.Add festival/holiday flags

7.Model Training → Fit multiple regression algorithms

8.Evaluation → Compare RMSE, R²; visualize residuals & feature importances

9.Result Interpretation → Determine best model (Stacking Ensemble)

🏁 Results & Interpretation

The Stacking Ensemble combining Random Forest, XGBoost, and LightGBM achieved the lowest RMSE (0.33) and highest R² (0.81).

Factors such as road traffic density, festival days, delivery distance, and delivery person rating had the most impact on prediction accuracy.

Ensemble techniques significantly outperformed individual models, confirming that model blending improves generalization.

📊 Key Insights

Distance and Traffic Density are primary predictors of delay.

Festival/Weather conditions can add significant variance to delivery time.

Customer & Delivery Partner Ratings improve reliability of predictions.

Feature engineering and proper scaling drastically improved model accuracy.

🧩 Future Enhancements

Integrate real-time traffic and weather APIs for dynamic predictions.

Deploy as a Flask/FastAPI web app for interactive use.

Add Geo-based clustering to further improve model granularity.

Explore deep learning models (LSTM) for time-dependent predictions.


