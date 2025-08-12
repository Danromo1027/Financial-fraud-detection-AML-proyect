# 🕵️‍♂️💼 AML & Fraud Detection System with Machine Learning

This project implements a **financial fraud detection system** using **Python**, **Data Analysis**, and **Machine Learning**.  
It covers the entire process, from data exploration to building and testing the final predictive model.

---

## 📌 Main Features
- **Data exploration and cleaning**.
- **Feature engineering** to improve model performance.
- **Model training and evaluation** using **XGBoost** as the main algorithm.
- **Model persistence** for production use.
- **Visualizations and dashboards** for result analysis.



## 📂 Project Structure

```bash
📦 Suspicious Activities Detection
├── .venv/                          # Virtual environment (ignored by Git)
├── dashboards/                     # Visualizations and reports
├── data/                           # Dataset folder (ignored by Git)
├── notebooks/                      # Jupyter notebooks for each stage
│   ├── 01_dataexploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_trainning.ipynb
│   ├── 04_XGboost_final.ipynb
│   ├── src/                        # Auxiliary scripts
├── columnas_entrenamiento.pkl      # Training columns
├── config_modelo.json              # Model configuration (threshold, etc.)
├── data_test.py                    # Script to test the model
├── features_seleccionadas.pkl      # Selected features
├── modelo_xgboost_fraude.pkl       # Trained XGBoost model
├── scaler.pkl                      # Data scaler
├── selector_kbest.pkl              # Best features selector
├── requirements.txt                # Project dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation



## 🚀 How to Test the Model



Make sure all model files (.pkl) and config_modelo.json are in the project root.

Edit the data_test.py file to select the test case you want (fraudulent or legitimate).

Change the option variable to the index of the desired example.

Run the script:
