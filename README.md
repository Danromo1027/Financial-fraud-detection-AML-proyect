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


```
## ⚙️ Installation
```bash
# 1️⃣ Clone the repository
git clone https://github.com/your_username/your_repository.git
cd your_repository

# 2️⃣ Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

```


## 🚀 How to Test the Model

-  Make sure all model files (.pkl) and config_modelo.json are in the project root.
-  Edit the data_test.py file to select the test case you want (fraudulent or legitimate).
     Change the option variable to the index of the desired example.
-  Run the script:

```bash
python notebooks/data_test.py

```
- The script will output:

- - Whether the transaction is fraudulent or legitimate.
- - The fraud probability estimated by the model.

## 🧠 Example Output

``` plaintext
Available examples:
0 -> Fraud
1 -> Legitimate
2 -> Fraud
3 -> Legitimate
4 -> Fraud

Example: Fraud
Is it fraud?: ✅ YES
Fraud probability: 96.45%

```

## 📊 Feature Description
| Feature                | Description                                         | Relation to Fraud                                                |
| ---------------------- | --------------------------------------------------- | ---------------------------------------------------------------- |
| `step`                 | Time step in simulated hours.                       | Fraud often occurs at specific times or unusual hours.           |
| `amount`               | Transaction amount.                                 | Unusually high amounts are a key indicator.                      |
| `oldbalanceOrg`        | Origin account balance before the transaction.      | High balance combined with a full withdrawal can indicate fraud. |
| `newbalanceOrig`       | Origin account balance after the transaction.       | Accounts left almost empty are suspicious.                       |
| `oldbalanceDest`       | Destination account balance before the transaction. | Destination accounts with low prior balance are common in fraud. |
| `newbalanceDest`       | Destination account balance after the transaction.  | Large sudden increases may indicate suspicious activity.         |
| `hour_of_day`          | Transaction hour.                                   | Off-hours activity increases fraud risk.                         |
| `off_hours`            | Whether the transaction is outside working hours.   | Many frauds occur late at night or during unusual hours.         |
| `day_of_week`          | Day of the week.                                    | Some fraud patterns are more common on weekdays.                 |
| `weekend_activity`     | Whether it happened on a weekend.                   | Some fraud takes advantage of reduced oversight on weekends.     |
| `amount_frequency`     | Frequency of transactions with this amount.         | Amounts outside the user’s usual pattern are suspicious.         |
| `suspicious_frequency` | High frequency of transactions in a short time.     | Common in compromised accounts.                                  |
| `structured_amount`    | Whether the amount was split to avoid detection.    | Technique used for "smurfing".                                   |
| `dest_diversity`       | Number of different recipients.                     | Fraud sometimes involves multiple recipients.                    |
| `high_dest_diversity`  | High diversity of recipients.                       | Can be a sign of money laundering.                               |
| `type_encoded`         | Transaction type encoded numerically.               | Some types are more vulnerable to fraud.                         |
| `type_CASH_IN`         | Incoming cash operation.                            | Low fraud rate.                                                  |
| `type_CASH_OUT`        | Cash withdrawal.                                    | Frequently used in fraud.                                        |
| `type_DEBIT`           | Direct debit.                                       | Less common in fraud cases.                                      |
| `type_PAYMENT`         | Payments.                                           | Moderate risk.                                                   |
| `type_TRANSFER`        | Transfers.                                          | Very high correlation with fraud.                                |
| `type_fraud_rate`      | Historical fraud rate by transaction type.          | Key metric for prediction.                                       |
| `hour_sin`             | Trigonometric transformation of the hour.           | Captures hourly patterns.                                        |
| `hour_cos`             | Trigonometric transformation of the hour.           | Complements `hour_sin`.                                          |

