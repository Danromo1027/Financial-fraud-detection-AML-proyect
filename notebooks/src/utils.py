import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import kagglehub
from kagglehub import KaggleDatasetAdapter

def get_sample_data_kaggle():
    file_path = "Synthetic_Financial_datasets_log.csv"  # nombre real del archivo en Kaggle
    
    # Descargar y cargar el dataset como DataFrame
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "sriharshaeedala/financial-fraud-detection-dataset",  # dataset en Kaggle
        file_path,
        pandas_kwargs={
            "encoding": "latin1",  # evita el error de utf-8
            "low_memory": False    # mejora carga de CSV grande
        }
    )
    print(f"✅ Dataset cargado desde Kaggle: {df.shape[0]} filas y {df.shape[1]} columnas")
    return df

def get_sample_data():
    conn = sqlite3.connect(
        r'C:\Users\User\Desktop\PORTAFOLIO\Detección de Actividades Sospechosas con SQL + Python\data\fraud_detection.db',
        check_same_thread=False
    )
    df_sample = pd.read_sql_query("""
        SELECT * FROM transactions 
        ORDER BY RANDOM() 
        LIMIT 100000
    """, conn)
    conn.close()
    return df_sample

### algoritmos de optimizacion o sintonizacion de hiperparametros

def logistic_regression_sintonizar(X_train, y_train, X_test, y_test):
    # Definición del modelo base
    base_model = LogisticRegression(max_iter=1000)

    # Definición del grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced']
    }

    # Grid Search con validación cruzada
    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    print("🔍 Buscando mejores hiperparámetros para Regresión Logística...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("✅ Mejores parámetros encontrados:", grid.best_params_)

    # Predicciones finales
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("🔎 Evaluación del mejor modelo:")
    print(classification_report(y_test, y_pred))
    print("AUC Score:", roc_auc_score(y_test, y_prob))

    return best_model

"""
result = logistic_regression(X_test_final, y_test, X_train_final, y_train)
print("Modelo de regresión logística entrenado y evaluado.")
print("Características seleccionadas:", result)  # Asumiendo que el cuarto elemento es 'selected_features'
"""

def random_forest_sintonizar(X_train_final, y_train, X_test_final, y_test):
    # Definir la malla de hiperparámetros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 10, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }

    # Modelo base
    rf_base = RandomForestClassifier(random_state=42)

    # Grid Search con validación cruzada
    print("🔍 Optimizando Random Forest...")
    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Entrenar con validación cruzada
    grid_search.fit(X_train_final, y_train)

    # Mejor modelo entrenado
    best_rf = grid_search.best_estimator_

    # Evaluar en test
    y_pred = best_rf.predict(X_test_final)
    y_prob = best_rf.predict_proba(X_test_final)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    # Resultados
    print("\n✅ Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)
    print("\n🎯 Evaluación en conjunto de prueba:")
    print("AUC Score:", auc_score)
    print(classification_report(y_test, y_pred))

    # Importancia de features
    importance_df = pd.DataFrame({
        'feature': X_train_final.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n📊 Top features importantes:")
    print(importance_df.head(10))

    return best_rf, importance_df


def xgboost_sintonizar(X_train_final, y_train, X_test_final, y_test):
    # Calcular peso de clases (importante en fraude)
    normal_count = len(y_train[y_train == 0])
    fraud_count = len(y_train[y_train == 1])
    pos_weight = normal_count / fraud_count

    # Modelo base
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=pos_weight,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )

    # Espacio de búsqueda de hiperparámetros
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3, 0.5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2.0],
    }

    # Randomized Search
    print("🔍 Optimizando XGBoost con RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=30,  # puedes subir si quieres más precisión
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Entrenar el modelo con validación cruzada
    random_search.fit(X_train_final, y_train)

    # Evaluar el mejor modelo encontrado
    best_xgb = random_search.best_estimator_
    y_pred = best_xgb.predict(X_test_final)
    y_prob = best_xgb.predict_proba(X_test_final)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    print("\n✅ Mejores hiperparámetros encontrados:")
    print(random_search.best_params_)

    print("\n🎯 Evaluación del modelo optimizado:")
    print("AUC Score:", auc_score)
    print(classification_report(y_test, y_pred))

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X_train_final.columns,
        'importance': best_xgb.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n📊 Top features importantes:")
    print(importance_df.head(10))
    return best_xgb, importance_df

# XGBoost con ajuste de threshold util para evaluar diferentes métricas
def xgboost_threshold_adjustment(X_train_final, y_train, X_test_final, y_test):
    # XGBoost para máximo rendimiento
    import xgboost as xgb
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

    # Calcular peso para balancear clases
    normal_count = len(y_train[y_train == 0])
    fraud_count = len(y_train[y_train == 1])
    pos_weight = normal_count / fraud_count

    # Configurar modelo
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        subsample=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0,
        colsample_bytree=1.0,
        max_depth=10,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc'
    )

    # Entrenar
    print("Entrenando XGBoost...")
    xgb_model.fit(X_train_final, y_train)

    # Predecir probabilidades
    y_prob = xgb_model.predict_proba(X_test_final)[:, 1]

    # Probar varios thresholds
    thresholds = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15]

    print("\n🎯 Ajuste de Threshold - Evaluación de métricas:\n")
    for threshold in thresholds:
        y_pred_custom = (y_prob >= threshold).astype(int)
        precision = precision_score(y_test, y_pred_custom)
        recall = recall_score(y_test, y_pred_custom)
        f1 = f1_score(y_test, y_pred_custom)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred_custom)

        print(f"🔸 Threshold: {threshold}")
        print(f"   Precision: {precision:.2f}")
        print(f"   Recall:    {recall:.2f}")
        print(f"   F1 Score:  {f1:.2f}")
        print(f"   AUC Score: {auc:.4f}")
        print(f"   Confusion Matrix:\n{cm}\n")

    # ✅ Si decides usar uno fijo, por ejemplo 0.3:
    final_threshold = 0.3
    y_pred_final = (y_prob >= final_threshold).astype(int)

    print(f"🔒 Resultados finales con threshold = {final_threshold}")
    print(classification_report(y_test, y_pred_final))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_final))
    print("AUC Score:", roc_auc_score(y_test, y_prob))

