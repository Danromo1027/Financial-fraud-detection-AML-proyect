# Logistic Regression para detección de fraude
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix





def logistic_regression(X_train, y_train, X_test, y_test, threshold=0.5):
    """
    Entrena un modelo de regresión logística con threshold personalizable
    """
    
    # Configurar modelo
    lr_model = LogisticRegression(
        C=10,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    
    # Entrenar
    print("Entrenando Logistic Regression...")
    lr_model.fit(X_train, y_train)
    
    # Obtener probabilidades
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    
    # AQUÍ ESTÁ EL CAMBIO: Usar threshold personalizado
    y_pred = (y_prob >= threshold).astype(int)  # ← THRESHOLD AJUSTABLE
    
    auc_score = roc_auc_score(y_test, y_prob)
    
    print(f"Resultados (Threshold = {threshold}):")
    print("AUC Score:", auc_score)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    return auc_score

# Usar así:
# logistic_regression(X_train, y_train, X_test, y_test, threshold=0.7)


