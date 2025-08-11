import pandas as pd

import numpy as np



def create_round_amount_features(df):
    """
    Detecta montos redondos sospechosos
    RED FLAG: Lavadores usan montos exactos para facilitar cálculos
    """
    # Montos exactamente redondos (múltiplos de 1000, 5000, 10000)
    df['round_1000'] = (df['amount'] % 1000 == 0).astype(int)
    df['round_5000'] = (df['amount'] % 5000 == 0).astype(int) 
    df['round_10000'] = (df['amount'] % 10000 == 0).astype(int)
    
    # Score compuesto de "redondez"
    df['roundness_score'] = (
        df['round_1000'] * 1 + 
        df['round_5000'] * 2 + 
        df['round_10000'] * 3
    )
    
    # Detección de montos justo debajo de límites de reporte
    df['just_under_10k'] = ((df['amount'] >= 9000) & (df['amount'] < 10000)).astype(int)
    df['just_under_5k'] = ((df['amount'] >= 4500) & (df['amount'] < 5000)).astype(int)
    
    return df

def create_balance_features(df):
    """
    Detecta inconsistencias en saldos que indican manipulación de cuentas
    RED FLAG: Saldos que no cuadran = posible cuenta fantasma o error sistémico
    """
    # Inconsistencia en cuenta origen
    df['balance_error_orig'] = abs(
        df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    )
    df['balance_inconsistent_orig'] = (df['balance_error_orig'] > 0.01).astype(int)
    
    # Inconsistencia en cuenta destino  
    df['balance_error_dest'] = abs(
        df['oldbalanceDest'] - df['newbalanceDest'] + df['amount']
    )
    df['balance_inconsistent_dest'] = (df['balance_error_dest'] > 0.01).astype(int)
    
    # Score combinado de inconsistencia
    df['total_inconsistency'] = (
        df['balance_inconsistent_orig'] + df['balance_inconsistent_dest']
    )
    
    # Cuentas fantasma (saldo 0 antes y después)
    df['ghost_account_orig'] = (
        (df['oldbalanceOrg'] == 0) & (df['newbalanceOrig'] == 0)
    ).astype(int)
    
    df['ghost_account_dest'] = (
        (df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)
    ).astype(int)
    
    # Cuentas que se vacían completamente
    df['account_drained'] = (
        (df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)
    ).astype(int)
    
    return df

def create_pattern_features(df):
    """
    Detecta patrones de comportamiento sospechosos
    RED FLAG: Comportamientos que se desvían de la norma
    """
    # Actividad fuera de horarios normales (simulado con step)
    df['hour_of_day'] = df['step'] % 24
    df['off_hours'] = (
        (df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)
    ).astype(int)
    
    # Weekend activity (simulado)
    df['day_of_week'] = (df['step'] // 24) % 7
    df['weekend_activity'] = (df['day_of_week'].isin([5, 6])).astype(int)
    
    # Frecuencia de montos específicos (estructuración)
    amount_counts = df['amount'].value_counts()
    df['amount_frequency'] = df['amount'].map(amount_counts)
    df['suspicious_frequency'] = (df['amount_frequency'] > 100).astype(int)
    
    # Transferencias a múltiples destinos (fan-out pattern)
    orig_dest_counts = df.groupby('nameOrig')['nameDest'].nunique()
    df['dest_diversity'] = df['nameOrig'].map(orig_dest_counts)
    df['high_dest_diversity'] = (df['dest_diversity'] > 10).astype(int)
    
    # Monto vs límites de reporte (structured transactions)
    df['structured_amount'] = (
        (df['amount'] >= 9000) & (df['amount'] <= 9999)
    ).astype(int)
    
    return df

def create_statistical_features(df):
    """
    Crea features estadísticas para detectar anomalías
    CONCEPTO: Transacciones que se desvían significativamente de la norma
    """
    # Z-scores por tipo de transacción
    for transaction_type in df['type'].unique():
        mask = df['type'] == transaction_type
        type_data = df[mask]['amount']
        
        mean_amount = type_data.mean()
        std_amount = type_data.std()
        
        df.loc[mask, f'z_score_{transaction_type}'] = (
            (type_data - mean_amount) / std_amount
        )
    
    # Z-score general
    df['z_score_amount'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    df['extreme_z_score'] = (abs(df['z_score_amount']) > 3).astype(int)
    
    # Percentile ranking
    df['amount_percentile'] = df['amount'].rank(pct=True)
    df['top_1_percent'] = (df['amount_percentile'] > 0.99).astype(int)
    df['top_5_percent'] = (df['amount_percentile'] > 0.95).astype(int)
    
    # Log transformation para manejar distribución skewed
    df['log_amount'] = np.log1p(df['amount'])
    df['log_z_score'] = (df['log_amount'] - df['log_amount'].mean()) / df['log_amount'].std()
    
    return df

def create_rolling_features(df):

    """
    Crea features basadas en ventanas temporales
    CONCEPTO: Comportamiento histórico vs actual
    """
    # Ordenar por cliente y tiempo
    df_sorted = df.sort_values(['nameOrig', 'step'])
    
    # Rolling statistics por cliente
    for window in [5, 10, 20]:
        # Promedio móvil de montos
        df_sorted[f'rolling_mean_{window}'] = (
            df_sorted.groupby('nameOrig')['amount']
            .rolling(window=window, min_periods=1)
            .mean().reset_index(0, drop=True)
        )
        
        # Desviación estándar móvil
        df_sorted[f'rolling_std_{window}'] = (
            df_sorted.groupby('nameOrig')['amount']
            .rolling(window=window, min_periods=1)
            .std().reset_index(0, drop=True)
        )
        
        # Ratio vs promedio histórico
        df_sorted[f'amount_vs_rolling_{window}'] = (
            df_sorted['amount'] / df_sorted[f'rolling_mean_{window}']
        )
        
        # Transacciones inusuales (> 2x promedio histórico)
        df_sorted[f'unusual_vs_history_{window}'] = (
            df_sorted[f'amount_vs_rolling_{window}'] > 2
        ).astype(int)
    
    # Frecuencia de transacciones en ventana
    for window in [5, 10]:
        df_sorted[f'txn_frequency_{window}'] = (
            df_sorted.groupby('nameOrig')['step']
            .rolling(window=window, min_periods=1)
            .count().reset_index(0, drop=True)
        )
        
        df_sorted[f'high_frequency_{window}'] = (
            df_sorted[f'txn_frequency_{window}'] > window * 0.8
        ).astype(int)
    
    return df_sorted

def encode_categorical_features(df):
    """
    Prepara variables categóricas para ML
    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    # Label encoding para variables ordinales
    le_type = LabelEncoder()
    df['type_encoded'] = le_type.fit_transform(df['type'])
    
    # One-hot encoding para análisis más profundo
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, type_dummies], axis=1)
    
    # Target encoding basado en tasa de fraude
    type_fraud_rates = df.groupby('type')['isFraud'].mean()
    df['type_fraud_rate'] = df['type'].map(type_fraud_rates)
    
    # Encoding de horas del día en cíclico (para capturar ciclos de 24h)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    print("Categorical features encoded:")
    print(f"Total features: {df.shape[1]}")
    
    return df, le_type

def scale_features(df, target_col='isFraud'):
    """
    Normaliza features para ML
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    
    # Separar features y target (solo columnas numéricas)
    feature_cols = [
        col for col in df.columns
        if col not in ['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud']
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train-test split estratificado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    
    # Standard scaling (z-score normalization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de vuelta a DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    print(f"Features escaladas: {X_train_scaled.shape[1]} variables")
    print(f"Training set: {X_train_scaled.shape[0]} muestras")
    print(f"Test set: {X_test_scaled.shape[0]} muestras")
    print(f"Tasa de fraude train: {y_train.mean():.4f}")
    print(f"Tasa de fraude test: {y_test.mean():.4f}")

    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def select_important_features(X_train, y_train, top_k=20):
       
    #Selecciona las features más importantes para detección de fraude

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    
    
    # Random Forest para feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("🔥 TOP 15 FEATURES MÁS IMPORTANTES:")
    print("="*50)
    for i, (_, row) in enumerate(feature_importance.head(20).iterrows()):
        print(f"{i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")
    
    # Imputar NaNs con la media de cada columna antes de SelectKBest
    X_train_no_nan = X_train.fillna(X_train.mean())

    # Selección automática de mejores features
    selector = SelectKBest(score_func=f_classif, k=top_k)
    X_train_selected = selector.fit_transform(X_train_no_nan, y_train)
    
    # Features seleccionadas
    selected_features = X_train.columns[selector.get_support()]
    print(f"\n✅ {len(selected_features)} features seleccionadas automáticamente")
    
    return feature_importance, selected_features, selector