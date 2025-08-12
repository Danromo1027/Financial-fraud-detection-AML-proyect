# ejemplos_transacciones.py
# ---------------------------------------------------------
# 5 ejemplos de transacciones: mezcla de Fraude y Legítimo.
# Cada feature está comentado explicando por qué ese valor
# específico indica o no fraude en ese caso concreto.
# Cambia `opcion` para elegir el caso a probar.
# ---------------------------------------------------------

ejemplos = [
    # ======================================================
    # CASO 0 - FRAUDE
    # ======================================================
    {"Fraude": {
        "step": 50,               # Horas desde el inicio: No relevante directamente al fraude.
        "amount": 9500.0,         # Monto muy alto → alto riesgo, sobre todo si vacía la cuenta.
        "oldbalanceOrg": 9700.0,  # Saldo casi igual al monto → indica posible vaciado total.
        "newbalanceOrig": 200.0,  # Cuenta origen casi vacía después → típico de fraude.
        "oldbalanceDest": 100.0,  # Destino con saldo bajo previo → inusual para recibir tanto.
        "newbalanceDest": 9600.0, # Salto abrupto en saldo destino → actividad sospechosa.
        "hour_of_day": 2,         # Madrugada → fuera de horario habitual de transacciones.
        "off_hours": 1,           # Confirmación de horario inusual → indicador de riesgo.
        "day_of_week": 0,         # Lunes → no es un día crítico, pero la hora sí lo es.
        "weekend_activity": 0,    # No es fin de semana → neutral.
        "amount_frequency": 0,    # Este usuario no suele mover montos similares → sospechoso.
        "suspicious_frequency": 1,# Alta frecuencia reciente de operaciones raras.
        "structured_amount": 0,   # No es un monto justo bajo límites legales → neutro.
        "dest_diversity": 0,      # Siempre al mismo destino → puede ser mula de dinero.
        "high_dest_diversity": 0, # No aplica diversidad → neutro.
        "type_encoded": 4,        # Tipo: transferencia → común en fraudes.
        "type_CASH_IN": 0,        # No es depósito en efectivo → neutro.
        "type_CASH_OUT": 0,       # No es retiro directo.
        "type_DEBIT": 0,          # No es débito.
        "type_PAYMENT": 0,        # No es pago regular.
        "type_TRANSFER": 1,       # Transferencia directa → frecuente en fraudes.
        "type_fraud_rate": 0.25,  # 25% de este tipo de transacciones históricamente son fraude.
        "hour_sin": 0.51,         # Representación de la hora para ML.
        "hour_cos": -0.86         # Representación de la hora para ML.
    }},

    # ======================================================
    # CASO 1 - LEGÍTIMO
    # ======================================================
    {"Legitimo": {
        "step": 12,               # Dentro de primer día laboral → habitual.
        "amount": 75.0,           # Monto pequeño → poco probable que sea fraude.
        "oldbalanceOrg": 500.0,   # Saldo suficiente y no coincide con monto total → normal.
        "newbalanceOrig": 425.0,  # Saldo residual razonable.
        "oldbalanceDest": 300.0,  # Destino ya tenía saldo decente → típico de cuenta activa.
        "newbalanceDest": 375.0,  # Incremento pequeño → normal.
        "hour_of_day": 15,        # Hora laboral → reduce riesgo.
        "off_hours": 0,           # Dentro de horario → legítimo.
        "day_of_week": 2,         # Miércoles → actividad bancaria común.
        "weekend_activity": 0,    # No es fin de semana.
        "amount_frequency": 3,    # Usuario hace montos similares frecuentemente.
        "suspicious_frequency": 0,# Sin picos raros de actividad.
        "structured_amount": 0,   # No hay patrones de estructuración.
        "dest_diversity": 1,      # Envíos a varios destinos → comportamiento normal.
        "high_dest_diversity": 0, # No es excesivo → neutro.
        "type_encoded": 3,        # Tipo: pago → bajo riesgo.
        "type_CASH_IN": 0,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 1,        # Pagos suelen ser legítimos.
        "type_TRANSFER": 0,
        "type_fraud_rate": 0.01,  # Solo 1% de este tipo son fraude históricamente.
        "hour_sin": 0.97,
        "hour_cos": -0.23
    }},

    # ======================================================
    # CASO 2 - FRAUDE
    # ======================================================
    {"Fraude": {
        "step": 5,                # Hora muy temprana en dataset → neutro.
        "amount": 5000.0,         # Monto alto → alerta.
        "oldbalanceOrg": 5100.0,  # Casi todo el saldo → posible vaciado.
        "newbalanceOrig": 100.0,  # Cuenta casi vacía → sospechoso.
        "oldbalanceDest": 50.0,   # Destino sin historial → alerta.
        "newbalanceDest": 5050.0, # Gran salto en saldo → típico mula de dinero.
        "hour_of_day": 4,         # Horario no laboral → sospechoso.
        "off_hours": 1,           # Confirmado fuera de horario.
        "day_of_week": 6,         # Domingo → actividad bancaria menor.
        "weekend_activity": 1,    # Fin de semana → más riesgo.
        "amount_frequency": 0,    # No hay historial de montos así.
        "suspicious_frequency": 1,# Actividad rara en poco tiempo.
        "structured_amount": 1,   # Monto justo bajo umbral legal → indicio de evasión.
        "dest_diversity": 0,      # Un solo destino → posible muleo.
        "high_dest_diversity": 0,
        "type_encoded": 4,        # Transferencia → riesgo alto.
        "type_CASH_IN": 0,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 1,
        "type_fraud_rate": 0.30,
        "hour_sin": 0.69,
        "hour_cos": -0.72
    }},

    # ======================================================
    # CASO 3 - LEGÍTIMO
    # ======================================================
    {"Legitimo": {
        "step": 30,
        "amount": 200.0,
        "oldbalanceOrg": 800.0,
        "newbalanceOrig": 600.0,
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 1200.0,
        "hour_of_day": 10,
        "off_hours": 0,
        "day_of_week": 4,
        "weekend_activity": 0,
        "amount_frequency": 2,
        "suspicious_frequency": 0,
        "structured_amount": 0,
        "dest_diversity": 1,
        "high_dest_diversity": 0,
        "type_encoded": 3,
        "type_CASH_IN": 0,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 1,
        "type_TRANSFER": 0,
        "type_fraud_rate": 0.02,
        "hour_sin": 0.87,
        "hour_cos": 0.49
    }},

    # ======================================================
    # CASO 4 - FRAUDE
    # ======================================================
    {"Fraude": {
        "step": 80,
        "amount": 7800.0,
        "oldbalanceOrg": 8000.0,
        "newbalanceOrig": 200.0,
        "oldbalanceDest": 500.0,
        "newbalanceDest": 8300.0,
        "hour_of_day": 23,
        "off_hours": 1,
        "day_of_week": 5,
        "weekend_activity": 1,
        "amount_frequency": 0,
        "suspicious_frequency": 1,
        "structured_amount": 0,
        "dest_diversity": 0,
        "high_dest_diversity": 0,
        "type_encoded": 4,
        "type_CASH_IN": 0,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 1,
        "type_fraud_rate": 0.40,
        "hour_sin": -0.13,
        "hour_cos": -0.99
    }}
]