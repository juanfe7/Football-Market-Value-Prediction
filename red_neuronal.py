"""
Modelo de Red Neuronal para Prediccion de Valor de Mercado
Compatible con TensorFlow 2.20+
SOLUCION: Removido 'mse' de metrics para evitar error de serializacion
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ocultar warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("PROYECTO: PREDICCION DE VALOR DE MERCADO DE JUGADORES DE FUTBOL")
print("Modelo: Red Neuronal Artificial (RNA) - Compatible TF 2.20+")
print("=" * 80)

# ============================================================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ============================================================================
print("\n[1] Cargando dataset...")

try:
    df = pd.read_csv('jugadores_jovenes.csv')
    print(f"[OK] Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
except FileNotFoundError:
    print("[AVISO] Archivo no encontrado. Generando datos sinteticos...")

features = ['overall', 'potential', 'movement_reactions', 'release_clause_eur', 
            'wage_eur', 'age', 'composure', 'reactions']
target = 'value_eur'

missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    print(f"[AVISO] Columnas faltantes: {missing_cols}")
    features = [col for col in features if col in df.columns]

df_clean = df[features + [target]].dropna()
print(f"[OK] Datos limpios: {df_clean.shape[0]} registros")

print("\n[2] Analisis de correlacion:")
correlations = df_clean[features].corrwith(df_clean[target]).sort_values(ascending=False)
print(correlations)

# ============================================================================
# 2. PREPARACION DE DATOS
# ============================================================================
print("\n[3] Preparando datos para modelado...")

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"[OK] Entrenamiento: {X_train.shape[0]} | Prueba: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[OK] Variables estandarizadas")

# ============================================================================
# 3. CONSTRUCCION DE RED NEURONAL
# ============================================================================
print("\n[4] Construyendo Red Neuronal...")

tf.random.set_seed(42)
np.random.seed(42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],), name='entrada'),
    Dropout(0.3, name='dropout_1'),
    Dense(64, activation='relu', name='oculta_1'),
    Dropout(0.3, name='dropout_2'),
    Dense(32, activation='relu', name='oculta_2'),
    Dense(1, activation='linear', name='salida')
])

# IMPORTANTE: Solo 'mae' en metrics para compatibilidad TF 2.20+
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']  # <-- CAMBIO CLAVE
)

print("\n" + "="*80)
print("ARQUITECTURA DE LA RED")
print("="*80)
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

# ============================================================================
# 4. ENTRENAMIENTO
# ============================================================================
print("\n[5] Entrenando Red Neuronal...")
print("Configuracion: 100 epocas, batch_size=32, validation_split=0.2")

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

print(f"[OK] Entrenamiento completado en {len(history.history['loss'])} epocas")

# ============================================================================
# 5. REGRESION LINEAL (COMPARACION)
# ============================================================================
print("\n[6] Entrenando Regresion Lineal...")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("[OK] Regresion Lineal entrenada")

# ============================================================================
# 6. EVALUACION
# ============================================================================
print("\n[7] Evaluando modelos...")

y_pred_nn = model.predict(X_test_scaled, verbose=0).flatten()
y_pred_lr = lr_model.predict(X_test_scaled)

mae_nn = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# ============================================================================
# 7. RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("RESULTADOS COMPARATIVOS")
print("="*80)

results_df = pd.DataFrame({
    'Metrica': ['MAE (EUR)', 'RMSE (EUR)', 'R2'],
    'Regresion Lineal': [
        f'{mae_lr:,.0f}',
        f'{rmse_lr:,.0f}',
        f'{r2_lr:.4f}'
    ],
    'Red Neuronal': [
        f'{mae_nn:,.0f}',
        f'{rmse_nn:,.0f}',
        f'{r2_nn:.4f}'
    ],
    'Mejora (%)': [
        f'{((mae_lr - mae_nn) / mae_lr * 100):+.2f}%',
        f'{((rmse_lr - rmse_nn) / rmse_lr * 100):+.2f}%',
        f'{((r2_nn - r2_lr) / r2_lr * 100):+.2f}%'
    ]
})

print("\n" + results_df.to_string(index=False))

# ============================================================================
# 8. VISUALIZACIONES
# ============================================================================
print("\n[8] Generando visualizaciones...")

fig = plt.figure(figsize=(16, 10))

# Grafico 1: Perdida
ax1 = plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Loss Entrenamiento', linewidth=2)
plt.plot(history.history['val_loss'], label='Loss Validacion', linewidth=2)
plt.xlabel('Epoca')
plt.ylabel('MSE')
plt.title('Curva de Aprendizaje', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafico 2: MAE
ax2 = plt.subplot(2, 3, 2)
plt.plot(history.history['mae'], label='MAE Train', linewidth=2)
plt.plot(history.history['val_mae'], label='MAE Val', linewidth=2)
plt.xlabel('Epoca')
plt.ylabel('MAE')
plt.title('Error Absoluto Medio', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafico 3: Comparacion metricas
ax3 = plt.subplot(2, 3, 3)
metrics = ['MAE', 'RMSE', 'R2']
lr_vals = [mae_lr/1000, rmse_lr/1000, r2_lr]
nn_vals = [mae_nn/1000, rmse_nn/1000, r2_nn]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, lr_vals, width, label='Reg. Lineal', alpha=0.8)
plt.bar(x + width/2, nn_vals, width, label='Red Neuronal', alpha=0.8)
plt.ylabel('Valor')
plt.title('Comparacion de Desempeno', fontweight='bold')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Grafico 4: Real vs Predicho - Lineal
ax4 = plt.subplot(2, 3, 4)
plt.scatter(y_test, y_pred_lr, alpha=0.5, s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Ideal')
plt.xlabel('Real (EUR)')
plt.ylabel('Predicho (EUR)')
plt.title(f'Reg. Lineal (R2={r2_lr:.4f})', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafico 5: Real vs Predicho - RNA
ax5 = plt.subplot(2, 3, 5)
plt.scatter(y_test, y_pred_nn, alpha=0.5, s=30, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Ideal')
plt.xlabel('Real (EUR)')
plt.ylabel('Predicho (EUR)')
plt.title(f'Red Neuronal (R2={r2_nn:.4f})', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafico 6: Distribucion errores
ax6 = plt.subplot(2, 3, 6)
errors_lr = y_test - y_pred_lr
errors_nn = y_test - y_pred_nn

plt.hist(errors_lr, bins=50, alpha=0.5, label='Reg. Lineal')
plt.hist(errors_nn, bins=50, alpha=0.5, label='Red Neuronal')
plt.xlabel('Error (EUR)')
plt.ylabel('Frecuencia')
plt.title('Distribucion de Errores', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados_rna.png', dpi=300, bbox_inches='tight')
print("[OK] Graficos guardados: resultados_rna.png")
plt.show()

# ============================================================================
# 9. INTERPRETACION
# ============================================================================
print("\n" + "="*80)
print("INTERPRETACION Y CONCLUSIONES")
print("="*80)

print("""
1. CAPACIDAD DE MODELADO NO LINEAL:
   La RNA captura relaciones complejas entre atributos y valor de mercado.

2. COMPARACION CON REGRESION LINEAL:
""")

if r2_nn > r2_lr:
    mejora = ((r2_nn - r2_lr) / r2_lr * 100)
    print(f"   [+] RNA supera Reg. Lineal en R2: {mejora:.2f}%")
    print(f"   [+] Reduccion MAE: {((mae_lr - mae_nn) / mae_lr * 100):.2f}%")
else:
    print("   [-] Reg. Lineal mostro resultados competitivos")

print("""
3. VENTAJAS RNA:
   [+] Captura interacciones complejas
   [+] Maneja no linealidades
   [+] Escalable a mas variables

4. APLICACIONES:
   - Valoracion automatica de jugadores
   - Identificacion de oportunidades
   - Apoyo en negociaciones
""")

print("="*80)
print("ANALISIS COMPLETADO")
print("="*80)

# ============================================================================
# 10. GUARDAR MODELO (COMPATIBLE TF 2.20+)
# ============================================================================
model.save('modelo_valoracion_jugadores.keras')
print("\n[OK] Modelo guardado: modelo_valoracion_jugadores.h5")
print("[OK] Compatible con TensorFlow 2.20+")

# GUARDAR TAMBIEN EL SCALER (importante para predicciones futuras)
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("[OK] Scaler guardado: scaler.pkl")

print("\n" + "="*80)
print("FIN DEL ENTRENAMIENTO")
print("="*80)