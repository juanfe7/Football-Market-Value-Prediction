"""
Script para Probar el Modelo Entrenado con Jugadores Individuales
Simula el uso del modelo en produccion/scouting
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ocultar warnings de TensorFlow

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SISTEMA DE VALORACION DE JUGADORES - MODO PRUEBA")
print("="*80)

# ============================================================================
# 1. CARGAR MODELO ENTRENADO
# ============================================================================
print("\n[1] Cargando modelo entrenado...")

# Verificar directorio actual
print(f"[INFO] Directorio actual: {os.getcwd()}")

# Buscar archivo del modelo
nombre_modelo = 'modelo_valoracion_jugadores.keras'
ruta_modelo = os.path.join(os.getcwd(), nombre_modelo)

print(f"[INFO] Buscando: {ruta_modelo}")

if not os.path.exists(ruta_modelo):
    print(f"[ERROR] No se encontro el archivo: {nombre_modelo}")
    print("\n[SOLUCION] Archivos .h5 disponibles en este directorio:")
    archivos_h5 = [f for f in os.listdir('.') if f.endswith('.h5')]
    if archivos_h5:
        for f in archivos_h5:
            print(f"  - {f}")
        print(f"\n[TIP] Si ves otro archivo .h5, cambia el nombre en el script")
    else:
        print("  (ninguno)")
        print("\n[ACCION REQUERIDA] Entrena el modelo primero:")
        print("  python red_neuronal.py")
    exit()

try:
    modelo = keras.models.load_model(ruta_modelo)
    print("[OK] Modelo cargado exitosamente")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    print("\n[SOLUCION] Re-entrena el modelo ejecutando:")
    print("  python red_neuronal.py")
    exit()

# ============================================================================
# 2. CARGAR SCALER (necesario para normalizar datos nuevos)
# ============================================================================
print("\n[2] Preparando escalador de datos...")

# Intentar cargar scaler guardado primero
import pickle
scaler = None

if os.path.exists('scaler.pkl'):
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("[OK] Scaler cargado desde scaler.pkl")
    except:
        print("[AVISO] No se pudo cargar scaler.pkl, creando nuevo...")

# Si no hay scaler guardado, crear uno nuevo
if scaler is None:
    try:
        df = pd.read_csv('jugadores_jovenes.csv')
        print(f"[INFO] CSV cargado con columnas: {df.columns.tolist()}")
    except:
        # Generar datos sinteticos (misma semilla)
        np.random.seed(42)
        n_samples = 2000
        df = pd.DataFrame({
            'overall': np.random.randint(60, 90, n_samples),
            'potential': np.random.randint(65, 95, n_samples),
            'age': np.random.randint(18, 24, n_samples),
            'movement_reactions': np.random.randint(60, 90, n_samples),
            'composure': np.random.randint(55, 85, n_samples),
            'reactions': np.random.randint(60, 90, n_samples),
        })
        df['wage_eur'] = (df['overall'] * 1000 + df['potential'] * 800 + 
                          np.random.normal(0, 5000, n_samples)).clip(lower=5000)
        df['release_clause_eur'] = (df['overall'] ** 2 * 10000 + 
                                     df['potential'] ** 2 * 8000 + 
                                     np.random.normal(0, 500000, n_samples)).clip(lower=100000)
        print("[INFO] Usando datos sinteticos")
    
    # Detectar features disponibles
    features_deseadas = ['overall', 'potential', 'movement_reactions', 'release_clause_eur', 
                         'wage_eur', 'age', 'composure', 'reactions']
    
    features = [f for f in features_deseadas if f in df.columns]
    
    if len(features) != len(features_deseadas):
        faltantes = set(features_deseadas) - set(features)
        print(f"[AVISO] Columnas faltantes en CSV: {faltantes}")
        print(f"[INFO] Usando columnas disponibles: {features}")
    
    # Ajustar el scaler
    scaler = StandardScaler()
    scaler.fit(df[features])
    print("[OK] Escalador preparado")
    
    # Guardar para uso futuro
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("[OK] Scaler guardado en scaler.pkl")

# Determinar el numero de features esperadas por el modelo
n_features = scaler.n_features_in_
print(f"[INFO] Modelo espera {n_features} features")

# ============================================================================
# 3. DEFINIR JUGADORES DE PRUEBA
# ============================================================================
print("\n[3] Definiendo jugadores de prueba...")

# Ejemplo 1: Joven promesa (estilo Mbappe joven)
jugador_1 = {
    'nombre': 'Joven Promesa',
    'overall': 85,
    'potential': 92,
    'movement_reactions': 88,
    'release_clause_eur': 80000000,
    'wage_eur': 150000,
    'age': 20,
    'composure': 75,
    'reactions': 87
}

# Ejemplo 2: Jugador promedio
jugador_2 = {
    'nombre': 'Jugador Promedio',
    'overall': 72,
    'potential': 78,
    'movement_reactions': 70,
    'release_clause_eur': 15000000,
    'wage_eur': 35000,
    'age': 22,
    'composure': 68,
    'reactions': 72
}

# Ejemplo 3: Talento sin desarrollar
jugador_3 = {
    'nombre': 'Talento Bruto',
    'overall': 68,
    'potential': 85,
    'movement_reactions': 75,
    'release_clause_eur': 8000000,
    'wage_eur': 20000,
    'age': 18,
    'composure': 60,
    'reactions': 74
}

# Ejemplo 4: Jugador cercano al tope
jugador_4 = {
    'nombre': 'Estrella Consolidada',
    'overall': 88,
    'potential': 90,
    'movement_reactions': 89,
    'release_clause_eur': 120000000,
    'wage_eur': 250000,
    'age': 23,
    'composure': 85,
    'reactions': 90
}

jugadores_prueba = [jugador_1, jugador_2, jugador_3, jugador_4]

# ============================================================================
# 4. HACER PREDICCIONES
# ============================================================================
print("\n[4] Realizando predicciones...")
print("="*80)

resultados = []

for jugador in jugadores_prueba:
    # Extraer caracteristicas en el orden correcto
    X_nuevo = np.array([[ 
        jugador['overall'],
        jugador['potential'],
        jugador['movement_reactions'],
        jugador['release_clause_eur'],
        jugador['wage_eur'],
        jugador['age']
    ]])
    # Escalar las caracteristicas
    X_nuevo_scaled = scaler.transform(X_nuevo)
    
    # Predecir
    valor_predicho = modelo.predict(X_nuevo_scaled, verbose=0)[0][0]
    
    # Guardar resultados
    resultado = {
        'Nombre': jugador['nombre'],
        'Overall': jugador['overall'],
        'Potential': jugador['potential'],
        'Age': jugador['age'],
        'Valor Predicho': valor_predicho
    }
    resultados.append(resultado)
    
    # Imprimir resultado individual
    print(f"\n{'='*80}")
    print(f"JUGADOR: {jugador['nombre']}")
    print(f"{'='*80}")
    print(f"Overall:           {jugador['overall']}")
    print(f"Potential:         {jugador['potential']}")
    print(f"Age:               {jugador['age']}")
    print(f"Movement:          {jugador['movement_reactions']}")
    print(f"Composure:         {jugador['composure']}")
    print(f"Reactions:         {jugador['reactions']}")
    print(f"Wage:              EUR {jugador['wage_eur']:,}")
    print(f"Release Clause:    EUR {jugador['release_clause_eur']:,}")
    print(f"\n>>> VALOR ESTIMADO: EUR {valor_predicho:,.0f} <<<")

# ============================================================================
# 5. RESUMEN COMPARATIVO
# ============================================================================
print("\n\n" + "="*80)
print("RESUMEN COMPARATIVO DE VALORACIONES")
print("="*80)

df_resultados = pd.DataFrame(resultados)
df_resultados['Valor Predicho'] = df_resultados['Valor Predicho'].apply(lambda x: f'EUR {x:,.0f}')

print("\n" + df_resultados.to_string(index=False))

# ============================================================================
# 6. MODO INTERACTIVO (OPCIONAL)
# ============================================================================
print("\n\n" + "="*80)
print("MODO INTERACTIVO - Prueba tu propio jugador")
print("="*80)

respuesta = input("\nDeseas valorar un jugador personalizado? (s/n): ")

if respuesta.lower() == 's':
    print("\nIngresa los atributos del jugador:")
    print("(Presiona Enter para valores por defecto)")
    
    try:
        overall = int(input("Overall (60-90) [75]: ") or 75)
        potential = int(input("Potential (65-95) [82]: ") or 82)
        age = int(input("Age (18-24) [21]: ") or 21)
        movement = int(input("Movement Reactions (60-90) [75]: ") or 75)
        composure = int(input("Composure (55-85) [70]: ") or 70)
        reactions = int(input("Reactions (60-90) [76]: ") or 76)
        wage = int(input("Wage EUR (5000-300000) [50000]: ") or 50000)
        release_clause = int(input("Release Clause EUR (100000-150000000) [10000000]: ") or 10000000)
        
        # Crear array con los datos
        X_custom = np.array([[
            overall, potential, movement, release_clause,
            wage, age, composure, reactions
        ]])
        
        # Escalar y predecir
        X_custom_scaled = scaler.transform(X_custom)
        valor_custom = modelo.predict(X_custom_scaled, verbose=0)[0][0]
        
        print("\n" + "="*80)
        print("RESULTADO DE TU JUGADOR PERSONALIZADO")
        print("="*80)
        print(f"\n>>> VALOR ESTIMADO: EUR {valor_custom:,.0f} <<<\n")
        
        # Comparacion con rangos
        if valor_custom < 5000000:
            categoria = "Jugador de Desarrollo"
        elif valor_custom < 20000000:
            categoria = "Promesa Emergente"
        elif valor_custom < 50000000:
            categoria = "Jugador Consolidado"
        elif valor_custom < 100000000:
            categoria = "Estrella de Elite"
        else:
            categoria = "Super Estrella Mundial"
        
        print(f"Categoria: {categoria}")
        
    except ValueError:
        print("\n[ERROR] Entrada invalida. Usa numeros enteros.")
    except Exception as e:
        print(f"\n[ERROR] {e}")

print("\n" + "="*80)
print("SISTEMA DE PRUEBA FINALIZADO")
print("="*80)
print("\nPara usar este modelo en produccion:")
print("1. Guarda el scaler con: import pickle; pickle.dump(scaler, open('scaler.pkl', 'wb'))")
print("2. Carga modelo y scaler en tu aplicacion")
print("3. Escala nuevos datos y predice valores")
print("="*80)