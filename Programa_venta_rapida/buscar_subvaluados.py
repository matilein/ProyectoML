from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

# --- Archivos ---
CSV_DATOS = "intermedio_sin_outliers.csv"
MODELO_PATH = r"C:\Users\andyd\Udesa\Machine Learning\ProyectoML\Programa_venta_rapida\modelo_keras.keras"
SCALER_X_PATH = "Programa_venta_rapida/scaler_X.pkl"
SCALER_Y_PATH = "Programa_venta_rapida/scaler_y.pkl"

# --- 1. Entrada del usuario ---
marca_input = input("🔤 Ingresá la marca (por ejemplo: Ford): ").strip()
modelo_input = input("🚗 Ingresá el modelo (por ejemplo: Ecosport): ").strip()
umbral_input = input("📉 Ingresá el porcentaje mínimo de subvaluación (ej: 10): ").strip()

try:
    UMBRAL = float(umbral_input)
except ValueError:
    print("❌ Porcentaje inválido. Debe ser un número.")
    exit()

col_marca = f"Marca_{marca_input}"
col_modelo = f"Modelo_{modelo_input}"

# --- 2. Cargar modelo y escaladores ---
modelo = load_model(MODELO_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

# --- 3. Cargar datos y verificar columnas ---
df = pd.read_csv(CSV_DATOS)

if col_marca not in df.columns:
    print(f"❌ La marca '{marca_input}' no existe en el dataset.")
    exit()
if col_modelo not in df.columns:
    print(f"❌ El modelo '{modelo_input}' no existe en el dataset.")
    exit()

# --- 4. Filtrar autos por marca y modelo ---
filtro = (df[col_marca] == 1) & (df[col_modelo] == 1)
df_filtrado = df[filtro].copy()

if df_filtrado.empty:
    print(f"⚠️ No se encontraron autos de marca '{marca_input}' y modelo '{modelo_input}'.")
    exit()

# --- 5. Predecir precios ---
X = scaler_X.transform(df_filtrado.drop(columns=["Precio_usd"]))
y_pred_scaled = modelo.predict(X).flatten()
y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# --- 6. Calcular subvaluación ---
df_filtrado["precio_estimado"] = y_pred_real
df_filtrado["subvaluacion_%"] = 100 * (df_filtrado["precio_estimado"] - df_filtrado["Precio_usd"]) / df_filtrado["precio_estimado"]

# --- 7. Filtrar subvaluados ---
df_subvaluados = df_filtrado[df_filtrado["subvaluacion_%"] >= UMBRAL]

# --- 8. Mostrar resultados ---
print(f"\n🔍 Autos '{marca_input} {modelo_input}' subvaluados más del {UMBRAL}%:")

if df_subvaluados.empty:
    print("❌ No se encontraron autos subvaluados con ese criterio.")
else:
    print(df_subvaluados[["Precio_usd", "precio_estimado", "subvaluacion_%"]].round(2))
    mejor = df_subvaluados.loc[df_subvaluados["subvaluacion_%"].idxmax()]
    print("\n🏆 Auto más subvaluado:")
    print(mejor[["Precio_usd", "precio_estimado", "subvaluacion_%"]].round(2))

# --- 9. Top 5 autos subvaluados del dataset completo ---
X_total = scaler_X.transform(df.drop(columns=["Precio_usd"]))
y_total_scaled = modelo.predict(X_total).flatten()
y_total_real = scaler_y.inverse_transform(y_total_scaled.reshape(-1, 1)).flatten()

df["precio_estimado"] = y_total_real
df["subvaluacion_%"] = 100 * (df["precio_estimado"] - df["Precio_usd"]) / df["precio_estimado"]

# Extraer marca y modelo
cols_marca = [c for c in df.columns if c.startswith("Marca_")]
cols_modelo = [c for c in df.columns if c.startswith("Modelo_")]

def extraer_nombre(col_prefix, row):
    for c in col_prefix:
        if row.get(c, 0) == 1:
            return c.split("_", 1)[1]
    return "Desconocido"

df["Marca"] = df.apply(lambda row: extraer_nombre(cols_marca, row), axis=1)
df["Modelo"] = df.apply(lambda row: extraer_nombre(cols_modelo, row), axis=1)

top5 = df.sort_values("subvaluacion_%", ascending=False).head(5)

print("\n🌍 Top 5 autos más subvaluados de TODO el dataset:")
print(top5[["Marca", "Modelo", "Precio_usd", "precio_estimado", "subvaluacion_%"]].round(2))
