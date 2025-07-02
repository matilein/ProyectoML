from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from construir_modelo import construir_modelo
import joblib

# 1. Cargar datos
df = pd.read_csv("intermedio_sin_outliers.csv")
X = df.drop(columns=["Precio_usd"]).values
y = df["Precio_usd"].values

# 2. Normalizar
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 3. Separar train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Construir modelo
modelo = construir_modelo(input_dim=X_train.shape[1])

# 5. Entrenar
callback = EarlyStopping(patience=20, restore_best_weights=True)
modelo.fit(X_train, y_train,
           validation_split=0.2,
           epochs=200,
           batch_size=32,
           callbacks=[callback],
           verbose=1)

# 6. Guardar modelo y escaladores
modelo.save("Programa_venta_rapida/modelo_keras.keras")
joblib.dump(scaler_X, "Programa_venta_rapida/scaler_X.pkl")
joblib.dump(scaler_y, "Programa_venta_rapida/scaler_y.pkl")

print("✅ Modelo y escaladores guardados")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predecimos en test
y_pred_scaled = modelo.predict(X_test).flatten()

# Desescalamos
y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Métricas interpretables
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2 = r2_score(y_test_real, y_pred_real)

print(f"📊 MAE: {mae:.2f} USD")
print(f"📉 RMSE: {rmse:.2f} USD")
print(f"📈 R²: {r2:.4f}")
