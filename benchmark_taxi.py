import pandas as pd
import numpy as np
import glob
import time
import psutil
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------
# Configuración
# -----------------------------
DATA_PATH = "data/*.csv"  # Cambia a tu carpeta
N_JOBS = 8  # Todos los hilos del i7
TREES = 1200

def log_system():
    print(f"CPU usage: {psutil.cpu_percent()}%")
    print(f"RAM usage: {psutil.virtual_memory().percent}%")
    print("-" * 40)

# -----------------------------
# 1️⃣ Carga masiva
# -----------------------------
start = time.time()

files = glob.glob(DATA_PATH)
df_list = []

for f in files:
    print(f"Leyendo {f}")
    df = pd.read_csv(f)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

print(f"\nFilas totales: {len(df):,}")
print(f"Tiempo carga: {time.time() - start:.2f}s")
log_system()

# -----------------------------
# 2️⃣ Feature engineering pesado
# -----------------------------
start = time.time()

df["pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month

df["trip_duration"] = (
    pd.to_datetime(df["tpep_dropoff_datetime"]) -
    df["pickup_datetime"]
).dt.total_seconds()

df = df[df["trip_duration"] > 0]

print(f"Tiempo features: {time.time() - start:.2f}s")
log_system()

# -----------------------------
# 3️⃣ Entrenamiento pesado
# -----------------------------
features = ["trip_distance", "hour", "day", "month", "passenger_count"]
df = df.dropna(subset=features + ["trip_duration"])

X = df[features]
y = df["trip_duration"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=TREES,
    max_depth=10,
    learning_rate=0.05,
    n_jobs=N_JOBS,
    tree_method="hist"
)

start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start

print(f"\nTiempo entrenamiento: {train_time:.2f}s")
log_system()

# -----------------------------
# 4️⃣ Evaluación
# -----------------------------
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"\nRMSE: {rmse:.2f}")
print("BENCHMARK COMPLETADO 🔥")
