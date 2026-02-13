import glob
import time
import numpy as np
import pandas as pd
import psutil

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# =========================
# CONFIG
# =========================
DATA_PATH = "data/*.parquet"     # carpeta donde has puesto los .parquet
N_JOBS = 8                      # i7-4790 = 8 hilos
TREES = 1600                    # sube a 2500-3000 para más estrés
MAX_DEPTH = 10                  # sube a 12 para más estrés

# Reduce RAM leyendo solo columnas necesarias (muy recomendado con 32GB)
USE_COLUMN_PRUNING = True

# Columnas mínimas FH-VHV
NEEDED_COLS = [
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
]


# =========================
# HELPERS
# =========================
def sysline(tag=""):
    vm = psutil.virtual_memory()
    print(f"{tag} CPU {psutil.cpu_percent():.0f}% | RAM {vm.percent:.0f}% ({vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB)")


def read_parquet_file(path: str) -> pd.DataFrame:
    if USE_COLUMN_PRUNING:
        return pd.read_parquet(path, columns=NEEDED_COLS)
    return pd.read_parquet(path)


# =========================
# 1) LOAD
# =========================
files = sorted(glob.glob(DATA_PATH))
if not files:
    raise SystemExit(f"No encuentro ficheros en: {DATA_PATH}")

t0 = time.time()
dfs = []
for f in files:
    print("Leyendo", f)
    dfs.append(read_parquet_file(f))

df = pd.concat(dfs, ignore_index=True)

print("\nShape:", df.shape)
print("Columnas:", list(df.columns))
print(f"Tiempo carga: {time.time() - t0:.2f}s")
sysline("Post-carga")
print("-" * 70)


# =========================
# 2) FEATURE ENGINEERING
# =========================
t0 = time.time()

# Datetimes
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

# Target: duración en segundos
df["trip_duration"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()

# Features temporales (downcast para RAM)
df["hour"] = df["pickup_datetime"].dt.hour.astype("int16")
df["day"] = df["pickup_datetime"].dt.day.astype("int16")
df["month"] = df["pickup_datetime"].dt.month.astype("int16")

# Trip miles a float32 (RAM)
df["trip_miles_num"] = pd.to_numeric(df["trip_miles"], errors="coerce").astype("float32")

# Location IDs a int32 (SIN category -> evita el bug)
df["PULocationID"] = pd.to_numeric(df["PULocationID"], errors="coerce").astype("Int32")
df["DOLocationID"] = pd.to_numeric(df["DOLocationID"], errors="coerce").astype("Int32")

# Limpieza NaNs
df = df.dropna(subset=[
    "trip_duration", "trip_miles_num", "hour", "day", "month", "PULocationID", "DOLocationID"
])

# Pasamos Int32 (nullable) a int32 normal (XGBoost lo prefiere)
df["PULocationID"] = df["PULocationID"].astype("int32")
df["DOLocationID"] = df["DOLocationID"].astype("int32")

# Filtros de outliers (ajusta si quieres)
df = df[(df["trip_duration"] > 0) & (df["trip_duration"] < 6 * 3600)]
df = df[(df["trip_miles_num"] >= 0) & (df["trip_miles_num"] < 200)]

print(f"Tiempo features: {time.time() - t0:.2f}s")
sysline("Post-features")
print("-" * 70)


# =========================
# 3) TRAIN (CPU STRESS)
# =========================
features = ["trip_miles_num", "hour", "day", "month", "PULocationID", "DOLocationID"]
target = "trip_duration"

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target].astype("float32")  # reduce RAM y va sobrado

# Split (sin stratify, es regresión)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo XGBoost (sin categorías, robusto)
model = XGBRegressor(
    n_estimators=TREES,
    max_depth=MAX_DEPTH,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    n_jobs=N_JOBS,
    tree_method="hist",
    random_state=42,
)

t0 = time.time()
model.fit(X_train, y_train)
train_time = time.time() - t0

print(f"Tiempo entrenamiento: {train_time:.2f}s")
sysline("Post-train")
print("-" * 70)


# =========================
# 4) EVAL
# =========================
preds = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
print(f"RMSE (trip_duration): {rmse:.2f}")

print("\nBENCHMARK COMPLETADO 🔥")
