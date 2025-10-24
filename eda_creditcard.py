# eda_creditcard.py
import pandas as pd
from ydata_profiling import ProfileReport
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

# === a. Cargar dataset y generar Profile Report ===
df = pd.read_csv("creditcard.csv")
print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())

# Generar reporte exploratorio automático
profile = ProfileReport(df, title="EDA Credit Card Dataset", explorative=True)
profile.to_file("creditcard_eda_report.html")
print("\nReporte EDA generado: creditcard_eda_report.html")

# === b. Imputar o eliminar columnas con más del 10% de valores perdidos ===
missing = df.isnull().mean() * 100
cols_to_drop = missing[missing > 10].index
if len(cols_to_drop) > 0:
    print(f"\nColumnas con >10% de valores perdidos eliminadas: {list(cols_to_drop)}")
    df.drop(columns=cols_to_drop, inplace=True)
else:
    print("\nNo hay columnas con más del 10% de valores perdidos.")

# Rellenar los faltantes restantes (si existen) con la media (numéricos) o moda (categóricos)
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# === c. Eliminar outliers con winsorize ===
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = winsorize(df[col], limits=[0.05, 0.05])  # recorta 5% inferior y superior
print("\nOutliers suavizados con winsorize.")

# === d. Convertir variables categóricas a numéricas ===
cat_cols = df.select_dtypes(include=["object"]).columns
if len(cat_cols) > 0:
    print("\nColumnas categóricas convertidas a numéricas:", list(cat_cols))
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
else:
    print("\nNo se detectaron columnas categóricas.")

# === e. Normalizar el dataset ===
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("\nDataset normalizado (valores entre 0 y 1).")

# Guardar resultado final
df_scaled.to_csv("creditcard_clean.csv", index=False)
print("\nArchivo final guardado como: creditcard_clean.csv")