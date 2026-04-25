import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================
# Configuración general
# =========================
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 5)

# =========================
# Rutas del proyecto
# =========================
DATA_PATH = Path("data/raw")
OUTPUT_PATH = Path("reports/figures")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =========================
# Carga de datos
# =========================
df_creditos = pd.read_csv(DATA_PATH / "datos_creditos.csv", sep=";")
df_tarjetas = pd.read_csv(DATA_PATH / "datos_tarjetas.csv", sep=";")

# =========================
# TARGET DISTRIBUTION
# =========================
plt.figure()
sns.countplot(x="falta_pago", data=df_creditos)
plt.title("Distribución de la variable objetivo (falta_pago)")
plt.xlabel("¿Presentó mora?")
plt.ylabel("Cantidad de clientes")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "target_distribution_falta_pago.png")
plt.close()

# Distribución porcentual (consola)
target_distribution = (
    df_creditos["falta_pago"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nDistribución porcentual de la variable objetivo (falta_pago):")
print(target_distribution)

# =========================
# VARIABLES CATEGÓRICAS - CRÉDITOS
# =========================
categorical_cols_creditos = (
    df_creditos
    .select_dtypes(include=["object"])
    .columns
    .drop("falta_pago")
)

for col in categorical_cols_creditos:
    plt.figure()
    order = df_creditos[col].value_counts().index
    sns.countplot(y=col, data=df_creditos, order=order)
    plt.title(f"Distribución de {col} (Créditos)")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / f"creditos_{col}.png")
    plt.close()

# =========================
# VARIABLES CATEGÓRICAS - TARJETAS
# =========================
categorical_cols_tarjetas = (
    df_tarjetas
    .select_dtypes(include=["object"])
    .columns
)

for col in categorical_cols_tarjetas:
    plt.figure()
    order = df_tarjetas[col].value_counts().index
    sns.countplot(y=col, data=df_tarjetas, order=order)
    plt.title(f"Distribución de {col} (Tarjetas)")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / f"tarjetas_{col}.png")
    plt.close()

# =========================
# CORRELACIONES - CRÉDITOS
# =========================
num_creditos = df_creditos.select_dtypes(include=["int64", "float64"])
corr_creditos = num_creditos.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_creditos, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlaciones - Créditos")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "correlaciones_creditos.png")
plt.close()

# =========================
# CORRELACIONES - TARJETAS
# =========================
num_tarjetas = df_tarjetas.select_dtypes(include=["int64", "float64"])
corr_tarjetas = num_tarjetas.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_tarjetas, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlaciones - Tarjetas")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "correlaciones_tarjetas.png")
plt.close()

print("\n✅ Visualización de datos finalizada correctamente.")
