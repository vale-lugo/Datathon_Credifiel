import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Base path: la carpeta donde está el script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cargar datos de ejecución ya preparados
df_exec_all = pd.read_csv(os.path.join(BASE_DIR, "resumen_cobranza_por_bancoFINAL.csv"))

# Depuración: Mostrar los nombres de bancos disponibles
print("Bancos disponibles en los datos:", df_exec_all["nombre"].unique())

# Escenario base (A)
def escenario_a(df):
    """Simulación del escenario histórico (sin cambios)."""
    df_a = df.copy()
    print("Datos del Escenario A antes de resumen:", df_a[["año", "nombre", "total_cobrado"]].head())
    return df_a

# Escenario modificado (B)
def escenario_b(df, bancos_mejorados, incremento_pct=0.2):
    """Mejora la cobranza de ciertos bancos en un porcentaje."""
    df_b = df.copy()
    mejora_mask = df_b["nombre"].isin(bancos_mejorados)
    print("Filas afectadas por la mejora (Escenario B):", mejora_mask.sum())
    df_b.loc[mejora_mask, "total_cobrado"] *= (1 + incremento_pct)
    print("Datos del Escenario B después de la mejora:", df_b[["año", "nombre", "total_cobrado"]].head())
    return df_b

# Bancos que mejoran en el escenario B (ajustados para coincidir con los datos)
bancos_objetivo = ["BANORTE", "SANTANDER", "BBVA MEXICO"]  # Ajustados a mayúsculas; bancomext no está en los datos

# Aplicar escenarios
escenario_a_df = escenario_a(df_exec_all)
escenario_b_df = escenario_b(df_exec_all, bancos_objetivo)

# Comparar totales por año
def resumen_por_año(df, escenario_nombre):
    resumen = df.groupby("año")["total_cobrado"].sum().reset_index()
    resumen["escenario"] = escenario_nombre
    print(f"Resumen {escenario_nombre}:", resumen)
    return resumen

resumen_a = resumen_por_año(escenario_a_df, "Escenario A")
resumen_b = resumen_por_año(escenario_b_df, "Escenario B")

# Depuración: Verificar concatenación
print("Resumen comparado antes de graficar:", pd.concat([resumen_a, resumen_b]))

# Concatenar y graficar comparación
resumen_comparado = pd.concat([resumen_a, resumen_b])

# Graficar con diferentes estilos de línea para distinguir los escenarios
sns.lineplot(data=resumen_comparado, x="año", y="total_cobrado", hue="escenario", style="escenario", marker="o")
plt.title("Comparación de cobranza total por año entre escenarios A y B")
plt.ylabel("Total cobrado")
plt.xticks(resumen_comparado["año"].unique())
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "comparacion_escenarios.png"))