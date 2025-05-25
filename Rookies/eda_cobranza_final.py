# eda_cobranza.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("Directorio actual:", os.getcwd())

# Configuraciones iniciales
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Base path: la carpeta donde está el script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas absolutas basadas en la carpeta del script
catalog_dir = os.path.join(BASE_DIR, "ExtraccionDomiVersionFinal")
data_dir = os.path.join(BASE_DIR, "ExtraccionDomiVersionFinal")

# Confirmar rutas
print("Catalog dir:", catalog_dir)
print("Data dir:", data_dir)

# Carga de catálogos
cat_banks = pd.read_csv(os.path.join(catalog_dir, "CatBanco.csv"))
cat_respuestas = pd.read_csv(os.path.join(catalog_dir, "CatRespuestaBancos.csv"))
cat_emisoras = pd.read_csv(os.path.join(catalog_dir, "CatEmisora.csv"))
lista_cobro = pd.read_csv(os.path.join(catalog_dir, "ListaCobro.csv"))
lista_cobro_emisora = pd.read_csv(os.path.join(catalog_dir, "ListaCobroEmisora.csv"))

# Normalizar nombres de columnas (convertir a minúsculas para evitar problemas de mayúsculas)
cat_banks.columns = cat_banks.columns.str.lower()
cat_respuestas.columns = cat_respuestas.columns.str.lower()
cat_emisoras.columns = cat_emisoras.columns.str.lower()
lista_cobro.columns = lista_cobro.columns.str.lower()
lista_cobro_emisora.columns = lista_cobro_emisora.columns.str.lower()

# Convertir claves a string para evitar errores de merge
cat_banks["idbanco"] = cat_banks["idbanco"].astype(str)
cat_respuestas["idrespuestabanco"] = cat_respuestas["idrespuestabanco"].astype(str)
cat_emisoras["idemisora"] = cat_emisoras["idemisora"].astype(str)
lista_cobro["idlistacobro"] = lista_cobro["idlistacobro"].astype(str)
lista_cobro_emisora["idlistacobro"] = lista_cobro_emisora["idlistacobro"].astype(str)
lista_cobro_emisora["idemisora"] = lista_cobro_emisora["idemisora"].astype(str)

# Cargar datos de cobro de 2022 a 2025
cobro_data = []
for year in range(2022, 2026):
    cobro_file = f"ListaCobroDetalle{year}.csv"
    df_cobro = pd.read_csv(os.path.join(data_dir, cobro_file), low_memory=False)
    df_cobro.columns = df_cobro.columns.str.lower()  # Normalizar columnas
    df_cobro["año"] = year
    df_cobro["idbanco"] = df_cobro["idbanco"].astype(str)
    df_cobro["idrespuestabanco"] = df_cobro["idrespuestabanco"].astype(str)
    df_cobro["idlistacobro"] = df_cobro["idlistacobro"].astype(str)
    cobro_data.append(df_cobro)

df_cobro_all = pd.concat(cobro_data, ignore_index=True)

# Depuración: Mostrar columnas antes del merge
print("Columnas de df_cobro_all antes del merge:", df_cobro_all.columns.tolist())
print("Columnas de cat_banks:", cat_banks.columns.tolist())

# Unir con catálogos (usando nombres normalizados)
df_cobro_all = df_cobro_all.merge(cat_banks, on="idbanco", how="left", suffixes=('_orig', '_banks'))
df_cobro_all = df_cobro_all.merge(cat_respuestas, on="idrespuestabanco", how="left")
df_cobro_all = df_cobro_all.merge(lista_cobro, on="idlistacobro", how="left")
df_cobro_all = df_cobro_all.merge(lista_cobro_emisora, on="idlistacobro", how="left")
df_cobro_all = df_cobro_all.merge(cat_emisoras, on="idemisora", how="left")

# Depuración: Mostrar columnas después del merge
print("Columnas de df_cobro_all después del merge:", df_cobro_all.columns.tolist())

# Renombrar columna 'nombre_banks' a 'nombre' para usar en groupby
df_cobro_all = df_cobro_all.rename(columns={'nombre_x': 'nombre'})

# Guardar resumen anual por banco
summary = df_cobro_all.groupby(["año", "nombre"]).agg(
    total_cobrado=pd.NamedAgg(column="montocobrado", aggfunc="sum"),
    total_intentos=pd.NamedAgg(column="idlistacobro", aggfunc="count")
).reset_index()
summary.to_csv(os.path.join(BASE_DIR, "resumen_cobranza_por_bancoFINAL.csv"), index=False)

# --- Visualizaciones ---
df_cobro_all["fechacobrobanco"] = pd.to_datetime(df_cobro_all["fechacobrobanco"], errors="coerce", dayfirst=True)
df_cobro_all["mes"] = df_cobro_all["fechacobrobanco"].dt.to_period("M").astype(str)

# Serie de tiempo
serie = df_cobro_all.groupby("mes")["montocobrado"].sum().reset_index()
sns.lineplot(data=serie, x="mes", y="montocobrado")
plt.xticks(rotation=45)
plt.title("Cobranza mensual total")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "cobranza_mensualFINAL.png"))
plt.clf()

# Top bancos
top_bancos = summary.groupby("nombre")["total_cobrado"].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_bancos.values, y=top_bancos.index)
plt.title("Top 10 bancos por cobranza total")
plt.xlabel("Monto total cobrado")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "top_bancosFINAL.png"))
plt.clf()

# Motivos de rechazo
motivos = df_cobro_all["descripcion"].value_counts().head(10)
sns.heatmap(motivos.to_frame().T, annot=True, fmt="d", cmap="Reds")
plt.title("Motivos de rechazo más frecuentes")
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "motivos_rechazoFINAL.png"))
plt.clf()