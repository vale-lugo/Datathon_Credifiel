# Proyecto de Optimización de Cobranza Domiciliada para Credifiel

## Resumen del Proyecto

Este proyecto aborda el desafío de optimizar el proceso de cobranza domiciliada en Credifiel, una entidad financiera especializada en créditos con descuento por nómina y domiciliación bancaria. El objetivo principal es maximizar la recuperación de cartera y minimizar los costos por comisiones bancarias, transformando la estrategia actual de envíos masivos a un enfoque selectivo y optimizado.

El análisis se centra en utilizar los datos históricos de intentos de cobro, enriquecidos con catálogos de información bancaria y de servicios, para construir un modelo predictivo de éxito de cobro. Este modelo, junto con una lógica de decisión basada en costos y probabilidad de éxito, permite seleccionar el canal de cobro (banco adquirente y servicio específico) óptimo para cada intento.

## Componentes Clave del Proyecto

### 1. Consolidación y Limpieza Exhaustiva de Datos

Una parte fundamental del proyecto fue la integración y preparación de los datos. Esto incluyó:

*   **Carga de Múltiples Fuentes:** Se cargó el dataset principal de intentos de cobro y múltiples archivos de catálogo (`CatBanco.csv`, `CatEmisora(in).csv`, `CatRespuestaBancos.csv`, `ListaCobro.csv`, `ListaCobroEmisora.csv`).
*   **Unión (Merge) Estratégica:** Los catálogos se unieron al dataset principal para enriquecer cada intento de cobro con información detallada, como el nombre del banco del cliente, la descripción de la respuesta del banco, detalles de la lista de cobro, y el banco adquirente y servicio utilizado por Credifiel.
*   **Conversión de Tipos de Datos:** Las columnas de fecha se convirtieron al formato `datetime` para permitir cálculos basados en tiempo. Otras columnas se ajustaron a tipos numéricos o categóricos según correspondía.
*   **Manejo de Valores Nulos:** Se analizaron y gestionaron los valores nulos. Por ejemplo, los `NaN` en `fechaCobroBanco` se interpretaron correctamente como intentos no exitosos. Para otras columnas, se aplicaron estrategias de imputación o eliminación informadas.
*   **Ingeniería de Características (Feature Engineering):** Se crearon nuevas características para mejorar el poder predictivo del modelo, tales como:
    *   `intento_exitoso`: Variable booleana indicando si el cobro fue exitoso.
    *   `nombre_banco_adquirente`: Nombre del banco que Credifiel utilizó para el intento.
    *   `exitos_previos_credito`: Número de pagos exitosos anteriores para un mismo crédito.
    *   `intento_para_pago_num`: Identifica si un intento corresponde al 1er, 2do, 3er, etc., pago *esperado* del crédito.
    *   Características temporales: `dia_semana_envio_cobro`, `mes_envio_cobro`, `dia_mes_envio_cobro`.
    *   `dias_desde_apertura_credito`: Antigüedad del crédito al momento del intento.
    *   `mismo_banco_cliente_adquirente`: Si el banco del cliente y el banco adquirente son el mismo.
    *   `intentos_previos_totales_credito`: Número total de intentos (exitosos o no) para ese crédito.
*   **Estandarización de Información de Servicios Bancarios:**
    *   Se digitalizaron y procesaron tablas con las características de los servicios ofrecidos por diferentes bancos adquirentes (Banamex, BBVA, Santander, Banorte), incluyendo costos, horarios, tiempos de respuesta y capacidades especiales (monitoreo, parcialidad).
    *   Se desarrolló una lógica robusta (`normalizar_nombre_servicio_v2`) para mapear los diversos nombres de `nombre_servicio_emisora` (provenientes de los datos históricos) a un conjunto estandarizado de `servicio_key` definido en las tablas de características de servicios. Esto fue crucial para poder unir consistentemente los detalles del servicio (costos, tiempos, etc.) a cada intento de cobro en el dataset principal (`df_completo_final`).
    *   Las filas con servicios no mapeables o información faltante de origen fueron cuidadosamente gestionadas.

El resultado de esta fase es un DataFrame consolidado (`df_completo_final`) que contiene una visión integral de cada intento de cobro, listo para el modelado.

### 2. Modelo Predictivo de Éxito de Cobro (Regresión Logística)

Para predecir la probabilidad de que un intento de cobro sea exitoso, se implementó un modelo de Regresión Logística utilizando `scikit-learn`.

*   **Selección de Características:** Se eligió un conjunto de características numéricas y categóricas relevantes del `df_completo_final`.
*   **Preprocesamiento dentro de un Pipeline:**
    *   **Características Numéricas:** Se imputaron valores faltantes con la mediana y luego se escalaron usando `StandardScaler`.
    *   **Características Categóricas:** Se imputaron valores faltantes con la etiqueta "Desconocido" y luego se transformaron usando `OneHotEncoder`.
    *   `ColumnTransformer` se utilizó para aplicar estas transformaciones de manera selectiva a los tipos de columnas correspondientes.
*   **División de Datos:** El dataset se dividió en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo de manera imparcial.
*   **Entrenamiento del Modelo:** Se entrenó un modelo de `LogisticRegression`. Se incluyó `class_weight='balanced'` para mitigar el desbalance inherente en los datos (mayoría de intentos no exitosos). Se ajustó el solver y `max_iter` para asegurar la convergencia.
*   **Evaluación del Modelo:** El rendimiento se evaluó mediante métricas como Accuracy, ROC AUC Score, y un informe de clasificación detallado (precisión, recall, F1-score). Se visualizó la curva ROC.

### 3. Pipeline de Estrategia de Decisión para Optimización

Con el modelo predictivo, se desarrolló un framework conceptual para una estrategia de decisión que busca optimizar la selección del servicio bancario para cada intento de cobro:

*   **Cálculo de Costo del Intento:** Se definió una función (`calcular_costo_intento`) que estima el costo de un intento de cobro para un servicio bancario particular, considerando su estructura de comisiones (por envío, por éxito, por devolución) y la probabilidad de éxito predicha por el modelo.
*   **Selección del Mejor Servicio (`seleccionar_mejor_servicio`):**
    1.  Para un intento de cobro dado (con sus características de crédito y cliente), la función itera sobre todos los servicios bancarios disponibles para Credifiel (con sus respectivos costos, tiempos de respuesta, etc., extraídos de `servicios_bancarios_df`).
    2.  Para cada combinación (intento + servicio candidato), se construye el conjunto de características completo.
    3.  El modelo de Regresión Logística entrenado predice la `prob_exito`.
    4.  Se calcula el `costo_intento` para ese servicio.
    5.  Se calcula el **Valor Esperado Neto (VEN)**: `VEN = (prob_exito * montoACobrar) - costo_intento`.
    6.  La estrategia elige el servicio bancario que maximiza el VEN. Si ningún servicio ofrece un VEN positivo, la estrategia podría sugerir no realizar el intento por ese canal o en ese momento.
*   **Simulación:** Se incluyó un ejemplo de cómo simular esta lógica de decisión sobre una muestra de intentos históricos para comparar los resultados de la estrategia optimizada con los resultados reales.

## Cómo Utilizar el Código

1.  **Entorno:** Asegúrate de tener Python y las librerías listadas en el notebook instaladas (principalmente `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
2.  **Archivos de Datos:** Coloca todos los archivos `.csv` requeridos en el mismo directorio que el notebook `optimizacion_cobranza.ipynb`.
3.  **Ejecución del Notebook:** Abre y ejecuta las celdas del notebook en orden.
    *   Las primeras celdas se encargan de la carga, limpieza, unión de datos y la ingeniería de características.
    *   Luego, se entrena y evalúa el modelo de Regresión Logística.
    *   Finalmente, se presenta el framework para la lógica de decisión y un ejemplo de su aplicación.

## Próximos Pasos y Mejoras Potenciales

*   Experimentar con modelos predictivos más avanzados (Random Forest, Gradient Boosting, XGBoost, LightGBM).
*   Realizar un ajuste de hiperparámetros exhaustivo para el modelo seleccionado.
*   Incorporar restricciones operativas más detalladas en la lógica de decisión (e.g., horarios límite de envío de forma dinámica).
*   Desarrollar una estrategia para manejar los pagos críticos (2do y 5to pago) de manera más específica si el modelo general no los prioriza suficientemente.
*   Implementar un sistema de monitoreo y reentrenamiento periódico del modelo y la estrategia.

