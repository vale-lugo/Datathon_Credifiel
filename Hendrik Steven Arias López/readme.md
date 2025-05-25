
# 📌 Proyecto de Optimización Multiobjetivo con Deep Embedding Clustering (DEC)

## 📋 Descripción del Proyecto

Este proyecto aborda un problema de optimización multiobjetivo orientado a maximizar la cobranza de una empresa y, simultáneamente, minimizar el costo por transacción. Se aplican técnicas avanzadas de segmentación y optimización utilizando **Deep Embedding Clustering (DEC)** y modelos predictivos basados en Gradient Boosting.

---

## 🎯 Objetivos Principales

* Realizar una limpieza exhaustiva y preprocesamiento adecuado de los datos.
* Aplicar Deep Embedding Clustering (DEC) para identificar comportamientos significativos mediante clusters bien definidos.
* Ejecutar una optimización multiobjetivo sobre los clusters generados para obtener soluciones óptimas.
* Generar recomendaciones personalizadas o generales, según las necesidades y recursos de la empresa.

---

## 🛠️ Metodología

El proceso del proyecto se divide en varias fases claramente definidas:

### 1. **Limpieza y Preprocesamiento**

* Tratamiento de valores nulos y conversión de variables categóricas y booleanas.
* Aplicación de medidas estadísticas (media, mediana, moda) para agrupar datos por persona inicialmente.

### 2. **Segmentación con DEC**

* Generación de embeddings representativos usando redes neuronales profundas.
* Aplicación de K-means sobre estos embeddings para obtener clusters óptimos.
* Selección final: 7 dimensiones en el embedding y 5 clusters como configuración óptima.

### 3. **Optimización Multiobjetivo**

* Entrenamiento de modelos predictivos independientes (XGBoost) para maximizar el monto cobrado y minimizar el costo por transacción.
* Uso de Optuna para explorar configuraciones óptimas y determinar el frente de Pareto.
* Análisis del frente de Pareto para elegir decisiones estratégicas en función de objetivos específicos.

### 4. **Automatización y Recomendaciones**

* Conexión inversa desde clusters hasta la base de datos original para automatizar la toma de decisiones.
* Generación de dos resultados principales:

  * **`clusters_unidos_con_recomendaciones.csv`**: Recomendaciones por cluster.
  * **`recomendaciones_Gradient_Boosting.csv`**: Recomendaciones personalizadas por individuo.


## 📈 Conclusiones

Este proyecto proporciona herramientas potentes para automatizar decisiones basadas en datos, optimizando de manera efectiva tanto el rendimiento financiero como el control de costos de transacción. Las metodologías utilizadas aseguran precisión, flexibilidad estratégica y facilidad de adaptación a contextos cambiantes.
