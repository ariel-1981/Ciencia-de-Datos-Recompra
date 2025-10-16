# 📝 Informe de Análisis Predictivo – Recompra de Clientes
### 1. Objetivo del Proyecto
El objetivo de este análisis es determinar qué factores influyen en la **recompra** de los clientes tras **recibir una promoción**, evaluando específicamente:
* Si recibir una promoción aumenta la probabilidad de recompra.
* Si el monto de la promoción tiene relación con la recompra.
* Si variables como edad e ingreso también influyen en la decisión de recompra.
---
### 2. Carga y Preparación de Datos
Se cargó un **archivo Excel** con información de clientes y promociones.
* Se verificó la presencia de columnas clave: *Cliente_ID, Recompra, Recibio_Promo, Monto_Promo, Edad, Ingreso.*
* Se realizó una codificación de variables categóricas (Género, Recompra, Recibio_Promo).
* Se analizaron valores nulos y se manejaron mediante imputación básica (fillna(0)).
---
### 3. Análisis Exploratorio
* **¿Recibir una promoción influye en la recompra?**
Sí. El análisis de contingencia (crosstab) mostró que los clientes que recibieron una promoción tenían una mayor tasa de recompra comparado con quienes no la recibieron.
* **¿Importa el monto de la promoción?**
Sí. El promedio del Monto_Promo fue notablemente mayor entre quienes realizaron una recompra, lo que sugiere una relación positiva entre el monto promocionado y la recompra.
* **¿Influyen la edad y el ingreso?**
Sí. Los clientes que recompraron presentaron en promedio:
* Mayor edad
* Mayor ingreso mensual
**Esto indica que clientes más maduros y con mejor capacidad adquisitiva responden mejor a las promociones.**
---
### 4. Modelado Predictivo
Se utilizó un **árbol de decisión (DecisionTreeClassifier)** para predecir la variable Recompra.
Resultados del modelo:
* Se dividieron los datos en entrenamiento y prueba (80/20).
* Se obtuvo la matriz de confusión y el reporte de clasificación para evaluar desempeño.
*El modelo logró distinguir correctamente muchos casos, aunque se recomienda evaluar con métricas específicas como precisión y recall para cada clase.
### Importancia de variables (según el árbol de decisión):
* Variable	Importancia
* Monto_Promo	Alta
* Recibio_Promo	Alta
* Ingreso	Media
* Edad	Baja/Media
**Esto refuerza los hallazgos del análisis exploratorio: el monto y la recepción de la promoción son factores clave, seguidos por características socioeconómicas del cliente.**
---
### 5. Conclusiones
* Recibir una promoción sí influye positivamente en la recompra.
* El monto de la promoción también importa: a mayor monto, mayor tasa de recompra.
* Edad e ingreso están relacionados con la recompra, posiblemente por una mayor capacidad de gasto y madurez en decisiones de consumo.
* El modelo predictivo respalda estos resultados, asignando alta importancia a estas variables clave.
### 6. Recomendaciones
* Focalizar campañas promocionales en clientes con mayor ingreso y edad.
* Evaluar el monto mínimo de promoción efectivo para estimular recompras.
