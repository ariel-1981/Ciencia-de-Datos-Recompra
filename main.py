import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Estilo para gráficos
sns.set(style="whitegrid")

print("=" * 60)
print("INICIANDO CARGA DE DATOS")
print("=" * 60)

# 1. VERIFICAR Y CARGAR EL DATAFRAME
ruta_archivo = "Mini_Proyecto_Clientes_Promociones.xlsx"

if not os.path.exists(ruta_archivo):
    print(f"ERROR: El archivo no existe en la ruta: {ruta_archivo}")
    print("Buscando archivos Excel en la carpeta Downloads...")

    downloads_path = "C:/Users/47-01/Downloads/"
    if os.path.exists(downloads_path):
        archivos_excel = [f for f in os.listdir(downloads_path) if f.endswith(('.xlsx', '.xls'))]
        if archivos_excel:
            print("Archivos Excel encontrados:")
            for archivo in archivos_excel:
                print(f"   - {archivo}")
            ruta_archivo = os.path.join(downloads_path, archivos_excel[0])
            print(f"Usando archivo: {ruta_archivo}")
        else:
            print("No se encontraron archivos Excel en Downloads")
            exit()
    else:
        print("No se puede acceder a la carpeta Downloads")
        exit()

try:
    df = pd.read_excel(ruta_archivo)
    print(f"ARCHIVO CARGADO EXITOSAMENTE")
    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Nombre del archivo: {os.path.basename(ruta_archivo)}")
except Exception as e:
    print(f"ERROR AL LEER EL ARCHIVO: {e}")
    print("Posibles soluciones:")
    print("   1. Verifica que el archivo no esté abierto en Excel")
    print("   2. Instala openpyxl: pip install openpyxl")
    print("   3. Verifica que el archivo no esté corrupto")
    exit()

# 2. INFORMACIÓN DEL DATAFRAME
print("\n" + "=" * 60)
print("INFORMACIÓN DEL DATASET")
print("=" * 60)

print("PRIMERAS 5 FILAS:")
print(df.head())

print("\n NOMBRES DE COLUMNAS:")
print(df.columns.tolist())

print("\n INFORMACIÓN GENERAL:")
print(df.info())

print("\n VALORES NULOS POR COLUMNA:")
print(df.isnull().sum())

print("\n DISTRIBUCIÓN DE DATOS:")
print(df.describe())

# 3. VERIFICACIÓN DE COLUMNAS
columnas_requeridas = ['Cliente_ID', 'Recompra']
columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]

if columnas_faltantes:
    print(f"\n COLUMNAS FALTANTES: {columnas_faltantes}")
    print("COLUMNAS DISPONIBLES:")
    for col in df.columns:
        print(f"   - {col}")
    exit()
else:
    print(f"\n TODAS LAS COLUMNAS REQUERIDAS ESTÁN PRESENTES")

# 4. PROCESAMIENTO DE DATOS
print("\n" + "=" * 60)
print("PROCESAMIENTO DE DATOS")
print("=" * 60)

if 'Genero' in df.columns:
    print("Valores únicos en Género:", df['Genero'].unique())
    df['Genero'] = df['Genero'].map({'F': 0, 'M': 1, 'Femenino': 0, 'Masculino': 1})

if 'Recibio_Promo' in df.columns or 'Recibió_Promo' in df.columns:
    col_name = 'Recibio_Promo' if 'Recibio_Promo' in df.columns else 'Recibió_Promo'
    print(f"Valores únicos en {col_name}:", df[col_name].unique())
    df['Recibio_Promo'] = df[col_name].map({'Si': 1, 'No': 0, 'Sí': 1})

if 'Recompra' in df.columns:
    print("Valores únicos en Recompra:", df['Recompra'].unique())
    df['Recompra'] = df['Recompra'].map({'Si': 1, 'No': 0, 'Sí': 1})

columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

# 5. ANÁLISIS EXPLORATORIO
print("\n" + "=" * 60)
print("ANÁLISIS EXPLORATORIO - PREGUNTAS CLAVE")
print("=" * 60)

# 1. Promoción vs Recompra
if 'Recibio_Promo' in df.columns:
    print("\n1. ¿Recibir una promoción influye en la recompra?")
    tabla = pd.crosstab(df['Recibio_Promo'], df['Recompra'], normalize='index') * 100
    print("Distribución porcentual de Recompra según Recibio_Promo:")
    print(tabla)

    # Gráfico
    tabla.plot(kind='bar', stacked=True, colormap='Set2')
    plt.title("Distribución de Recompra según si Recibió Promoción")
    plt.xlabel("Recibió Promoción (0=No, 1=Sí)")
    plt.ylabel("Porcentaje")
    plt.legend(["No Recompra", "Sí Recompra"], title="Recompra")
    plt.tight_layout()
    plt.show()

# 2. Monto de Promoción
if 'Monto_Promo' in df.columns:
    print("\n2. ¿Importa el monto de la promoción?")
    df['Monto_Promo'] = pd.to_numeric(df['Monto_Promo'], errors='coerce')
    promedio_recompra = df.groupby('Recompra')['Monto_Promo'].mean()
    print("Promedio de Monto_Promo según Recompra (0=No, 1=Sí):")
    print(promedio_recompra)

    # Gráfico
    sns.boxplot(x='Recompra', y='Monto_Promo', data=df, palette='Set3')
    plt.title("Monto de Promoción según Recompra")
    plt.xlabel("Recompra (0=No, 1=Sí)")
    plt.ylabel("Monto de Promoción")
    plt.tight_layout()
    plt.show()

# 3. Edad e Ingreso
if 'Edad' in df.columns and 'Ingreso' in df.columns:
    print("\n3. ¿Influyen la edad y el ingreso en la recompra?")
    df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')
    df['Ingreso'] = pd.to_numeric(df['Ingreso'], errors='coerce')
    print("Promedio de Edad e Ingreso según Recompra:")
    print(df.groupby('Recompra')[['Edad', 'Ingreso']].mean())

    # Boxplot Edad
    sns.boxplot(x='Recompra', y='Edad', data=df, palette='coolwarm')
    plt.title("Edad según Recompra")
    plt.xlabel("Recompra (0=No, 1=Sí)")
    plt.ylabel("Edad")
    plt.tight_layout()
    plt.show()

    # Boxplot Ingreso
    sns.boxplot(x='Recompra', y='Ingreso', data=df, palette='coolwarm')
    plt.title("Ingreso según Recompra")
    plt.xlabel("Recompra (0=No, 1=Sí)")
    plt.ylabel("Ingreso")
    plt.tight_layout()
    plt.show()

# 6. MODELADO PREDICTIVO
print("\n" + "=" * 60)
print("MODELADO PREDICTIVO")
print("=" * 60)

X = df.drop(['Cliente_ID', 'Recompra'], axis=1)
y = df['Recompra']

X = X.fillna(0)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

# Matriz de Confusión
print("\nMATRIZ DE CONFUSIÓN:")
matriz = confusion_matrix(y_test, y_pred)
print(matriz)

sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Reporte
print("\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred))

# Importancia de variables
importancias = pd.Series(modelo.feature_importances_, index=X.columns)
importancias_ordenadas = importancias.sort_values(ascending=False)

print("\nIMPORTANCIA DE VARIABLES SEGÚN EL MODELO:")
print(importancias_ordenadas)

plt.figure(figsize=(10, 6))
sns.barplot(x=importancias_ordenadas, y=importancias_ordenadas.index, palette="viridis")
plt.title("Importancia de Variables en el Modelo de Árbol de Decisión")
plt.xlabel("Importancia")
plt.ylabel("Variables")
plt.tight_layout()
plt.show()

print("=" * 60)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 60)
