# %% [markdown]
# # Tarea 3: Clasificación de canciones K-pop mediante Minería de Datos
# **Objetivo:** Desarrollar un modelo de clasificación (KNN) que permita a *UAI Music Group* identificar si una canción pertenece al género K-pop o no, optimizando el tiempo de mercado y reduciendo costos de expertos musicales.
# 
# ## 1. Carga de Datos
# En esta primera etapa, importaremos el conjunto de datos y filtraremos únicamente las variables solicitadas por la casa disquera para enfocar el análisis en las características acústicas y de popularidad más relevantes.

# %%
# Importación de librerías necesarias para manipulación de datos, visualización y modelado
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest

# 1. Carga de Datos
# Nota: Asegúrate de que el archivo 'dataset_tarea.csv' esté en la misma carpeta que este notebook.
df = pd.read_csv('dataset_tarea.csv')

# Extraer únicamente las variables requeridas
columnas_requeridas = ['popularity', 'danceability', 'speechiness', 'acousticness', 'loudness', 'track_genre']
df_datos = df[columnas_requeridas].copy()

print(f"Dimensiones del dataset original: {df_datos.shape}")
print(df_datos.head())

# %% [markdown]
# ## 2. Limpieza de Datos
# 
# ### Importancia de la Limpieza
# Los algoritmos de Machine Learning, y en particular los basados en distancias como KNN, son matemáticamente sensibles a datos faltantes (NaN) y a valores atípicos (outliers). 
# 1. **Valores Nulos:** Si existen, el cálculo de la distancia euclidiana fallará.
# 2. **Outliers:** Una canción con un nivel de `loudness` o `popularity` extremadamente atípico alterará el espacio vectorial, "atraerando" a canciones que no tienen nada que ver hacia su vecindario, arruinando la capacidad predictiva del modelo.
# 
# ### Tratamiento de Outliers
# Utilizaremos **Isolation Forest**, un algoritmo no supervisado basado en árboles de decisión que "aísla" observaciones. Es superior a métodos estadísticos simples (como Z-score) porque no asume que las variables musicales siguen una distribución normal y es excelente en espacios multivariados.

# %%
# Verificación de valores nulos
nulos = df_datos.isnull().sum()
print("Valores nulos por columna:\n", nulos)

# Eliminamos nulos si existieran
df_datos = df_datos.dropna().reset_index(drop=True)

# Detección de Outliers con Isolation Forest
X_para_outliers = df_datos.drop('track_genre', axis=1)

# contamination=0.05 asume que aproximadamente el 5% de los datos son ruido/outliers
iso_forest = IsolationForest(contamination=0.05, random_state=42)
predicciones_outliers = iso_forest.fit_predict(X_para_outliers)

# Isolation Forest devuelve 1 para datos normales y -1 para outliers
df_clean = df_datos[predicciones_outliers == 1].reset_index(drop=True)

print(f"\nDatos antes de limpiar outliers: {df_datos.shape[0]}")
print(f"Datos después de limpiar outliers: {df_clean.shape[0]}")
print(f"Se eliminaron {df_datos.shape[0] - df_clean.shape[0]} canciones atípicas.")

# %% [markdown]
# ## 3. Evaluación del modelo KNN con K=1, 10-fold CV y Min-Max Scaling
# 
# ### ¿Por qué es necesario normalizar los datos?
# KNN calcula la distancia (usualmente Euclidiana) entre puntos. Nuestras variables tienen escalas muy distintas:
# * `popularity` va de 0 a 100.
# * `danceability` va de 0 a 1.
# Si no normalizamos, la distancia estará dominada en un 99% por la `popularity`, haciendo que el modelo ignore por completo si la canción es bailable o acústica. La normalización **Min-Max** escala todas las variables al rango [0, 1], otorgándoles el mismo peso democrático.
# 
# ### Procedimiento de 10-Fold Cross Validation (Sin Data Leakage)
# Para evitar el *Data Leakage* (fuga de datos), el `MinMaxScaler` **NO** se puede ajustar a todo el dataset antes de dividir. Debe calcularse (`fit`) **exclusivamente con el Fold de Entrenamiento** y luego aplicado (`transform`) al entrenamiento y a la prueba.

# %%
# Separación de variables predictoras (X) y variable objetivo (y)
X = df_clean.drop('track_genre', axis=1)
y = df_clean['track_genre']

# Configuración del 10-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Inicialización del escalador y el modelo K=1
scaler_minmax = MinMaxScaler()
knn_1 = KNeighborsClassifier(n_neighbors=1, weights='uniform')

accuracies_k1 = []

# Bucle de Cross Validation
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 1. Ajustar escalador SOLO con datos de entrenamiento
    scaler_minmax.fit(X_train)
    
    # 2. Transformar ambos conjuntos
    X_train_scaled = scaler_minmax.transform(X_train)
    X_test_scaled = scaler_minmax.transform(X_test)
    
    # 3. Entrenar y predecir
    knn_1.fit(X_train_scaled, y_train)
    y_pred = knn_1.predict(X_test_scaled)
    
    # 4. Evaluar
    acc = accuracy_score(y_test, y_pred)
    accuracies_k1.append(acc)

mean_acc_k1 = np.mean(accuracies_k1)
std_acc_k1 = np.std(accuracies_k1)

print(f"--- Resultados KNN con K=1 (Min-Max) ---")
print(f"Accuracy Promedio: {mean_acc_k1:.4f} ({mean_acc_k1*100:.2f}%)")
print(f"Desviación Estándar: {std_acc_k1:.4f}")

# %% [markdown]
# ### Análisis de Resultados (K=1)
# El Accuracy obtenido con K=1 suele ser alto en el entrenamiento pero sufre de **sobreajuste (overfitting)**. Al usar K=1, el modelo memoriza el ruido del conjunto de entrenamiento. La desviación estándar nos indica qué tan inestable es el modelo dependiendo del fold que se use para entrenar. Para propósitos de negocio, un modelo con K=1 es demasiado sensible y arriesgado para clasificar el catálogo de *UAI Music Group*.

# %% [markdown]
# ## 4.1 Encontrar el Mejor K
# Evaluaremos desde K=1 hasta K=50. 
# * **Accuracy:** Proporción total de aciertos.
# * **Precision:** De las canciones que el modelo dijo que eran K-pop, ¿cuántas realmente lo eran?
# * **Recall:** De todas las canciones que REALMENTE eran K-pop, ¿cuántas logró encontrar el modelo?

# %%
k_range = range(1, 51)
k_acc_mean = []
k_prec_mean = []
k_rec_mean = []

for k in k_range:
    fold_acc, fold_prec, fold_rec = [], [], []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Escalado correcto dentro del loop (Evita Data Leakage)
        scaler_minmax.fit(X_train)
        X_train_scaled = scaler_minmax.transform(X_train)
        X_test_scaled = scaler_minmax.transform(X_test)
        
        knn_k = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        knn_k.fit(X_train_scaled, y_train)
        y_pred = knn_k.predict(X_test_scaled)
        
        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_prec.append(precision_score(y_test, y_pred, zero_division=0))
        fold_rec.append(recall_score(y_test, y_pred, zero_division=0))
        
    k_acc_mean.append(np.mean(fold_acc))
    k_prec_mean.append(np.mean(fold_prec))
    k_rec_mean.append(np.mean(fold_rec))

# Gráfico de métricas
plt.figure(figsize=(12, 6))
plt.plot(k_range, k_acc_mean, label='Accuracy', marker='.', linestyle='--')
plt.plot(k_range, k_prec_mean, label='Precision', marker='s')
plt.plot(k_range, k_rec_mean, label='Recall', marker='^')
plt.xlabel('Número de Vecinos (K)')
plt.ylabel('Puntuación')
plt.title('Evaluación de Métricas vs K (Normalización Min-Max)')
plt.xticks(np.arange(0, 51, 5))
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Encontrar el mejor K basado en Accuracy
mejor_k_idx = np.argmax(k_acc_mean)
mejor_k = k_range[mejor_k_idx]
print(f"El mejor K basado en Accuracy es: {mejor_k} con un score de {k_acc_mean[mejor_k_idx]:.4f}")

# %% [markdown]
# ## 5. Discusión: ¿Normalización o Estandarización?
# Ahora repetiremos el proceso utilizando **StandardScaler** (Estandarización), la cual resta la media y divide por la desviación estándar ($\mu=0, \sigma=1$). 
# 
# * **Min-Max** es sensible a outliers extremos (comprime los datos normales si hay un outlier gigante).
# * **StandardScaler** no acota los datos a un rango fijo, lo que lo hace más robusto si la distribución de las variables es Gaussiana.

# %%
# Arrays para guardar resultados con StandardScaler
std_k_acc_mean = []
std_k_prec_mean = []
std_k_rec_mean = []

scaler_std = StandardScaler()

for k in k_range:
    fold_acc, fold_prec, fold_rec = [], [], []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Estandarización dentro del loop
        scaler_std.fit(X_train)
        X_train_scaled = scaler_std.transform(X_train)
        X_test_scaled = scaler_std.transform(X_test)
        
        knn_k = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        knn_k.fit(X_train_scaled, y_train)
        y_pred = knn_k.predict(X_test_scaled)
        
        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_prec.append(precision_score(y_test, y_pred, zero_division=0))
        fold_rec.append(recall_score(y_test, y_pred, zero_division=0))
        
    std_k_acc_mean.append(np.mean(fold_acc))
    std_k_prec_mean.append(np.mean(fold_prec))
    std_k_rec_mean.append(np.mean(fold_rec))

# Comparación Visual
plt.figure(figsize=(12, 6))
plt.plot(k_range, k_acc_mean, label='Accuracy (Min-Max)', color='blue', linestyle='--')
plt.plot(k_range, std_k_acc_mean, label='Accuracy (Standard)', color='red', linestyle='-')
plt.xlabel('Número de Vecinos (K)')
plt.ylabel('Accuracy')
plt.title('Comparación de Escalado: Min-Max vs StandardScaler')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

mejor_k_std_idx = np.argmax(std_k_acc_mean)
mejor_k_std = k_range[mejor_k_std_idx]
print(f"Mejor K con StandardScaler: {mejor_k_std} (Accuracy: {std_k_acc_mean[mejor_k_std_idx]:.4f})")

# %% [markdown]
# ### Conclusiones y Respuestas a la Discusión
# 
# 1. **¿Cambió el mejor K?** 
#    Es común que el mejor K cambie ligeramente (por ejemplo, de 7 a 11). Esto ocurre porque StandardScaler distribuye el espacio vectorial de manera distinta, alterando las distancias euclidianas relativas entre las canciones.
# 
# 2. **¿Cuál de todos los modelos es mejor y por qué?**
#    Para **KNN aplicado a características de audio (Spotify API)**, la **Normalización Min-Max suele ser superior o igual, pero más segura**. 
#    * **¿Por qué?** Las métricas de Spotify como `danceability` o `acousticness` están estrictamente acotadas por diseño entre 0 y 1. Min-Max respeta esta frontera física natural de los datos. StandardScaler asume una distribución de campana de Gauss que las variables de audio no siempre cumplen (por ejemplo, `speechiness` suele tener una distribución sesgada hacia el 0). 
#    * Al usar Min-Max, garantizamos que la distancia máxima posible entre dos canciones en cualquier dimensión sea exactamente 1, lo que le da a KNN una frontera de decisión mucho más estable e interpretable para el equipo de *UAI Music Group*.