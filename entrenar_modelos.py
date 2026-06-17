"""
entrenar_modelos.py
--------------------
Entrena y guarda los modelos finales en models/*.joblib.

- Regresion (costo_esperado_anual_mxn): RandomForestRegressor dentro de un
  Pipeline con el preprocesador. Se compara con Lineal / Ridge / Lasso y arbol.
- Clasificacion (riesgo_alto): RandomForestClassifier con class_weight="balanced"
  para tratar el desbalance (~15% positivos).

Todos los modelos usan train_test_split con random_state=42 y el
preprocesamiento se ajusta SOLO con el conjunto de entrenamiento (Pipeline),
evitando fugas de informacion.

Ejecucion:  python entrenar_modelos.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

from utils.preprocessing import (
    cargar_datos, preparar_features, construir_preprocesador,
    columnas_features, acotar_costo, TARGET_REG, TARGET_CLF,
)

RUTA_DATOS = os.path.join("data", "seguro_auto_actuarial.csv")
DIR_MODELOS = "models"
RANDOM_STATE = 42


def cargar_xy():
    df = cargar_datos(RUTA_DATOS)
    df = preparar_features(df)
    X = df[columnas_features()]
    # Se acotan los costos catastroficos extremos (percentil 99) por ser
    # outliers que distorsionan el modelo de regresion.
    y_reg = acotar_costo(df[TARGET_REG])
    y_clf = df[TARGET_CLF]
    return X, y_reg, y_clf


def entrenar_regresion(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    modelos = {
        "Regresion lineal": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=10.0, random_state=RANDOM_STATE, max_iter=10000),
        "Arbol de decision": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20,
                                                   random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=None,
                                               min_samples_leaf=5, n_jobs=-1,
                                               random_state=RANDOM_STATE),
    }
    resultados = {}
    mejor_nombre, mejor_pipe, mejor_r2 = None, None, -np.inf
    for nombre, modelo in modelos.items():
        pipe = Pipeline([("prep", construir_preprocesador()), ("model", modelo)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        resultados[nombre] = {
            "MAE": mean_absolute_error(yte, pred),
            "RMSE": rmse,
            "R2": r2_score(yte, pred),
        }
        if resultados[nombre]["R2"] > mejor_r2:
            mejor_r2 = resultados[nombre]["R2"]
            mejor_nombre, mejor_pipe = nombre, pipe
    return resultados, mejor_nombre, mejor_pipe


def entrenar_clasificacion(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE,
                                          stratify=y)
    modelos = {
        "Regresion logistica": LogisticRegression(max_iter=1000,
                                                  class_weight="balanced"),
        "Arbol de decision": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                    class_weight="balanced",
                                                    random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                                min_samples_leaf=3,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=RANDOM_STATE),
    }
    resultados = {}
    mejor_nombre, mejor_pipe, mejor_f1 = None, None, -np.inf
    for nombre, modelo in modelos.items():
        pipe = Pipeline([("prep", construir_preprocesador()), ("model", modelo)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        resultados[nombre] = {
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0),
            "F1": f1_score(yte, pred, zero_division=0),
        }
        if resultados[nombre]["F1"] > mejor_f1:
            mejor_f1 = resultados[nombre]["F1"]
            mejor_nombre, mejor_pipe = nombre, pipe
    return resultados, mejor_nombre, mejor_pipe


def main():
    os.makedirs(DIR_MODELOS, exist_ok=True)
    X, y_reg, y_clf = cargar_xy()

    print("=" * 60)
    print("REGRESION (costo_esperado_anual_mxn)")
    print("=" * 60)
    res_reg, nombre_reg, pipe_reg = entrenar_regresion(X, y_reg)
    print(pd.DataFrame(res_reg).T.round(3))
    print(f"\nMejor modelo de regresion: {nombre_reg}")
    joblib.dump(pipe_reg, os.path.join(DIR_MODELOS, "modelo_regresion.joblib"))

    print("\n" + "=" * 60)
    print("CLASIFICACION (riesgo_alto)")
    print("=" * 60)
    res_clf, nombre_clf, pipe_clf = entrenar_clasificacion(X, y_clf)
    print(pd.DataFrame(res_clf).T.round(3))
    print(f"\nMejor modelo de clasificacion: {nombre_clf}")
    joblib.dump(pipe_clf, os.path.join(DIR_MODELOS, "modelo_clasificacion.joblib"))

    print(f"\nModelos guardados en: {DIR_MODELOS}/")


if __name__ == "__main__":
    main()
