"""
utils/preprocessing.py
-----------------------
Definiciones de columnas, ingenieria de variables y construccion del
ColumnTransformer / Pipeline de scikit-learn para el proyecto actuarial.

Toda la logica de preprocesamiento vive aqui para que la app (app.py),
el notebook y el script de entrenamiento usen exactamente las mismas
transformaciones y se eviten fugas de informacion (data leakage):
la imputacion, el escalamiento y la codificacion se ajustan SOLO con los
datos de entrenamiento dentro del Pipeline.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Definicion de columnas segun el diccionario de datos
# ---------------------------------------------------------------------------

# Columnas que NUNCA deben usarse como predictores
ID_COL = "poliza_id"
TARGET_REG = "costo_esperado_anual_mxn"   # objetivo de regresion
TARGET_CLF = "riesgo_alto"                # objetivo de clasificacion
LEAKAGE_COLS = ["clase_costo"]            # se deriva directamente del costo

# Variables binarias Si/No -> se transforman a 1/0
BINARY_COLS = ["tiene_gps", "asistencia_vial", "mantenimiento_al_dia"]

# Variables categoricas nominales (sin orden) -> One-Hot Encoding
NOMINAL_COLS = [
    "sexo", "estado_civil", "ocupacion", "zona_residencia", "region",
    "tipo_vehiculo", "uso_vehiculo", "metodo_pago", "canal_venta",
]

# Variables categoricas ordinales (tienen un orden natural) -> Ordinal Encoding
# El orden se justifica: mas estudios = nivel mayor; segmento de marca de menor
# a mayor gama; grupo de edad de menor a mayor.
ORDINAL_COLS = ["nivel_estudios", "segmento_marca", "grupo_edad"]
ORDINAL_CATEGORIES = [
    ["Secundaria", "Preparatoria", "Licenciatura", "Posgrado"],   # nivel_estudios
    ["Economico", "Medio", "Premium"],                            # segmento_marca
    ["Joven", "Adulto_joven", "Adulto", "Mayor"],                 # grupo_edad
]

# Variables numericas originales (continuas / discretas)
NUMERIC_COLS_BASE = [
    "edad_conductor", "antiguedad_cliente_anios", "ingreso_mensual_mxn",
    "score_crediticio", "prima_mensual_mxn", "suma_asegurada_mxn",
    "deducible_pct", "historial_siniestros_3_anios", "km_anuales",
    "edad_vehiculo_anios", "dias_hasta_renovacion", "puntaje_riesgo_zona",
    "numero_siniestros_12m",
]

# Variables numericas nuevas creadas por ingenieria de caracteristicas
NUMERIC_COLS_ENGINEERED = [
    "km_por_anio_vehiculo", "log_ingreso", "ratio_prima_ingreso",
    "siniestros_x_suma", "siniestros_x_prima",
]

NUMERIC_COLS = NUMERIC_COLS_BASE + NUMERIC_COLS_ENGINEERED

# Cuantil para acotar (winsorizar) los costos extremos del objetivo de regresion.
# La base tiene ~10 polizas con costos catastroficos (hasta 131,385 MXN frente a
# una mediana de ~5,570) que distorsionan el ajuste e inflan el RMSE. En la
# practica actuarial estos siniestros catastroficos se tratan aparte (reaseguro),
# por lo que acotamos el objetivo al percentil 99 antes de modelar.
TARGET_CAP_QUANTILE = 0.99


# ---------------------------------------------------------------------------
# 2. Carga e ingenieria de variables
# ---------------------------------------------------------------------------

def cargar_datos(ruta_csv):
    """Carga el CSV. utf-8-sig elimina el BOM (caracter invisible al inicio)."""
    df = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    return df


def mapear_binarias(df):
    """Convierte las variables Si/No a 1/0. Los faltantes quedan como NaN
    para que el imputador los maneje aguas abajo."""
    df = df.copy()
    mapa = {"Si": 1, "No": 0}
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(mapa)
    return df


def discretizar_edad(edad):
    """Discretiza la edad del conductor en rangos (variable nueva ordinal)."""
    if pd.isna(edad):
        return np.nan
    if edad < 25:
        return "Joven"
    elif edad < 40:
        return "Adulto_joven"
    elif edad < 60:
        return "Adulto"
    else:
        return "Mayor"


def crear_variables(df):
    """Genera las variables nuevas por interaccion / transformacion.
    Son transformaciones fila a fila (deterministas) por lo que no producen
    fuga de informacion al calcularse antes del split.
    """
    df = df.copy()

    # Intensidad de uso del vehiculo: km recorridos por anio de antiguedad
    df["km_por_anio_vehiculo"] = df["km_anuales"] / (df["edad_vehiculo_anios"] + 1)

    # Transformacion logaritmica del ingreso (reduce el sesgo / outliers)
    df["log_ingreso"] = np.log1p(df["ingreso_mensual_mxn"])

    # Carga relativa de la prima respecto al ingreso
    df["ratio_prima_ingreso"] = df["prima_mensual_mxn"] / (df["ingreso_mensual_mxn"] + 1)

    # Interacciones con el numero de siniestros: el costo escala con la
    # frecuencia de siniestros multiplicada por la exposicion (suma asegurada
    # y prima). Son los predictores mas correlacionados con el costo.
    df["siniestros_x_suma"] = df["numero_siniestros_12m"] * df["suma_asegurada_mxn"]
    df["siniestros_x_prima"] = df["numero_siniestros_12m"] * df["prima_mensual_mxn"]

    # Discretizacion de la edad en grupos (ordinal)
    df["grupo_edad"] = df["edad_conductor"].apply(discretizar_edad)

    return df


def acotar_costo(y, q=TARGET_CAP_QUANTILE):
    """Acota (winsoriza) los costos extremos del objetivo de regresion al
    percentil indicado. Devuelve una copia de la serie con el tope aplicado."""
    tope = y.quantile(q)
    return y.clip(upper=tope)


def preparar_features(df):
    """Pipeline completo de preparacion previo al modelo:
    mapeo de binarias + creacion de variables nuevas.
    Devuelve el DataFrame listo para alimentar el ColumnTransformer.
    """
    df = mapear_binarias(df)
    df = crear_variables(df)
    return df


def columnas_features():
    """Lista ordenada de columnas que entran al modelo (predictores)."""
    return NUMERIC_COLS + BINARY_COLS + NOMINAL_COLS + ORDINAL_COLS


# ---------------------------------------------------------------------------
# 3. Construccion del preprocesador (ColumnTransformer)
# ---------------------------------------------------------------------------

def construir_preprocesador():
    """Crea el ColumnTransformer que imputa, escala y codifica.

    - Numericas : imputacion por mediana + estandarizacion (StandardScaler).
    - Binarias  : imputacion por moda (mas frecuente).
    - Nominales : imputacion por moda + One-Hot Encoding.
    - Ordinales : imputacion por moda + Ordinal Encoding con orden justificado.
    """
    transformador_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformador_binario = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    transformador_nominal = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformador_ordinal = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocesador = ColumnTransformer(transformers=[
        ("num", transformador_numerico, NUMERIC_COLS),
        ("bin", transformador_binario, BINARY_COLS),
        ("nom", transformador_nominal, NOMINAL_COLS),
        ("ord", transformador_ordinal, ORDINAL_COLS),
    ])

    return preprocesador


def nombres_features_transformadas(preprocesador):
    """Devuelve los nombres de las columnas despues del preprocesamiento,
    util para interpretar coeficientes e importancias de variables."""
    nombres = []
    nombres.extend(NUMERIC_COLS)
    nombres.extend(BINARY_COLS)
    # One-Hot expande cada categoria nominal
    onehot = preprocesador.named_transformers_["nom"].named_steps["onehot"]
    nombres.extend(list(onehot.get_feature_names_out(NOMINAL_COLS)))
    nombres.extend(ORDINAL_COLS)
    return nombres
