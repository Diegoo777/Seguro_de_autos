"""
limpieza_datos.py
------------------
Limpia la base de datos original y genera un CSV limpio para exploracion.

Pasos de limpieza:
 1. Lee el CSV con encoding utf-8-sig para eliminar el BOM inicial.
 2. Elimina filas/columnas vacias y registros duplicados.
 3. Imputa faltantes numericos con la MEDIANA (robusta a outliers).
 4. Imputa faltantes categoricos con la etiqueta "Desconocido".
 5. Trata outliers de ingreso_mensual_mxn por winsorizacion (percentiles 1-99).
 6. Crea variables nuevas (ingenieria de caracteristicas).
 7. Guarda data/seguro_auto_actuarial_limpio.csv

NOTA: este CSV limpio es para EXPLORACION. El modelado de la app aplica la
imputacion/escalamiento DENTRO de un Pipeline ajustado solo con el train,
para evitar fugas de informacion (data leakage).

Ejecucion:  python limpieza_datos.py
"""

import os
import numpy as np
import pandas as pd

from utils.preprocessing import (
    NUMERIC_COLS_BASE, NOMINAL_COLS, ORDINAL_COLS, BINARY_COLS,
    crear_variables,
)

RUTA_ENTRADA = os.path.join("data", "seguro_auto_actuarial.csv")
RUTA_SALIDA = os.path.join("data", "seguro_auto_actuarial_limpio.csv")


def limpiar(df):
    df = df.copy()
    n_inicial = len(df)

    # 2. Duplicados
    df = df.drop_duplicates().reset_index(drop=True)
    n_dup = n_inicial - len(df)

    # 3. Imputacion numerica con mediana
    cols_num_con_na = [c for c in NUMERIC_COLS_BASE if df[c].isna().any()]
    for c in cols_num_con_na:
        df[c] = df[c].fillna(df[c].median())

    # 4. Imputacion categorica con "Desconocido"
    cols_cat = NOMINAL_COLS + ["nivel_estudios", "segmento_marca"]
    cols_cat_con_na = [c for c in cols_cat if df[c].isna().any()]
    for c in cols_cat_con_na:
        df[c] = df[c].fillna("Desconocido")

    # Binarias Si/No faltantes -> moda
    bin_texto = ["mantenimiento_al_dia"]
    for c in bin_texto:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    # 5. Winsorizacion de ingreso_mensual_mxn (trata outliers extremos)
    lo, hi = df["ingreso_mensual_mxn"].quantile([0.01, 0.99])
    n_out = ((df["ingreso_mensual_mxn"] < lo) | (df["ingreso_mensual_mxn"] > hi)).sum()
    df["ingreso_mensual_mxn"] = df["ingreso_mensual_mxn"].clip(lower=lo, upper=hi)

    # 6. Variables nuevas
    df = crear_variables(df)

    print(f"Registros iniciales : {n_inicial}")
    print(f"Duplicados eliminados: {n_dup}")
    print(f"Columnas numericas imputadas (mediana): {cols_num_con_na}")
    print(f"Columnas categoricas imputadas (Desconocido): {cols_cat_con_na}")
    print(f"Outliers de ingreso winsorizados: {n_out} (limites {lo:.0f} - {hi:.0f})")
    print(f"Faltantes restantes  : {int(df.isna().sum().sum())}")
    return df


def main():
    df = pd.read_csv(RUTA_ENTRADA, encoding="utf-8-sig")
    df_limpio = limpiar(df)
    df_limpio.to_csv(RUTA_SALIDA, index=False, encoding="utf-8")
    print(f"\nArchivo limpio guardado en: {RUTA_SALIDA}")
    print(f"Dimensiones finales: {df_limpio.shape}")


if __name__ == "__main__":
    main()
