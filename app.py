"""
app.py
-------
Sistema actuarial de prediccion de riesgo y costo esperado en seguros de
automovil. Aplicacion en Streamlit (Proyecto Integrador - Actuaria y Ciencia
de Datos / Machine Learning).

Ejecucion:
    streamlit run app.py

Secciones (barra lateral):
    1. Inicio
    2. Exploracion de datos
    3. Preprocesamiento
    4. Modelado
    5. Reduccion de dimensionalidad
    6. Simulador actuarial
    7. Imagenes
    8. Conclusiones
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import cv2

from PIL import Image, ImageFilter, ImageEnhance, ImageOps

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from utils import preprocessing as pp
from utils import plots

# ---------------------------------------------------------------------------
# Configuracion general
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sistema Actuarial de Seguros de Auto",
                   layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.join(BASE_DIR, "data", "seguro_auto_actuarial.csv")
DIR_MODELOS = os.path.join(BASE_DIR, "models")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Carga de datos y modelos (con cache)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pp.cargar_datos(RUTA_DATOS)
    df_feat = pp.preparar_features(df)
    return df, df_feat


@st.cache_resource
def obtener_modelos():
    """Carga los modelos .joblib si existen; si no, los entrena al vuelo."""
    ruta_reg = os.path.join(DIR_MODELOS, "modelo_regresion.joblib")
    ruta_clf = os.path.join(DIR_MODELOS, "modelo_clasificacion.joblib")

    df, df_feat = cargar_datos()
    X = df_feat[pp.columnas_features()]
    y_reg = df_feat[pp.TARGET_REG]
    y_clf = df_feat[pp.TARGET_CLF]

    if os.path.exists(ruta_reg) and os.path.exists(ruta_clf):
        modelo_reg = joblib.load(ruta_reg)
        modelo_clf = joblib.load(ruta_clf)
    else:
        modelo_reg = Pipeline([("prep", pp.construir_preprocesador()),
                               ("model", RandomForestRegressor(
                                   n_estimators=200, min_samples_leaf=5,
                                   n_jobs=-1, random_state=RANDOM_STATE))])
        modelo_reg.fit(X, y_reg)
        modelo_clf = Pipeline([("prep", pp.construir_preprocesador()),
                               ("model", RandomForestClassifier(
                                   n_estimators=300, min_samples_leaf=3,
                                   class_weight="balanced", n_jobs=-1,
                                   random_state=RANDOM_STATE))])
        modelo_clf.fit(X, y_clf)
    return modelo_reg, modelo_clf


# ---------------------------------------------------------------------------
# Barra lateral / navegacion
# ---------------------------------------------------------------------------
st.sidebar.title("Menú")
SECCIONES = [
    "1. Inicio",
    "2. Exploracion de datos",
    "3. Preprocesamiento",
    "4. Modelado",
    "5. Reduccion de dimensionalidad",
    "6. Simulador actuarial",
    "7. Imagenes",
    "8. Conclusiones",
]
seccion = st.sidebar.radio("", SECCIONES)
st.sidebar.markdown("---")
st.sidebar.caption("Proyecto Integrador · Actuaria y Ciencia de Datos")
st.sidebar.caption("Hecho por:")
st.sidebar.caption("Abigail Sampedro Gutiérrez")
st.sidebar.caption("Diego Pedraza Barajas")
st.sidebar.caption("Juan Diego Hernández Zavala")

df, df_feat = cargar_datos()


# ===========================================================================
# 1. INICIO
# ===========================================================================
if seccion == SECCIONES[0]:
    st.title("App de predicción de costos para una aseguradora")
    st.subheader("Seguros de automovil")
    st.markdown("""
En este proyecto se buscan dos objetivos principales:

1. **Estimar el costo esperado anual** de siniestros de una poliza
   (problema de **regresion** → estimar el valor `costo_esperado_anual_mxn`).
2. **Clasificar si el asegurado pertenece al grupo de riesgo alto**
   (problema de **clasificacion** → `riesgo_alto`, alrededor del 15%).

Esta aplicacion permite explorar los datos, limpiarlos, entrenar/cargar
modelos, evaluar resultados, interpretar variables, simular nuevas
polizas y trabajar con imagenes de vehiculos o siniestros.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros (polizas)", f"{df.shape[0]:,}")
    c2.metric("Variables", f"{df.shape[1]}")
    c3.metric("% Riesgo alto", f"{100 * df['riesgo_alto'].mean():.1f}%")

    st.markdown("### Descripcion de la base de datos")
    st.markdown("""
- La base de datos utilizada simula polizas de autos (`seguro_auto_actuarial.csv`).
- Contiene variables numericas, categoricas nominales y ordinales, binarias.
    """)


# ===========================================================================
# 2. EXPLORACION DE DATOS
# ===========================================================================
elif seccion == SECCIONES[1]:
    st.title("Exploracion de datos")

    st.markdown("### Vista de los datos")
    st.dataframe(df.head(20), width="stretch")
    st.caption(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

    st.markdown("### Tipos de variables")
    tipos = pd.DataFrame({
        "Variable": df.columns,
        "Tipo": df.dtypes.astype(str).values,
        "Faltantes": df.isna().sum().values,
        "% Faltantes": (100 * df.isna().mean()).round(1).values,
        "Valores unicos": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(tipos, width="stretch", height=350)

    faltantes = tipos[tipos["Faltantes"] > 0]
    if len(faltantes) > 0:
        st.warning("Variables con valores faltantes: "
                   + ", ".join(f"{r.Variable} ({r.Faltantes})"
                               for r in faltantes.itertuples()))

    st.markdown("### Balance de la clase objetivo `riesgo_alto`")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.pyplot(plots.fig_balance_clases(df, "riesgo_alto"))
    with col_b:
        st.markdown("""
**Interpretacion:** la clase positiva (riesgo alto) representa solo ~15% de las
polizas. No basta con el accuracy, hay que vigilar recall, precision
 y F1, y usar `class_weight="balanced"` en los modelos para evitar que sea un problema desbalanceado.
        """)

    st.markdown("### Distribuciones (histogramas)")
    num_cols = pp.NUMERIC_COLS_BASE
    var_hist = st.selectbox("Variable numerica:", num_cols, index=0)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plots.fig_histograma(df, var_hist))
    with c2:
        st.pyplot(plots.fig_boxplot(df, var_hist))
    st.caption("""
            El boxplot ayuda a identificar outliers (puntos fuera de la
            caja). `ingreso_mensual_mxn` y `prima_mensual_mxn` presentan 
            valores extremos.
               """)

    st.markdown("### Matriz de correlacion (Pearson)")
    # Se incluye el objetivo de regresion (costo) al final, igual que en el
    # notebook, para ver que variables se relacionan con el costo esperado.
    st.pyplot(plots.fig_correlacion(df, num_cols + [pp.TARGET_REG]))
    st.caption("Permite ver que variables se relacionan con el costo esperado y "
               "detectar posibles redundancias entre predictores.")


# ===========================================================================
# 3. PREPROCESAMIENTO
# ===========================================================================
elif seccion == SECCIONES[2]:
    st.title("Preprocesamiento")
    st.markdown("""
Todo el preprocesamiento se lleva a cabo en un **`Pipeline` + `ColumnTransformer`**
de scikit-learn y se ajusta solo con los datos de entrenamiento, evitando
fugas de informacion.
    """)

    st.markdown("### Clasificacion de variables")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Numericas** (imputacion mediana + estandarizacion):")
        st.write(pp.NUMERIC_COLS)
        st.markdown("**Binarias** (Si/No → 1/0):")
        st.write(pp.BINARY_COLS)
    with c2:
        st.markdown("**Nominales** (imputacion moda + One-Hot Encoding):")
        st.write(pp.NOMINAL_COLS)
        st.markdown("**Ordinales** (imputacion moda + Ordinal Encoding):")
        for col, cats in zip(pp.ORDINAL_COLS, pp.ORDINAL_CATEGORIES):
            st.write(f"- `{col}`: {' < '.join(cats)}")

    st.markdown("### Estrategias aplicadas")
    st.markdown("""
| Tema | Estrategia | Justificacion |
|------|-----------|---------------|
| Faltantes numericos | Imputacion por **mediana** | Robusta a outliers |
| Faltantes categoricos | Imputacion por **moda** / "Desconocido" | Conserva todas las filas |
| Outliers | Deteccion (boxplot/IQR) + winsorizado de ingreso | Reduce distorsion |
| Escalamiento | **StandardScaler** (estandarizacion) | Modelos sensibles a escala |
| Nominales | **OneHotEncoder** | Sin orden artificial |
| Ordinales | **OrdinalEncoder** con orden justificado | Respeta jerarquia |
| Desbalance | `class_weight="balanced"` | Clase positiva ~15% |
    """)

    st.markdown("### Variables nuevas")
    st.markdown("""
- `km_por_anio_vehiculo` = `km_anuales` / (`edad_vehiculo_anios` + 1) → intensidad de uso.
- `log_ingreso` = log(1 + `ingreso_mensual_mxn`) → reduce sesgo/outliers del ingreso.
- `ratio_prima_ingreso` = `prima_mensual_mxn` / (`ingreso_mensual_mxn` + 1) → carga relativa.
- `grupo_edad` = discretizacion de `edad_conductor` en Joven / Adulto_joven / Adulto / Mayor.
    """)
    st.dataframe(df_feat[["edad_conductor", "grupo_edad", "km_por_anio_vehiculo",
                          "log_ingreso", "ratio_prima_ingreso"]].head(10),
                 width="stretch")

    st.markdown("")
    prep = pp.construir_preprocesador()
    X = df_feat[pp.columnas_features()]
    Xt = prep.fit_transform(X, df_feat[pp.TARGET_CLF])
    n_out = Xt.shape[1] if not hasattr(Xt, "toarray") else Xt.toarray().shape[1]
    st.success(f"La matriz de caracteristicas pasa de **{len(pp.columnas_features())} "
               f"columnas** originales a **{n_out} columnas** tras el One-Hot Encoding.")


# ===========================================================================
# 4. MODELADO
# ===========================================================================
elif seccion == SECCIONES[3]:
    st.title("Modelado y evaluacion")
    X = df_feat[pp.columnas_features()]
    # Se acotan los costos catastroficos (percentil 99) por ser outliers
    # extremos que distorsionan el modelo de regresion.
    y_reg = pp.acotar_costo(df_feat[pp.TARGET_REG])
    y_clf = df_feat[pp.TARGET_CLF]

    st.info(
        f"El objetivo de regresion se **acota al percentil 99** "
        f"(tope ≈ ${df_feat[pp.TARGET_REG].quantile(pp.TARGET_CAP_QUANTILE):,.0f} MXN). "
        "Las ~10 polizas con costos catastroficos se tratan como outliers (en la "
        "practica actuarial se cubren con reaseguro). Esto evita que distorsionen "
        "el ajuste y mejora notablemente el R²."
    )

    def relevancia_variables(pipe, nombres):
        """Extrae los coeficientes (modelos lineales) o las importancias
        (arboles/forest) del modelo final del pipeline. Devuelve la tabla
        ordenada por magnitud y el tipo de medida."""
        modelo = pipe.named_steps["model"]
        if hasattr(modelo, "coef_"):
            valores = np.ravel(modelo.coef_)
            tipo = "Coeficiente"
        else:
            valores = modelo.feature_importances_
            tipo = "Importancia"
        tabla = pd.DataFrame({"Variable": nombres, tipo: valores})
        tabla["__abs"] = tabla[tipo].abs()
        tabla = tabla.sort_values("__abs", ascending=False).drop(columns="__abs")
        return tabla.reset_index(drop=True), tipo

    def mostrar_relevancia(pipe):
        """Grafica + tabla de variables relevantes del modelo seleccionado."""
        nombres = pp.nombres_features_transformadas(pipe.named_steps["prep"])
        tabla, tipo = relevancia_variables(pipe, nombres)
        col_g, col_t = st.columns([1.2, 1])
        with col_g:
            if tipo == "Coeficiente":
                fig = plots.fig_coeficientes(nombres,
                                             np.ravel(pipe.named_steps["model"].coef_))
            else:
                fig = plots.fig_importancia(
                    nombres, pipe.named_steps["model"].feature_importances_,
                    titulo="Importancia de variables")
            st.pyplot(fig)
        with col_t:
            st.markdown(f"**Tabla de {tipo.lower()}s por variable**")
            st.dataframe(tabla.round(4), width="stretch", height=430)
        if tipo == "Coeficiente":
            st.caption("Variables numericas estandarizadas: un coeficiente mayor en "
                       "valor absoluto = mayor influencia. Lasso puede llevar "
                       "coeficientes a 0 para descartar las menos significativas")
        else:
            st.caption("La importancia mide cuanto reduce el error cada variable en "
                       "el conjunto de arboles (suma 1).")
        return tabla

    tab_reg, tab_clf = st.tabs(["Regresion (costo esperado)",
                                "Clasificacion (riesgo alto)"])

    # ---------------- REGRESION ----------------
    with tab_reg:
        st.markdown("Predecimos `costo_esperado_anual_mxn`. Comparamos modelos "
                    "lineales, regularizados y de arboles.")
        Xtr, Xte, ytr, yte = train_test_split(X, y_reg, test_size=0.2,
                                              random_state=RANDOM_STATE)

        @st.cache_data
        def evaluar_regresion():
            modelos = {
                "Regresion lineal": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=10.0, max_iter=10000),
                "Arbol de decision": DecisionTreeRegressor(
                    max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE),
                "Random Forest": RandomForestRegressor(
                    n_estimators=400, min_samples_leaf=2, max_features=0.5,
                    n_jobs=-1, random_state=RANDOM_STATE),
            }
            filas, pipes = [], {}
            for nombre, modelo in modelos.items():
                pipe = Pipeline([("prep", pp.construir_preprocesador()),
                                 ("model", modelo)])
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                filas.append({
                    "Modelo": nombre,
                    "MAE": mean_absolute_error(yte, pred),
                    "RMSE": np.sqrt(mean_squared_error(yte, pred)),
                    "R2": r2_score(yte, pred),
                })
                pipes[nombre] = pipe
            return pd.DataFrame(filas).set_index("Modelo"), pipes

        tabla_reg, pipes_reg = evaluar_regresion()
        st.dataframe(tabla_reg.round(3), width="stretch")
        st.caption("**MAE**: error absoluto medio (mismas unidades, MXN). "
                   "**RMSE**: penaliza mas los errores grandes. "
                   "**R²**: proporcion de varianza explicada.")

        mejor = tabla_reg["R2"].idxmax()
        st.success(f"Mejor modelo por R²: **{mejor}** "
                   f"(R²={tabla_reg.loc[mejor, 'R2']:.3f}, "
                   f"MAE={tabla_reg.loc[mejor, 'MAE']:.0f} MXN).")

        st.markdown("### Explorar un modelo")
        modelo_sel = st.selectbox("Selecciona el modelo de regresion:",
                                  list(pipes_reg.keys()),
                                  index=list(pipes_reg.keys()).index(mejor),
                                  key="sel_reg")
        st.markdown(f"#### Variables relevantes de **{modelo_sel}**")
        mostrar_relevancia(pipes_reg[modelo_sel])

    # ---------------- CLASIFICACION ----------------
    with tab_clf:
        st.markdown("Clasificamos `riesgo_alto` (clase positiva ~15%). "
                    "Usamos `class_weight=\"balanced\"` por el desbalance.")
        Xtr, Xte, ytr, yte = train_test_split(X, y_clf, test_size=0.2,
                                              random_state=RANDOM_STATE, stratify=y_clf)

        @st.cache_data
        def evaluar_clasificacion():
            modelos = {
                "Regresion logistica": LogisticRegression(
                    max_iter=1000, class_weight="balanced"),
                "Arbol de decision": DecisionTreeClassifier(
                    max_depth=6, min_samples_leaf=20, class_weight="balanced",
                    random_state=RANDOM_STATE),
                "Random Forest": RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                    n_jobs=-1, random_state=RANDOM_STATE),
            }
            filas, pipes = [], {}
            for nombre, modelo in modelos.items():
                pipe = Pipeline([("prep", pp.construir_preprocesador()),
                                 ("model", modelo)])
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                filas.append({
                    "Modelo": nombre,
                    "Accuracy": accuracy_score(yte, pred),
                    "Precision": precision_score(yte, pred, zero_division=0),
                    "Recall": recall_score(yte, pred, zero_division=0),
                    "F1": f1_score(yte, pred, zero_division=0),
                })
                pipes[nombre] = pipe
            return pd.DataFrame(filas).set_index("Modelo"), pipes

        tabla_clf, pipes_clf = evaluar_clasificacion()
        st.dataframe(tabla_clf.round(3), width="stretch")
        st.caption("En un problema desbalanceado, una **accuracy** alta puede "
                   "engañar. El **recall** mide cuantos riesgos altos detectamos; "
                   "el **F1** equilibra precision y recall.")

        mejor = tabla_clf["F1"].idxmax()
        st.success(f"Mejor modelo por F1: **{mejor}** "
                   f"(F1={tabla_clf.loc[mejor, 'F1']:.3f}, "
                   f"Recall={tabla_clf.loc[mejor, 'Recall']:.3f}).")

        st.markdown("### Explorar un modelo")
        modelo_sel_clf = st.selectbox("Selecciona el modelo de clasificacion:",
                                      list(pipes_clf.keys()),
                                      index=list(pipes_clf.keys()).index(mejor),
                                      key="sel_clf")
        pipe_clf = pipes_clf[modelo_sel_clf]

        st.markdown(f"#### Variables relevantes de **{modelo_sel_clf}**")
        mostrar_relevancia(pipe_clf)

        st.markdown(f"#### Matriz de confusion de **{modelo_sel_clf}**")
        pred = pipe_clf.predict(Xte)
        cm = confusion_matrix(yte, pred)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.pyplot(plots.fig_matriz_confusion(cm))
        with c2:
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
- **Verdaderos negativos (TN):** {tn} — riesgo normal bien clasificado.
- **Falsos positivos (FP):** {fp} — normales marcados como riesgo alto.
- **Falsos negativos (FN):** {fn} — riesgos altos **no detectados** (lo mas costoso).
- **Verdaderos positivos (TP):** {tp} — riesgos altos detectados.

En seguros conviene **minimizar los falsos negativos** (no detectar un riesgo
alto), por eso vigilamos el **recall**.
            """)


# ===========================================================================
# 5. PCA
# ===========================================================================
elif seccion == SECCIONES[4]:
    st.title("Reduccion de dimensionalidad")
    st.markdown("""
Aplicamos un analisis de componentes principales de 2 componentes sobre la matriz de caracteristicas
preprocesada para visualizar las polizas en un plano y observar si los grupos
de riesgo se separan.
    """)

    X = df_feat[pp.columnas_features()]
    prep = pp.construir_preprocesador()
    Xt = prep.fit_transform(X, df_feat[pp.TARGET_CLF])
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    comp = pca.fit_transform(Xt)
    var = pca.explained_variance_ratio_

    color_por = st.radio("Observar por:", ["riesgo_alto", "clase_costo"],
                         horizontal=True)
    etiquetas = df_feat[color_por].values
    st.pyplot(plots.fig_pca(comp, etiquetas, nombre_etiqueta=color_por))

    c1, c2, c3 = st.columns(3)
    c1.metric("Varianza CP1", f"{var[0]*100:.1f}%")
    c2.metric("Varianza CP2", f"{var[1]*100:.1f}%")
    c3.metric("Varianza acumulada", f"{var.sum()*100:.1f}%")
    st.caption(f"Las 2 primeras componentes explican {var.sum()*100:.1f}% de la "
               "varianza. Si la separacion no es nitida, significa que el riesgo "
               "depende de combinaciones no lineales de variables (de ahi que los "
               "modelos de arboles funcionen mejor que los lineales).")


# ===========================================================================
# 6. SIMULADOR ACTUARIAL
# ===========================================================================
elif seccion == SECCIONES[5]:
    st.title("Simulador actuarial")
    st.markdown("Ingresa los datos de una poliza nueva para estimar su "
                "costo esperado anual y su probabilidad de riesgo alto.")

    modelo_reg, modelo_clf = obtener_modelos()

    with st.form("form_simulador"):
        c1, c2, c3 = st.columns(3)
        # Columna 1: datos del conductor (9 campos)
        with c1:
            st.markdown("##### Conductor")
            edad_conductor = st.slider("Edad del conductor", 18, 80, 40)
            sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "No especificado"])
            estado_civil = st.selectbox("Estado civil", ["Soltero", "Casado", "Divorciado", "Viudo"])
            nivel_estudios = st.selectbox("Nivel de estudios",
                ["Secundaria", "Preparatoria", "Licenciatura", "Posgrado"], index=2)
            ocupacion = st.selectbox("Ocupacion",
                ["Empleado", "Independiente", "Estudiante", "Profesor", "Empresario", "Otro"])
            zona_residencia = st.selectbox("Zona de residencia", ["Urbana", "Suburbana", "Rural"])
            ingreso_mensual_mxn = st.number_input("Ingreso mensual (MXN)", 1000, 400000, 18000, step=500)
            score_crediticio = st.slider("Score crediticio", 300, 850, 600)
            antiguedad_cliente_anios = st.slider("Antiguedad como cliente (anios)", 0.0, 30.0, 5.0)
        # Columna 2: vehiculo y uso (9 campos)
        with c2:
            st.markdown("##### Vehiculo")
            tipo_vehiculo = st.selectbox("Tipo de vehiculo",
                ["Sedan", "SUV", "Compacto", "Pickup", "Deportivo"])
            uso_vehiculo = st.selectbox("Uso del vehiculo", ["Particular", "Trabajo", "Taxi/Plataforma"])
            segmento_marca = st.selectbox("Segmento de marca", ["Economico", "Medio", "Premium"])
            edad_vehiculo_anios = st.slider("Edad del vehiculo (anios)", 0.0, 25.0, 5.0)
            km_anuales = st.number_input("Km anuales", 1000, 60000, 12000, step=500)
            suma_asegurada_mxn = st.number_input("Suma asegurada (MXN)", 50000, 2000000, 300000, step=10000)
            tiene_gps = st.selectbox("Tiene GPS", ["Si", "No"])
            asistencia_vial = st.selectbox("Asistencia vial", ["Si", "No"])
            mantenimiento_al_dia = st.selectbox("Mantenimiento al dia", ["Si", "No"])
        # Columna 3: poliza e historial (9 campos)
        with c3:
            st.markdown("##### Poliza e historial")
            prima_mensual_mxn = st.number_input("Prima mensual (MXN)", 100, 5000, 700, step=50)
            deducible_pct = st.selectbox("Deducible (%)", [3, 5, 10, 15, 20], index=1)
            metodo_pago = st.selectbox("Metodo de pago", ["Mensual", "Trimestral", "Anual"])
            canal_venta = st.selectbox("Canal de venta", ["Agente", "Online", "Banco", "Broker"])
            region = st.selectbox("Region", ["Centro", "Occidente", "Norte", "Bajio", "Sur"])
            dias_hasta_renovacion = st.slider("Dias hasta renovacion", 0, 365, 120)
            puntaje_riesgo_zona = st.slider("Puntaje riesgo de zona", 0.0, 100.0, 60.0)
            historial_siniestros_3_anios = st.slider("Siniestros previos (3 anios)", 0, 10, 1)
            numero_siniestros_12m = st.slider("Siniestros en 12 meses", 0, 8, 0)

        umbral = st.slider("Umbral de decision para riesgo alto", 0.05, 0.95, 0.50, 0.05)
        enviado = st.form_submit_button("Calcular prediccion", type="primary")

    if enviado:
        registro = {
            "poliza_id": "SIMULADA",
            "edad_conductor": edad_conductor, "sexo": sexo, "estado_civil": estado_civil,
            "nivel_estudios": nivel_estudios, "ocupacion": ocupacion,
            "zona_residencia": zona_residencia, "region": region,
            "antiguedad_cliente_anios": antiguedad_cliente_anios,
            "ingreso_mensual_mxn": ingreso_mensual_mxn, "score_crediticio": score_crediticio,
            "prima_mensual_mxn": prima_mensual_mxn, "suma_asegurada_mxn": suma_asegurada_mxn,
            "deducible_pct": deducible_pct,
            "historial_siniestros_3_anios": historial_siniestros_3_anios,
            "km_anuales": km_anuales, "edad_vehiculo_anios": edad_vehiculo_anios,
            "tipo_vehiculo": tipo_vehiculo, "uso_vehiculo": uso_vehiculo,
            "segmento_marca": segmento_marca, "metodo_pago": metodo_pago,
            "canal_venta": canal_venta, "tiene_gps": tiene_gps,
            "asistencia_vial": asistencia_vial, "mantenimiento_al_dia": mantenimiento_al_dia,
            "dias_hasta_renovacion": dias_hasta_renovacion,
            "puntaje_riesgo_zona": puntaje_riesgo_zona,
            "numero_siniestros_12m": numero_siniestros_12m,
        }
        entrada = pd.DataFrame([registro])
        entrada = pp.preparar_features(entrada)
        X_new = entrada[pp.columnas_features()]

        costo = float(modelo_reg.predict(X_new)[0])
        proba = float(modelo_clf.predict_proba(X_new)[0, 1])
        clase = int(proba >= umbral)

        st.markdown("### Resultado de la simulacion")
        c1, c2, c3 = st.columns(3)
        c1.metric("Costo esperado anual", f"${costo:,.0f} MXN")
        c2.metric("Probabilidad de riesgo alto", f"{proba*100:.1f}%")
        c3.metric("Clasificacion", "RIESGO ALTO" if clase else "Riesgo normal")

        if clase:
            st.error(f"Con umbral {umbral:.2f}, la poliza se clasifica como "
                     f"**riesgo alto** (probabilidad {proba*100:.1f}%).")
        else:
            st.success(f"Con umbral {umbral:.2f}, la poliza se clasifica como "
                       f"**riesgo normal** (probabilidad {proba*100:.1f}%).")
        st.caption("Ajusta el umbral para volver el modelo mas sensible (mayor "
                   "recall) o mas conservador (mayor precision).")


# ===========================================================================
# 7. IMAGENES
# ===========================================================================
elif seccion == SECCIONES[6]:
    st.title("Procesamiento de imagenes con OpenCV")
    st.markdown("""
Carga una imagen de un vehiculo o evidencia de siniestro y aplica las
transformaciones con la libreria OpenCV
(`cv2`).
    """)

    archivo = st.file_uploader("Sube una imagen (JPG / PNG)",
                               type=["jpg", "jpeg", "png"])
    if archivo is None:
        st.info("Esperando una imagen para mostrar las transformaciones...")
        st.stop()

    # Carga (receta 8.1): los bytes subidos se decodifican con cv2.imdecode.
    # OpenCV trabaja en BGR; convertimos a RGB para mostrar y a gris para las
    # recetas que operan sobre una sola intensidad por pixel.
    file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    st.markdown("### Imagen original")
    st.image(img_rgb, width=350)
    st.caption(f"Resolucion: {w} x {h} pixeles. Internamente la imagen es una "
               "matriz de NumPy (alto x ancho x 3 canales BGR/RGB).")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Transformaciones",
        "Realce y color",
        "Deteccion",
        "Caracteristicas para ML",
    ])

    # -------------------------------------------------------------------
    # TAB 1: transformaciones geometricas y filtros basicos (8.3 - 8.6)
    # -------------------------------------------------------------------
    with tab1:
        c1, c2 = st.columns(2)

        # 8.3 Resize
        with c1:
            st.markdown("**Resize (cv2.resize) - 200x200**")
            st.image(cv2.resize(img_rgb, (200, 200)), width="stretch")
            st.caption("Estandariza el tamano de entrada.")

        # 8.4 Crop (recorte por slicing del array)
        with c2:
            st.markdown("**Crop central (slicing del array)**")
            izq, arr_y = w // 4, h // 4
            st.image(img_rgb[arr_y:h - arr_y, izq:w - izq], width="stretch")
            st.caption("Recortar.")

        c3, c4 = st.columns(2)

        # 8.5 Blur (kernel promedio)
        with c3:
            k = st.slider("Tamano del kernel de blur", 1, 50, 5)
            st.markdown("**Blur (cv2.blur)** — promedia cada pixel con sus vecinos.")
            st.image(cv2.blur(img_rgb, (k, k)), width="stretch")
            st.caption(f"Kernel {k}x{k}: a mayor kernel, mas suavizado.")

        # 8.6 Sharpen (kernel que resalta el pixel central)
        with c4:
            st.markdown("**Sharpening (cv2.filter2D)** — resalta bordes y detalles.")
            kernel_sharp = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])
            st.image(cv2.filter2D(img_rgb, -1, kernel_sharp), width="stretch")
            st.caption("enfatiza el pixel central.")

    # -------------------------------------------------------------------
    # TAB 2: realce de contraste y aislamiento de color (8.7 - 8.8)
    # -------------------------------------------------------------------
    with tab2:
        c1, c2 = st.columns(2)

        # 8.7 Enhancing contrast (ecualizacion de histograma sobre el canal Y)
        with c1:
            st.markdown("**Contraste (ecualizacion de histograma)**")
            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            st.image(img_eq, width="stretch")
            st.caption("Usa un rango mas "
                       "amplio de intensidades y hace resaltar objetos y formas.")

        # 8.8 Isolating colors (mascara en espacio HSV)
        with c2:
            color = st.selectbox("Color a aislar (HSV)", ["Azul", "Verde", "Rojo"])
            st.markdown("**Aislar color (mascara HSV + cv2.inRange)**")
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            if color == "Azul":
                mask = cv2.inRange(img_hsv, np.array([100, 50, 50]),
                                   np.array([140, 255, 255]))
            elif color == "Verde":
                mask = cv2.inRange(img_hsv, np.array([40, 50, 50]),
                                   np.array([80, 255, 255]))
            else:  # Rojo: ocupa los dos extremos del tono (hue) -> dos mascaras
                m1 = cv2.inRange(img_hsv, np.array([0, 50, 50]),
                                 np.array([10, 255, 255]))
                m2 = cv2.inRange(img_hsv, np.array([160, 50, 50]),
                                 np.array([180, 255, 255]))
                mask = cv2.bitwise_or(m1, m2)
            img_mask = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
            st.image(img_mask, width="stretch")
            st.caption("Se define un rango de tono/saturacion/valor "
                       "y se aplica una mascara para conservar solo ese color.")

    # -------------------------------------------------------------------
    # TAB 3: deteccion (binarizacion, bordes y esquinas) (8.9, 8.11, 8.12)
    # -------------------------------------------------------------------
    with tab3:
        c1, c2 = st.columns(2)

        # 8.9 Binarizing (umbralizado adaptativo)
        with c1:
            vecindad = st.slider("Tamano de vecindad (impar)", 3, 99, 11, step=2)
            st.markdown("**Binarizar (cv2.adaptiveThreshold)**")
            img_bin = cv2.adaptiveThreshold(
                img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, vecindad, 10)
            st.image(img_bin, width="stretch")
            st.caption("Cada pixel se vuelve blanco/negro segun la intensidad de sus "
                       "vecinos. Util para denoising y aislar texto/contornos.")

        # 8.11 Detecting edges (Canny con umbrales automaticos)
        with c2:
            st.markdown("**Bordes (detector de Canny)**")
            mediana = np.median(img_gray)
            low = int(max(0, (1.0 - 0.33) * mediana))
            high = int(min(255, (1.0 + 0.33) * mediana))
            img_canny = cv2.Canny(img_gray, low, high)
            st.image(img_canny, width="stretch")
            st.caption(f"Umbrales automaticos low={low}, high={high} (a +/-1 desviacion "
                       "de la mediana). Los bordes son zonas de alta informacion.")

        # 8.12 Detecting corners (detector de Harris)
        st.markdown("**Esquinas (detector de Harris, cv2.cornerHarris)**")
        gray_f = np.float32(img_gray)
        respuestas = cv2.cornerHarris(gray_f, blockSize=2, ksize=29, k=0.04)
        respuestas = cv2.dilate(respuestas, None)
        img_corners = img_rgb.copy()
        img_corners[respuestas > 0.02 * respuestas.max()] = [255, 0, 0]
        st.image(img_corners, width=350)
        st.caption("Las esquinas (interseccion de dos bordes) se marcan en rojo; son "
                   "puntos de alta informacion para vision por computadora.")

    # -------------------------------------------------------------------
    # TAB 4: caracteristicas para machine learning (8.13 - 8.15)
    # -------------------------------------------------------------------
    with tab4:
        st.markdown("Las imagenes tambien se pueden convertir en"
                    "features para alimentar un modelo de Machine Learning.")

        # 8.14 Mean color as a feature
        st.markdown("**Color medio como caracteristica (cv2.mean)**")
        canales = cv2.mean(img_bgr)   # (B, G, R, alpha)
        media_rgb = (canales[2], canales[1], canales[0])
        c1, c2, c3 = st.columns(3)
        c1.metric("Rojo medio", f"{media_rgb[0]:.1f}")
        c2.metric("Verde medio", f"{media_rgb[1]:.1f}")
        c3.metric("Azul medio", f"{media_rgb[2]:.1f}")
        swatch = np.zeros((60, 240, 3), dtype=np.uint8)
        swatch[:, :] = [int(media_rgb[0]), int(media_rgb[1]), int(media_rgb[2])]
        st.image(swatch, caption="Color promedio de la imagen", width=240)

        # 8.13 Creating features for ML (aplanar la imagen a un vector)
        st.markdown("**Imagen como vector (flatten)**")
        small_gray = cv2.resize(img_gray, (10, 10))
        small_color = cv2.resize(img_rgb, (10, 10))
        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.resize(small_gray, (120, 120), interpolation=cv2.INTER_NEAREST),
                     caption="Reducida a 10x10 (gris)", width=120)
        with c2:
            st.markdown(f"""
- 10x10 en gris -> vector de **{small_gray.flatten().shape[0]}** valores.
- 10x10 a color -> vector de **{small_color.flatten().shape[0]}** valores.
- Esta imagen completa en gris -> **{img_gray.flatten().shape[0]:,}** valores.
- Esta imagen completa a color -> **{img_rgb.flatten().shape[0]:,}** valores.
            """)
        st.caption("Al crecer la imagen, el numero de caracteristicas explota: por eso "
                   "luego se aplican tecnicas de reduccion de dimensionalidad (PCA).")

        # 8.15 Encoding color histograms as features
        st.markdown("**Histograma de color (cv2.calcHist)**")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3))
        for i, color in enumerate(["red", "green", "blue"]):
            hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=color)
        ax.set_xlim([0, 256])
        ax.set_title("Histograma de intensidades por canal de color")
        ax.set_xlabel("Intensidad (0-255)")
        ax.set_ylabel("Numero de pixeles")
        ax.legend()
        st.pyplot(fig)
        st.caption("256 valores por canal (768 en total) que describen la distribucion "
                   "de colores; sirven como caracteristicas para clasificar imagenes.")


# ===========================================================================
# 8. CONCLUSIONES
# ===========================================================================
elif seccion == SECCIONES[7]:
    st.title("Conclusiones")
    st.markdown("""
### Hallazgos principales
- El costo esperado anual estaba dominado por unas ~10 polizas con costos
  catastroficos (hasta 131,385 MXN). Tras acotarlos al percentil 99 y
  agregar variables de interaccion (`siniestros_x_suma`, `siniestros_x_prima`),
  el R² sube de un promedio de los modelos ~0.42 a en promedio ~0.75.
- Las variables mas influyentes son `suma_asegurada_mxn`, `numero_siniestros_12m`
  y sus interacciones, seguidas de `prima_mensual_mxn`.
- En **clasificacion de riesgo alto**, dado el desbalance (~15%), la *accuracy*
  por si sola es engañosa: priorizamos **recall** y **F1** y usamos
  `class_weight="balanced"`.

### Limitaciones
- El clasificador tiene margen de presición que se puede equivocar debido al desbalance entre los 
    casos de riesgo alto y riesgo bajo.
- Pocos datos debido a la base sintética. 

### Posibles mejoras
- Ajuste de hiperparametros (GridSearch / validacion cruzada).
- Modelos de **boosting** (GradientBoosting, XGBoost) y calibracion de probabilidades.
- Tecnicas de balanceo (SMOTE) y analisis de equidad (fairness) por subgrupos.
    """)

