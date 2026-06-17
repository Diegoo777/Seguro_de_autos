# Sistema actuarial de prediccion de riesgo y costo esperado en seguros de automovil

Proyecto Integrador · Actuaria y Ciencia de Datos · **Machine Learning + Streamlit**

Aplicacion que permite explorar, limpiar, modelar y simular una cartera de
polizas de seguro de automovil. Resuelve dos problemas:

- **Regresion:** estimar el `costo_esperado_anual_mxn` de una poliza.
- **Clasificacion:** predecir `riesgo_alto` (clase positiva ~15%, desbalanceada).

## Estructura del proyecto

```
entregable/
├── app.py                  # Aplicacion Streamlit (8 secciones)
├── limpieza_datos.py       # Limpieza de la base -> CSV limpio
├── entrenar_modelos.py     # Entrena y guarda los modelos .joblib
├── environment.yml         # Ambiente de Anaconda
├── requirements.txt        # Dependencias (pip)
├── INSTRUCCIONES.txt       # Guia paso a paso de ejecucion
├── README.md
├── data/
│   ├── seguro_auto_actuarial.csv          # Base original
│   └── seguro_auto_actuarial_limpio.csv   # Base limpia (generada)
├── notebooks/
│   └── 01_eda_modelado.ipynb              # visualizador de procesos en notebook
├── models/
│   ├── modelo_regresion.joblib
│   └── modelo_clasificacion.joblib
├── utils/
│   ├── preprocessing.py    # Mejora de modelos
    └── plots.py            # Funciones de graficado

```

## Ejecucion rapida

pasos a seguir para la ejecucion rapida:
Se debe de usar Anaconda PowerShell

```bash
cd c:\ruta\(ruta del proyecyo)
conda env create -f environment.yml   # crea el ambiente "IA"
conda activate IA
streamlit run app.py
```

Ver **INSTRUCCIONES.txt** para la guia detallada (incluye la opcion de usar un
ambiente ya existente con `pip install -r requirements.txt`). Con mas detalles para el uso de la terminal

## Temas integrados (segun rubrica)

- **Datos numericos:** imputacion (mediana), deteccion de outliers (boxplot/IQR),
  estandarizacion, transformaciones (log), discretizacion (`grupo_edad`) y
  variables nuevas (`km_por_anio_vehiculo`, `ratio_prima_ingreso`).
- **Datos categoricos:** One-Hot (nominales), Ordinal Encoding justificado
  (ordinales), manejo de faltantes ("Desconocido"/moda), tratamiento del
  desbalance con `class_weight="balanced"`.
- **Modelos:** Regresion lineal, Ridge, Lasso, Decision Tree y Random Forest
  (regresion y clasificacion). Metricas: MAE, RMSE, R² / accuracy, precision,
  recall, F1 y matriz de confusion. La seccion *Modelado* incluye un **menu para
  seleccionar el modelo** y ver sus variables relevantes (grafica + tabla de
  coeficientes/importancias) y la matriz de confusion.
- **Tratamiento del objetivo:** el costo se acota al **percentil 99** (los ~10
  siniestros catastroficos se tratan como outliers, en la practica cubiertos por
  reaseguro) y se agregan interacciones `siniestros_x_suma` y `siniestros_x_prima`,
  lo que eleva el **R² de ~0.42 a ~0.75**.
- **PCA:** visualizacion 2D con varianza explicada.
- **Imagenes:** resize, crop, blur, sharpening, contraste, aislado de color e
  histograma RGB (Pillow).
- **Buenas practicas:** todo el preprocesamiento vive en un `Pipeline` +
  `ColumnTransformer` ajustado solo con el conjunto de entrenamiento
  (`train_test_split`, `random_state=42`) para evitar fugas de informacion.

## Nota etica

La base es **sintetica**. Variables como `sexo`, `region` u `ocupacion` pueden
inducir sesgos; cualquier uso real exige validacion, auditoria y revision de
equidad.
