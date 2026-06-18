# Sistema actuarial de prediccion de riesgo y costo esperado en seguros de automovil

App de predicción de costos para una aseguradora

Aplicacion que permite explorar, limpiar, modelar y simular una cartera de
polizas de seguro de automovil. Resuelve dos problemas:

- **Regresion:** estimar el `costo_esperado_anual_mxn` de una poliza.
- **Clasificacion:** predecir `riesgo_alto` (clase positiva ~15%, desbalanceada).

## Estructura del proyecto

```
entregable/
├── app.py                  # Aplicacion Streamlit (8 secciones)
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

##  COMO EJECUTAR EL PROYECTO CON ANACONDA (PASO A PASO)

pasos a seguir para la ejecucion rapida:
Se debe de usar Anaconda PowerShell

```bash
0) Descargar todo el repositorio de Github y descomprirlo donde se quiera correr

1) Instala Anaconda o Miniconda.

2) Abre la terminal "Anaconda Prompt" o "Anaconda PowerShell".

3) Situate dentro de la carpeta donde se encuentre el proyecto con el comando cd:
       cd: ruta\hacia\del\proyecto

4) Crea el ambiente a partir del archivo environment.yml.
   (Esto crea un ambiente llamado "IA" con todas las librerias):
       conda env create -f environment.yml

   Si ya existe un ambiente "IA" y quieres reemplazarlo:
       conda env remove -n IA
       conda env create -f environment.yml

5) Activa el ambiente:
       conda activate IA

6) Ejecuta la aplicacion:
       streamlit run app.py

7) Se abrira automaticamente el navegador en:
       http://localhost:8501
   (si no abre solo, copia esa direccion en tu navegador).

8) (OPCIONAL) Cuando ya no necesites la app, puedes borrar el
   ambiente "IA" para que no ocupe espacio en disco.
   Primero desactivalo y luego eliminalo:
       conda deactivate
       conda env remove -n IA
```
