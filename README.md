# Sistema actuarial de prediccion de riesgo y costo esperado en seguros de automovil

App de predicción de costos para una aseguradora

Aplicacion que permite explorar, limpiar, modelar y simular una cartera de
polizas de seguro de automovil. Resuelve dos problemas:

- **Regresion:** estimar el `costo_esperado_anual_mxn` de una poliza.
- **Clasificacion:** predecir `riesgo_alto` (clase positiva ~15%, desbalanceada).

## Estructura del proyecto

```
entregable/
├── app.py                                # Aplicacion Streamlit (8 secciones)
├── entrenar_modelos.py                   # Entrena y guarda los modelos .joblib
├── environment.yml                       # Creacion de ambiente de anaconda
├── requirements.txt                      # Dependencias (pip)
├── README.md                             # estructura y guia de ejecucion
├── data/
│   └──  seguro_auto_actuarial.csv        # Base de datos
├── notebooks/
│   └── 01_eda_modelado.ipynb             # visualizador de procesos en notebook
├── models/
│   ├── modelo_regresion.joblib           # pipeline del simulador ya entrenados
│   └── modelo_clasificacion.joblib       # pipeline del simulador ya entrenados
├── utils/
│   ├── preprocessing.py                  # Mejora de modelos
│   └── plots.py                          # Funciones de graficado
├── assets/
    └── imagenes de muestra               # imagenes de referencia

```

##  COMO EJECUTAR EL PROYECTO CON ANACONDA (PASO A PASO)

pasos a seguir para la ejecucion rapida:
Se debe de usar Anaconda PowerShell


0) Descargar todo el repositorio de Github y descomprirlo donde se quiera correr

1) Instala Anaconda o Miniconda.

2) Abre la terminal "Anaconda Prompt" o "Anaconda PowerShell".

3) Situate dentro de la carpeta donde se encuentre el proyecto con el comando cd:
       cd ruta\del\proyecto

4) Crea el ambiente a partir del archivo environment.yml.
   (Esto crea un ambiente llamado "IA" con todas las librerias):
       
    ```bash
    conda env create -f environment.yml
    ```

   Si ya existe un ambiente "IA" y quieres reemplazarlo:
    ```bash
    conda env remove -n IA
    ```

    ```bash
    conda env create -f environment.yml
    ```

5) Activa el ambiente:
    ```bash
    conda activate IA
    ```
       

7) Ejecuta la aplicacion:
    ```bash
    streamlit run app.py
    ```
       

9) Se abrira automaticamente el navegador en:
    ```bash
    http://localhost:8501
    ```
   (si no abre solo, copia esa direccion en tu navegador).

11) (OPCIONAL) Cuando ya no necesites la app, puedes borrar el
   ambiente "IA" para que no ocupe espacio en disco.
   Primero desactivalo y luego eliminalo:
    ```bash
    conda deactivate
    ```
    
    ```bash
    conda env remove -n IA
    ```
