"""
utils/plots.py
---------------
Funciones auxiliares de graficado (matplotlib + seaborn) reutilizadas por la
app de Streamlit y el notebook. Todas las graficas incluyen titulo y etiquetas.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def fig_histograma(df, columna, bins=30):
    """Histograma de una variable numerica."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[columna].dropna(), bins=bins, kde=True, ax=ax, color="#2c6fbb")
    ax.set_title(f"Distribucion de {columna}")
    ax.set_xlabel(columna)
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig


def fig_boxplot(df, columna):
    """Boxplot para detectar outliers de una variable numerica."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df[columna].dropna(), ax=ax, color="#7bb3e0")
    ax.set_title(f"Boxplot de {columna} (deteccion de outliers)")
    ax.set_xlabel(columna)
    fig.tight_layout()
    return fig


def fig_correlacion(df, columnas):
    """Mapa de calor de correlacion de Pearson entre variables numericas."""
    corr = df[columnas].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title("Matriz de correlacion de Pearson (variables numericas)")
    fig.tight_layout()
    return fig


def fig_balance_clases(df, columna="riesgo_alto"):
    """Grafico de barras del balance de la clase objetivo."""
    fig, ax = plt.subplots(figsize=(5, 4))
    conteo = df[columna].value_counts().sort_index()
    etiquetas = ["Riesgo normal (0)", "Riesgo alto (1)"]
    ax.bar(etiquetas[:len(conteo)], conteo.values,
           color=["#4c9f70", "#d9534f"])
    for i, v in enumerate(conteo.values):
        pct = 100 * v / conteo.sum()
        ax.text(i, v, f"{v}\n({pct:.1f}%)", ha="center", va="bottom")
    ax.set_title("Balance de la clase riesgo_alto")
    ax.set_ylabel("Numero de polizas")
    fig.tight_layout()
    return fig


def fig_pca(componentes, etiquetas, nombre_etiqueta="riesgo_alto"):
    """Dispersion de las 2 componentes principales coloreada por la clase."""
    fig, ax = plt.subplots(figsize=(7, 5))
    etiquetas = np.asarray(etiquetas)
    for valor in np.unique(etiquetas):
        mask = etiquetas == valor
        ax.scatter(componentes[mask, 0], componentes[mask, 1],
                   s=18, alpha=0.6, label=f"{nombre_etiqueta}={valor}")
    ax.set_title("Proyeccion PCA (2 componentes)")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_importancia(nombres, importancias, top=15, titulo="Importancia de variables"):
    """Grafico de barras horizontales con las variables mas importantes."""
    idx = np.argsort(importancias)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(np.array(nombres)[idx][::-1], np.array(importancias)[idx][::-1],
            color="#2c6fbb")
    ax.set_title(titulo)
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    return fig


def fig_coeficientes(nombres, coeficientes, top=15):
    """Grafico de barras con los coeficientes mas influyentes (regresion lineal)."""
    idx = np.argsort(np.abs(coeficientes))[::-1][:top]
    valores = np.array(coeficientes)[idx][::-1]
    colores = ["#d9534f" if v < 0 else "#4c9f70" for v in valores]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(np.array(nombres)[idx][::-1], valores, color=colores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Coeficientes mas influyentes (regresion lineal)")
    ax.set_xlabel("Valor del coeficiente (escala estandarizada)")
    fig.tight_layout()
    return fig


def fig_matriz_confusion(cm, etiquetas=("Normal", "Riesgo alto")):
    """Matriz de confusion como mapa de calor."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=etiquetas, yticklabels=etiquetas)
    ax.set_title("Matriz de confusion")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real")
    fig.tight_layout()
    return fig
