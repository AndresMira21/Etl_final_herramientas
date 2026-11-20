"""
ANOVA de un factor sobre el dataset limpio (salida/ventas_limpio.csv).

Pruebas:
  1. SALES               ~ CATEGORIA        (y version con log)
  2. SALES               ~ REGION           (y version con log)
  3. QUANTITYORDERED     ~ RANGO_DESCUENTO  (relacion descuento-volumen, req. 9)

Para cada una: descriptivos, ANOVA clasico, tamano del efecto (eta cuadrado),
supuestos (Shapiro sobre residuos, Levene), alternativas robustas (Welch y
Kruskal-Wallis) y post-hoc de Tukey si el resultado es significativo.

Requisitos:  pip install pandas numpy scipy statsmodels matplotlib
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

ALFA = 0.05
MIN_N = 5            # grupos con menos filas se descartan (y se avisa)
os.makedirs("salida", exist_ok=True)

# ==============================
# 1. CARGA Y PREPARACION
# ==============================
df = pd.read_csv("salida/ventas_limpio.csv", encoding="utf-8-sig")

# Mismos rangos que la columna calculada de Power BI
d = df["DESCUENTO_PCT"]
df["RANGO_DESCUENTO"] = np.select(
    [d == 0, d <= 0.10, d <= 0.20, d <= 0.30, d <= 0.40],
    ["1. Sin descuento", "2. 0-10%", "3. 10-20%", "4. 20-30%", "5. 30-40%"],
    default="6. Mas de 40%",
)
df["LOG_SALES"] = np.log(df["SALES"])   # SALES esta muy sesgada a la derecha

print(f"Filas cargadas: {len(df)}")


# ==============================
# 2. FUNCION DE ANALISIS
# ==============================
def analizar(data, y, grupo, nombre):
    print("\n" + "=" * 70)
    print(f"{nombre}:  {y} ~ {grupo}")
    print("=" * 70)

    data = data[[y, grupo]].dropna().copy()
    conteo = data[grupo].value_counts()
    pequenos = conteo[conteo < MIN_N].index.tolist()
    if pequenos:
        print(f"Grupos descartados por tener menos de {MIN_N} filas: {pequenos}")
        data = data[~data[grupo].isin(pequenos)]
    k = data[grupo].nunique()

    # --- Descriptivos ---
    desc = data.groupby(grupo)[y].agg(["count", "mean", "median", "std"]).round(3)
    print("\nDescriptivos por grupo:")
    print(desc)

    # --- ANOVA clasico ---
    modelo = ols(f"{y} ~ C({grupo})", data=data).fit()
    tabla = anova_lm(modelo, typ=2)
    ss_entre = tabla.iloc[0]["sum_sq"]
    ss_total = tabla["sum_sq"].sum()
    eta2 = ss_entre / ss_total
    F = tabla.iloc[0]["F"]
    p_anova = tabla.iloc[0]["PR(>F)"]
    print("\nANOVA clasico:")
    print(tabla.round(4))
    print(f"Eta cuadrado (tamano del efecto): {eta2:.4f}")

    # --- Supuestos ---
    _, p_shapiro = stats.shapiro(modelo.resid)
    grupos = [g[y].values for _, g in data.groupby(grupo)]
    _, p_levene = stats.levene(*grupos, center="median")
    print("\nSupuestos:")
    print(f"  Normalidad de residuos (Shapiro-Wilk) p = {p_shapiro:.4g}")
    print(f"  Homogeneidad de varianzas (Levene)    p = {p_levene:.4g}")

    # --- Alternativas robustas ---
    welch = anova_oneway(data[y], data[grupo], use_var="unequal")
    p_welch = float(welch.pvalue)
    _, p_kruskal = stats.kruskal(*grupos)
    print("\nAlternativas robustas:")
    print(f"  ANOVA de Welch      p = {p_welch:.4g}")
    print(f"  Kruskal-Wallis      p = {p_kruskal:.4g}")

    # --- Post-hoc ---
    if p_anova < ALFA:
        tukey = pairwise_tukeyhsd(data[y], data[grupo], alpha=ALFA)
        print("\nPost-hoc Tukey HSD:")
        print(tukey.summary())
        s = tukey.summary().data
        archivo = f"salida/tukey_{nombre.replace(' ', '_')}.csv"
        pd.DataFrame(s[1:], columns=s[0]).to_csv(archivo, index=False)

    # --- Graficos: boxplot + QQ de residuos ---
    orden = data.groupby(grupo)[y].median().sort_values().index.tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.boxplot([data.loc[data[grupo] == g, y] for g in orden], showfliers=True)
    ax1.set_xticklabels(orden, rotation=30, ha="right")
    ax1.set_title(f"{y} por {grupo}")
    sm.qqplot(modelo.resid, line="s", ax=ax2)
    ax2.set_title("QQ-plot de residuos")
    fig.tight_layout()
    fig.savefig(f"salida/anova_{nombre.replace(' ', '_')}.png", dpi=120)
    plt.close(fig)

    # --- Conclusion ---
    supuestos_ok = (p_shapiro >= ALFA) and (p_levene >= ALFA)
    decision = "Se RECHAZA H0" if p_anova < ALFA else "NO se rechaza H0"
    print(f"\nConclusion (alfa = {ALFA}): {decision} (p ANOVA = {p_anova:.4g}).")
    if not supuestos_ok:
        concl_rob = "Se RECHAZA H0" if p_welch < ALFA else "NO se rechaza H0"
        print("  Los supuestos clasicos no se cumplen del todo; usar como respaldo "
              f"Welch/Kruskal: {concl_rob} (Welch p = {p_welch:.4g}).")

    return {
        "prueba": nombre, "variable": y, "factor": grupo, "grupos": k,
        "n": len(data), "F": round(F, 4), "p_anova": p_anova,
        "eta2": round(eta2, 4), "p_shapiro": p_shapiro, "p_levene": p_levene,
        "p_welch": p_welch, "p_kruskal": p_kruskal,
        "supuestos_ok": supuestos_ok, "decision": decision,
    }


# ==============================
# 3. EJECUCION
# ==============================
resultados = [
    analizar(df, "SALES", "CATEGORIA", "A1 Ventas por categoria"),
    analizar(df, "LOG_SALES", "CATEGORIA", "A1b Log ventas por categoria"),
    analizar(df, "SALES", "REGION", "A2 Ventas por region"),
    analizar(df, "LOG_SALES", "REGION", "A2b Log ventas por region"),
    analizar(df, "QUANTITYORDERED", "RANGO_DESCUENTO", "A3 Cantidad por rango de descuento"),
]

res = pd.DataFrame(resultados)
res.to_csv("salida/anova_resultados.csv", index=False, encoding="utf-8-sig")
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(res[["prueba", "F", "p_anova", "eta2", "p_levene", "p_welch",
           "p_kruskal", "decision"]].to_string(index=False))
print("\nArchivos generados en 'salida': anova_resultados.csv, tukey_*.csv, anova_*.png")