import json
import os
import sqlite3

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ==============================
# 0. PARAMETROS Y SUPUESTOS
# ==============================
RUTA_CSV = "sales_data_sample.csv"
RATIO_COSTO = 0.60                 # supuesto: costo = 60% del precio sugerido (MSRP)
ESTADOS_EXCLUIDOS = ["Cancelled"]  # pedidos cancelados no son ventas reales

os.makedirs("fuentes", exist_ok=True)
os.makedirs("salida", exist_ok=True)

# ------------------------------------------
# 0.1 PREPARAR LAS 3 FUENTES A PARTIR DEL CSV
# (el dataset es uno solo; se separa en CSV, JSON y SQLite
#  para simular un escenario con multiples fuentes)
# ------------------------------------------
raw = pd.read_csv(RUTA_CSV, encoding="latin-1", keep_default_na=False,
                  na_values=[""])  # "NA" (Norteamerica) se conserva como texto

productos = raw[["PRODUCTCODE", "PRODUCTLINE", "MSRP"]].drop_duplicates()
with open("fuentes/productos.json", "w", encoding="utf-8") as f:
    json.dump(productos.to_dict(orient="records"), f, indent=2)

clientes = raw[["CUSTOMERNAME", "CONTACTFIRSTNAME", "CONTACTLASTNAME", "PHONE",
                "ADDRESSLINE1", "CITY", "STATE", "POSTALCODE", "COUNTRY",
                "TERRITORY"]].drop_duplicates("CUSTOMERNAME")
if os.path.exists("fuentes/clientes.db"):
    os.remove("fuentes/clientes.db")
with sqlite3.connect("fuentes/clientes.db") as con:
    clientes.to_sql("clientes", con, index=False)

# ------------------------------------------
# 1. CREAR SESION SPARK
# ------------------------------------------
spark = SparkSession.builder \
    .appName("ETL_Ventas_Retail") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ==============================
# 2. EXTRACT (3 fuentes)
# ==============================
# Fuente 1: CSV
df_ventas = spark.read.csv(RUTA_CSV, header=True, inferSchema=True,
                           encoding="ISO-8859-1") \
    .select("ORDERNUMBER", "ORDERLINENUMBER", "ORDERDATE", "STATUS",
            "QUANTITYORDERED", "PRICEEACH", "SALES", "PRODUCTCODE",
            "CUSTOMERNAME", "DEALSIZE")

# Fuente 2: JSON
df_prod = spark.read.option("multiLine", True).json("fuentes/productos.json")

# Fuente 3: SQL (SQLite)
with sqlite3.connect("fuentes/clientes.db") as con:
    pdf = pd.read_sql_query("SELECT * FROM clientes", con)
df_cli = spark.createDataFrame(pdf.astype(str))

print("==== Ventas (CSV) ====")
df_ventas.show(5, truncate=False)
print("==== Productos (JSON) ====")
df_prod.show(5, truncate=False)
print("==== Clientes (SQL) ====")
df_cli.show(5, truncate=False)

# ==============================
# 2.1 AUDITORIA (conteos ANTES de transformar)
# ==============================
n_leidas = df_ventas.count()
n_sin_dup_clave = df_ventas.dropDuplicates(["ORDERNUMBER", "ORDERLINENUMBER"]).count()
n_sin_dup_exacta = df_ventas.dropDuplicates().count()
n_venta_invalida = df_ventas.filter(F.col("SALES").isNull() | (F.col("SALES") <= 0)).count()
n_cant_invalida = df_ventas.filter(F.col("QUANTITYORDERED").isNull() | (F.col("QUANTITYORDERED") <= 0)).count()
n_nombre_espacios = df_ventas.filter(F.col("CUSTOMERNAME") != F.trim(F.col("CUSTOMERNAME"))).count()
n_excluidos_status = df_ventas.filter(F.col("STATUS").isin(ESTADOS_EXCLUIDOS)).count()

n_cli_leidos = df_cli.count()
n_cli_usa = df_cli.filter(F.col("COUNTRY") == "USA").count()
n_cli_uk = df_cli.filter(F.col("COUNTRY") == "UK").count()
n_cli_na = df_cli.filter(F.col("TERRITORY") == "NA").count()
n_cli_japan = df_cli.filter(F.col("TERRITORY") == "Japan").count()
n_cli_state_nulo = df_cli.filter(F.col("STATE").isNull() | F.col("STATE").isin("None", "nan", "")).count()
n_cli_dup = n_cli_leidos - df_cli.dropDuplicates(["CUSTOMERNAME"]).count()
n_prod_leidos = df_prod.count()
n_prod_dup = n_prod_leidos - df_prod.dropDuplicates(["PRODUCTCODE"]).count()

# ==============================
# 3. TRANSFORM
# ==============================

# --- Clientes ---
# Textos nulos que genera pandas ("None", "nan", "") -> null real
for c in df_cli.columns:
    df_cli = df_cli.withColumn(
        c, F.when(F.col(c).isin("None", "nan", ""), F.lit(None)).otherwise(F.col(c)))

df_cli = df_cli.dropDuplicates(["CUSTOMERNAME"]) \
    .withColumn("CUSTOMERNAME", F.trim(F.col("CUSTOMERNAME"))) \
    .withColumn("STATE", F.coalesce(F.col("STATE"), F.lit("N/D"))) \
    .withColumn("COUNTRY",
                F.when(F.col("COUNTRY") == "USA", "United States")
                .when(F.col("COUNTRY") == "UK", "United Kingdom")
                .otherwise(F.trim(F.col("COUNTRY")))) \
    .withColumn("REGION",
                F.when(F.col("TERRITORY") == "NA", "Norteamerica")
                .when(F.col("TERRITORY") == "Japan", "Japon")
                .otherwise(F.coalesce(F.col("TERRITORY"), F.lit("Sin region"))))

# --- Productos ---
df_prod = df_prod.dropDuplicates(["PRODUCTCODE"]) \
    .withColumnRenamed("PRODUCTLINE", "CATEGORIA") \
    .withColumn("COSTO_UNIT", F.round(F.col("MSRP") * RATIO_COSTO, 2))

# --- Ventas ---
# Nota: DEALSIZE (Small/Medium/Large) se usa como proxy de "canal" porque el
# dataset no trae canal de venta. Se nombra TAMANO_PEDIDO para ser honestos.
df_ventas = df_ventas.dropDuplicates(["ORDERNUMBER", "ORDERLINENUMBER"]) \
    .withColumn("CUSTOMERNAME", F.trim(F.col("CUSTOMERNAME"))) \
    .filter(~F.col("STATUS").isin(ESTADOS_EXCLUIDOS)) \
    .filter(F.col("SALES").isNotNull() & (F.col("SALES") > 0)) \
    .filter(F.col("QUANTITYORDERED") > 0) \
    .withColumn("FECHA", F.to_date(F.to_timestamp(F.col("ORDERDATE"), "M/d/yyyy H:mm"))) \
    .withColumnRenamed("DEALSIZE", "TAMANO_PEDIDO")

# --- Unir las 3 fuentes ---
df_final = df_ventas \
    .join(df_prod, "PRODUCTCODE", "left") \
    .join(df_cli.select("CUSTOMERNAME", "CITY", "STATE", "POSTALCODE",
                        "COUNTRY", "TERRITORY", "REGION"),
          "CUSTOMERNAME", "left")

# --- Columnas calculadas ---
df_final = df_final \
    .withColumn("COSTO_TOTAL", F.round(F.col("QUANTITYORDERED") * F.col("COSTO_UNIT"), 2)) \
    .withColumn("UTILIDAD", F.round(F.col("SALES") - F.col("COSTO_TOTAL"), 2)) \
    .withColumn("DESCUENTO_PCT",
                F.round(F.greatest(F.lit(0.0),
                                   (F.col("MSRP") - F.col("PRICEEACH")) / F.col("MSRP")), 4)) \
    .withColumn("ANIO", F.year("FECHA")) \
    .withColumn("MES", F.month("FECHA")) \
    .withColumn("TRIMESTRE", F.quarter("FECHA")) \
    .withColumn("ANIO_MES", F.date_format("FECHA", "yyyy-MM")) \
    .drop("ORDERDATE")

print("==== Dataset final (5 registros) ====")
df_final.show(5, truncate=False)

# ==============================
# 4. VALIDACIONES
# ==============================
n_final = df_final.count()
n_fecha_nula = df_final.filter(F.col("FECHA").isNull()).count()
n_sin_producto = df_final.filter(F.col("CATEGORIA").isNull()).count()
n_sin_cliente = df_final.filter(F.col("COUNTRY").isNull()).count()
n_utilidad_neg = df_final.filter(F.col("UTILIDAD") < 0).count()
fechas = df_final.agg(F.min("FECHA").alias("min"), F.max("FECHA").alias("max")).first()

print("Filas finales        :", n_final)
print("Ventas totales       :", df_final.agg(F.sum("SALES")).first()[0])
print("Fechas nulas         :", n_fecha_nula)
print("Rango de fechas      :", fechas["min"], "->", fechas["max"])
print("Sin producto (join)  :", n_sin_producto)
print("Sin cliente (join)   :", n_sin_cliente)
print("Lineas con utilidad negativa (revisar RATIO_COSTO):", n_utilidad_neg)

# Fallan de forma explicita si algo critico se rompio
assert n_fecha_nula == 0, "Hay fechas nulas: revisar el formato de ORDERDATE"
assert n_sin_producto == 0, "Hay ventas sin producto: revisar el join de productos"
assert n_sin_cliente == 0, "Hay ventas sin cliente: revisar el join de clientes"

# Consultas de control
print("==== Ventas por pais ====")
df_final.groupBy("COUNTRY").agg(F.round(F.sum("SALES"), 2).alias("total")) \
    .orderBy(F.col("total").desc()).show(5)

print("==== Ventas por categoria ====")
df_final.groupBy("CATEGORIA").agg(F.round(F.sum("SALES"), 2).alias("total")) \
    .orderBy(F.col("total").desc()).show()

print("==== Ventas por anio (2005 es parcial) ====")
df_final.groupBy("ANIO").agg(F.round(F.sum("SALES"), 2).alias("total"),
                             F.countDistinct("MES").alias("meses")) \
    .orderBy("ANIO").show()

# ==============================
# 4.1 REPORTE DE AUDITORIA
# ==============================
# Nota: productos y clientes ya vienen sin duplicados desde la preparacion de
# fuentes; por eso esos conteos suelen ser 0 (el transform igual los elimina).
filas = [
    ("VENTAS (CSV)", "Filas leidas", n_leidas),
    ("VENTAS (CSV)", "Duplicados exactos (fila completa repetida)", n_leidas - n_sin_dup_exacta),
    ("VENTAS (CSV)", "Duplicados por clave (pedido + linea)", n_leidas - n_sin_dup_clave),
    ("VENTAS (CSV)", "Pedidos excluidos por STATUS (" + ", ".join(ESTADOS_EXCLUIDOS) + ")", n_excluidos_status),
    ("VENTAS (CSV)", "Filas con ventas nulas o <= 0", n_venta_invalida),
    ("VENTAS (CSV)", "Filas con cantidad nula o <= 0", n_cant_invalida),
    ("VENTAS (CSV)", "TOTAL filas eliminadas (sin doble conteo)", n_leidas - n_final),
    ("VENTAS (CSV)", "Filas finales", n_final),
    ("VENTAS (CSV)", "Valores modificados: CUSTOMERNAME con espacios (trim)", n_nombre_espacios),
    ("VENTAS (CSV)", "Valores modificados: ORDERDATE texto -> fecha", n_final),
    ("VENTAS (CSV)", "Fechas nulas tras la conversion", n_fecha_nula),
    ("CLIENTES (SQL)", "Filas leidas", n_cli_leidos),
    ("CLIENTES (SQL)", "Duplicados eliminados", n_cli_dup),
    ("CLIENTES (SQL)", "Valores modificados: COUNTRY 'USA' -> 'United States'", n_cli_usa),
    ("CLIENTES (SQL)", "Valores modificados: COUNTRY 'UK' -> 'United Kingdom'", n_cli_uk),
    ("CLIENTES (SQL)", "Valores modificados: TERRITORY 'NA' -> 'Norteamerica'", n_cli_na),
    ("CLIENTES (SQL)", "Valores modificados: TERRITORY 'Japan' -> 'Japon'", n_cli_japan),
    ("CLIENTES (SQL)", "Valores nulos en STATE rellenados con 'N/D'", n_cli_state_nulo),
    ("PRODUCTOS (JSON)", "Filas leidas", n_prod_leidos),
    ("PRODUCTOS (JSON)", "Duplicados eliminados", n_prod_dup),
    ("CALCULADOS", "Lineas con utilidad negativa (supuesto de costo)", n_utilidad_neg),
]
print("==== REPORTE DE AUDITORIA DE CALIDAD ====")
for fuente, control, cant in filas:
    print(f"{fuente:17s} | {control:58s} | {cant}")

pd.DataFrame(filas, columns=["fuente", "control", "cantidad"]) \
    .to_csv("salida/auditoria_calidad.csv", index=False, encoding="utf-8-sig")

# ==============================
# 5. LOAD
# ==============================
# Tabla plana (Parquet) para consumo en Spark
df_final.write.mode("overwrite").parquet("salida/ventas_limpio_parquet")

# CSV unico, listo para Power BI (el dataset es pequeno, cabe en memoria)
df_pd = df_final.toPandas()
df_pd.to_csv("salida/ventas_limpio.csv", index=False, encoding="utf-8-sig")

# Dimensiones para el modelo estrella en Power BI
df_pd[["PRODUCTCODE", "CATEGORIA", "MSRP", "COSTO_UNIT"]].drop_duplicates("PRODUCTCODE") \
    .to_csv("salida/dim_producto.csv", index=False, encoding="utf-8-sig")
df_pd[["CUSTOMERNAME", "CITY", "STATE", "POSTALCODE", "COUNTRY", "TERRITORY", "REGION"]] \
    .drop_duplicates("CUSTOMERNAME") \
    .to_csv("salida/dim_cliente.csv", index=False, encoding="utf-8-sig")

print("ETL finalizado: resultados en la carpeta 'salida'")
spark.stop()