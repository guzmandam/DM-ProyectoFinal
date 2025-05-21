import streamlit as st
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
# 1) Arranca Spark + Delta (si usas Delta; si no, lee Parquet normal)
spark = (SparkSession.builder.
           appName("simple_demo").
           config("spark.driver.memory", "8g").
           config("spark.executor.memory", "8g").
           getOrCreate())

model = PipelineModel.load("/home/entropiadev/models/brewery_pipeline")
# 3) Lectura única del dataset (o de tu muestra)
df_full = spark.read.parquet("./brewery_sales.parquet")

df_full = (df_full
      .withColumn("ratio_arr", F.split(F.col("Ingredient_Ratio"), ":"))
      .withColumn("Malt_Ratio" , F.col("ratio_arr").getItem(0).cast("double"))
      .withColumn("Hop_Ratio"  , F.col("ratio_arr").getItem(1).cast("double"))
      .withColumn("Yeast_Ratio", F.col("ratio_arr").getItem(2).cast("double"))
      .drop("Ingredient_Ratio", "ratio_arr")
)

feature_cols = [
    "Fermentation_Time", "Temperature", "pH_Level", "Gravity",
    "Alcohol_Content", "Bitterness", "Color",
    "Malt_Ratio", "Hop_Ratio", "Yeast_Ratio",
    "Loss_During_Brewing", "Loss_During_Fermentation", "Loss_During_Bottling_Kegging"
]

df_full = df_full.withColumn("Brew_Date", F.to_date("Brew_Date"))

st.title("Demo Calidad de Lotes (Simplificado)")
chart = st.line_chart()
table = st.empty()

# 4) Bucle de simulación
while True:
    # toma un micro-lote al azar de 1–5 filas
    df_batch = df_full.orderBy(rand()).limit(5)

    # predice
    df_batch = model.transform(df_batch)
    
    # Seleccionar solo las columnas necesarias y convertir vectores a valores simples
    df_display = df_batch.select(
        "Brew_Date",
        "Fermentation_Time",
        "Temperature",
        "pH_Level",
        "Gravity",
        "Alcohol_Content",
        "Bitterness",
        "Color",
        "prediction"
    )

    # muestra en Streamlit
    pdf = df_display.toPandas()
    chart.add_rows(pdf[["prediction"]].set_index(pd.RangeIndex(start=0, stop=len(pdf))))
    table.dataframe(pdf)

    time.sleep(2)
