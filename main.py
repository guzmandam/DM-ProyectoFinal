"""
Main module for brewery quality prediction pipeline.

This module handles the data loading, preprocessing, model training, 
and model saving for a machine learning pipeline that predicts 
beer quality scores based on brewing parameters.
"""

from retriever import Retriever

import os

from pyspark.sql import (
    SparkSession,
    functions as F
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StandardScaler,
    StringIndexer, 
    OneHotEncoder, 
    VectorAssembler
)
from pyspark.ml.regression import GBTRegressor

# Configuración de la aplicación Spark
# Estos parámetros determinan los recursos asignados al procesamiento
SPARK_APP_NAME = "Cadena de Producción de Cerveza"
SPARK_MEMORY = "8g"
SPARK_EXECUTOR_MEMORY = "8g"

def preprocess_data(df):
    """
    Preprocess the brewery data to prepare it for model training.
    
    Transforms include:
    - Splitting Ingredient_Ratio into separate columns
    - Converting Brew_Date to date format
    
    Args:
        df: PySpark DataFrame containing the raw brewery data
        
    Returns:
        PySpark DataFrame: The preprocessed dataframe
    """
    # Transformación de la columna "Ingredient_Ratio" (formato inicial "X:Y:Z")
    # Separamos en tres columnas individuales para cada ingrediente:
    # - Malt_Ratio: Proporción de malta
    # - Hop_Ratio: Proporción de lúpulo
    # - Yeast_Ratio: Proporción de levadura
    df = df.withColumn("ratio_arr", F.split(F.col("Ingredient_Ratio"), ":")) \
       .withColumn("Malt_Ratio" , F.col("ratio_arr").getItem(0).cast("double")) \
       .withColumn("Hop_Ratio"  , F.col("ratio_arr").getItem(1).cast("double")) \
       .withColumn("Yeast_Ratio", F.col("ratio_arr").getItem(2).cast("double")) \
       .drop("Ingredient_Ratio","ratio_arr")
    
    # Convertimos la columna "Brew_Date" a formato fecha
    # para facilitar el filtrado temporal en la división de datos
    df = df.withColumn("Brew_Date", F.to_date("Brew_Date"))

    return df

def split_data(df):
    """
    Split the data into training, validation, and test sets based on date.
    
    Training: Dates <= 2022-12-31
    Validation: Dates from 2023-01-01 to 2023-06-30
    Test: Dates >= 2023-07-01
    
    Args:
        df: PySpark DataFrame containing the preprocessed brewery data
        
    Returns:
        tuple: (train_df, val_df, test_df) - The three split dataframes
    """
    # División temporal de los datos:
    # - Datos históricos (hasta 2022) para entrenamiento
    train_df = df.filter(F.col("Brew_Date") <= F.lit("2022-12-31"))
    
    # - Primer semestre 2023 para validación
    val_df   = df.filter((F.col("Brew_Date") >= F.lit("2023-01-01")) &
                        (F.col("Brew_Date") <= F.lit("2023-06-30")))
    
    # - Segundo semestre 2023 para pruebas
    test_df  = df.filter(F.col("Brew_Date") >= F.lit("2023-07-01"))

    return train_df, val_df, test_df

def train_model(train_df, val_df, test_df):
    """
    Build and train a machine learning pipeline for beer quality prediction.
    
    The pipeline includes:
    - String indexing and one-hot encoding for categorical features
    - Imputation for numerical features
    - Feature assembly and scaling
    - Gradient Boosted Tree regression model
    
    Args:
        train_df: PySpark DataFrame containing the training data
        val_df: PySpark DataFrame containing the validation data
        test_df: PySpark DataFrame containing the test data
        
    Returns:
        PipelineModel: The trained pipeline model
    """
    # Definición de características base del proceso de elaboración
    base_feats = [
        "Fermentation_Time", "Temperature", "pH_Level",
        "Gravity", "Alcohol_Content", "Bitterness", "Color",
        "Malt_Ratio", "Hop_Ratio", "Yeast_Ratio"
    ]
    
    # Características relacionadas con pérdidas en diferentes etapas
    loss_feats = [
        "Loss_During_Brewing",
        "Loss_During_Fermentation",
        "Loss_During_Bottling_Kegging"
    ]
    
    # Variable categórica: estilo de cerveza
    cat_feat = "Beer_Style"

    # 1. Procesamiento de variables categóricas
    # Convertimos el estilo de cerveza a índices numéricos
    idx = StringIndexer(inputCol=cat_feat, outputCol="Beer_Style_idx", handleInvalid="keep")
    # Aplicamos codificación one-hot para crear variables dummy
    ohe = OneHotEncoder(inputCols=["Beer_Style_idx"], outputCols=["Beer_Style_ohe"])

    # 2. Procesamiento de variables numéricas
    # Imputamos valores faltantes en todas las variables numéricas
    all_num = base_feats + loss_feats
    imputer = Imputer(
        inputCols=all_num,
        outputCols=[f + "_imp" for f in all_num],
        strategy="median"
    )

    # 3. Ensamblado de características
    # Combinamos todas las variables procesadas en un vector de características
    assembler = VectorAssembler(
        inputCols=[f + "_imp" for f in all_num] + ["Beer_Style_ohe"],
        outputCol="features_raw"
    )

    # 4. Normalización de características
    # Estandarizamos las variables para que tengan media 0 y desviación estándar 1
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")

    # 5. Configuración del modelo GBT (Gradient Boosted Trees)
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="Quality_Score",
        maxIter=30,     # Número de árboles en el ensemble
        maxDepth=4,     # Profundidad máxima de cada árbol
        stepSize=0.15   # Tasa de aprendizaje
    )

    # 6. Creación del pipeline completo
    pipeline = Pipeline(stages=[idx, ohe, imputer, assembler, scaler, gbt])

    # 7. Balanceo del conjunto de entrenamiento
    # Separamos las cervezas de baja y alta calidad
    bad = train_df.filter(F.col("Quality_Score") < 7)
    good = train_df.filter(F.col("Quality_Score") >= 7)
    # Submuestreamos las cervezas de alta calidad para balancear
    good_sample = good.sample(False, fraction=bad.count()/good.count(), seed=42)
    balanced_train = bad.union(good_sample)

    # 8. Entrenamiento del modelo
    model = pipeline.fit(balanced_train)

    return model

def save_model(model, output_file):
    """
    Save the trained pipeline model to disk.
    
    Creates a models directory if it doesn't exist and saves the model there.
    
    Args:
        model: PipelineModel to be saved
        output_file: String representing the filename for the model
        
    Returns:
        None
    """
    # Creamos el directorio de modelos si no existe
    if not os.path.exists("./models"):
        os.makedirs("./models")

    # Construimos la ruta completa del archivo
    output_file = f"./models/{output_file}"

    # Guardamos el modelo, sobrescribiendo si ya existe
    model.write().overwrite().save(output_file)

def main():
    """
    Main function to execute the full machine learning pipeline.
    
    Steps:
    1. Load and save raw data
    2. Create Spark session
    3. Load data into Spark
    4. Preprocess the data
    5. Display date range information
    6. Split data into train/validation/test sets
    7. Display split information
    8. Train the model
    9. Save the model
    
    Returns:
        None
    """
    # 1. Inicialización de Spark con la configuración especificada
    spark = (SparkSession.builder
            .appName(SPARK_APP_NAME)
            .master("local[*]")
            .config("spark.driver.memory", SPARK_MEMORY)
            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
            .getOrCreate())
    
    # 3. Carga de datos en formato Parquet a Spark
    df = spark.read.parquet("./data/brewery_sales.parquet")
    
    # 4. Aplicación de transformaciones de preprocesamiento
    df = preprocess_data(df)

    # 5. Análisis del rango temporal de los datos
    min_date = df.select(F.min("Brew_Date")).first()[0]
    max_date = df.select(F.max("Brew_Date")).first()[0]

    # Mostramos el rango temporal para verificación
    print(f"Fecha minima: {min_date}")
    print(f"Fecha maxima: {max_date}")

    # 6. División de los datos en conjuntos de entrenamiento, validación y prueba
    train_df, val_df, test_df = split_data(df)

    # 7. Análisis de la distribución de los datos
    # Mostramos el tamaño absoluto de cada conjunto
    print(f"Train: {train_df.count()}")
    print(f"Val: {val_df.count()}")
    print(f"Test: {test_df.count()}")
    
    # Mostramos la distribución porcentual
    print(f"Train: {train_df.count() / df.count() * 100}%")
    print(f"Val: {val_df.count() / df.count() * 100}%")
    print(f"Test: {test_df.count() / df.count() * 100}%")

    # 8. Entrenamiento del modelo con los datos procesados
    model = train_model(train_df, val_df, test_df)

    # 9. Persistencia del modelo entrenado
    save_model(model, "brewery_pipeline")

if __name__ == "__main__":
    main()