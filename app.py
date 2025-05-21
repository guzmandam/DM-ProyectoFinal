import streamlit as st
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel

# Configuration Constants (Consider moving to a separate config file or environment variables)
SPARK_APP_NAME = "BreweryQualityApp"
SPARK_DRIVER_MEMORY = "8g"
SPARK_EXECUTOR_MEMORY = "8g"
MODEL_PATH = "/home/entropiadev/models/brewery_pipeline"
DATA_PATH = "./data/brewery_sales.parquet"
SIMULATION_BATCH_SIZE = 5
SIMULATION_SLEEP_INTERVAL = 2  # seconds

def initialize_spark_session():
    """Initializes and returns a Spark session with predefined configurations."""
    spark = (SparkSession.builder.
               appName(SPARK_APP_NAME).
               config("spark.driver.memory", SPARK_DRIVER_MEMORY).
               config("spark.executor.memory", SPARK_EXECUTOR_MEMORY).
               getOrCreate())
    return spark

def load_model(model_path: str):
    """
    Loads the pre-trained machine learning model from the specified path.
    
    Args:
        model_path (str): The file system path to the saved PipelineModel.
        
    Returns:
        PipelineModel: The loaded machine learning model.
    """
    model = PipelineModel.load(model_path)
    return model

def load_and_preprocess_data(spark_session: SparkSession, data_path: str):
    """
    Loads brewery sales data from a Parquet file and preprocesses it.
    
    Preprocessing steps include:
    - Splitting the 'Ingredient_Ratio' column into 'Malt_Ratio', 'Hop_Ratio', and 'Yeast_Ratio'.
    - Converting the 'Brew_Date' column to a date type.
    
    Args:
        spark_session (SparkSession): The active Spark session.
        data_path (str): The path to the Parquet file containing brewery sales data.
        
    Returns:
        DataFrame: The preprocessed Spark DataFrame.
    """
    df_full = spark_session.read.parquet(data_path)

    # Split ingredient ratio string into separate numeric columns
    df_full = (df_full
          .withColumn("ratio_arr", F.split(F.col("Ingredient_Ratio"), ":"))
          .withColumn("Malt_Ratio" , F.col("ratio_arr").getItem(0).cast("double"))
          .withColumn("Hop_Ratio"  , F.col("ratio_arr").getItem(1).cast("double"))
          .withColumn("Yeast_Ratio", F.col("ratio_arr").getItem(2).cast("double"))
          .drop("Ingredient_Ratio", "ratio_arr")  # Drop original and temporary columns
    )

    # Convert Brew_Date to date type
    df_full = df_full.withColumn("Brew_Date", F.to_date("Brew_Date"))
    return df_full

# Feature columns used for model prediction (excluding target or identifier columns if any)
# This list should match the features the model was trained on.
FEATURE_COLS = [
    "Fermentation_Time", "Temperature", "pH_Level", "Gravity",
    "Alcohol_Content", "Bitterness", "Color",
    "Malt_Ratio", "Hop_Ratio", "Yeast_Ratio",
    "Loss_During_Brewing", "Loss_During_Fermentation", "Loss_During_Bottling_Kegging"
]

def process_batch(df_full_data: pd.DataFrame, model_pipeline: PipelineModel, batch_size: int):
    """
    Processes a random batch of data using the loaded model.
    
    Steps:
    1. Takes a random sample (micro-batch) from the full dataset.
    2. Makes predictions using the provided model pipeline.
    3. Selects relevant columns for display.
    
    Args:
        df_full_data (DataFrame): The full Spark DataFrame to sample from.
        model_pipeline (PipelineModel): The trained ML model.
        batch_size (int): The number of rows to include in the micro-batch.
        
    Returns:
        DataFrame: A Spark DataFrame containing the input features and predictions for the batch.
    """
    # Take a random micro-batch of specified size
    df_batch = df_full_data.orderBy(rand()).limit(batch_size)

    # Make predictions
    df_batch_predictions = model_pipeline.transform(df_batch)
    
    # Select columns for display (original features + prediction)
    # Note: Ensure 'prediction' is the actual output column name from your model.
    columns_to_display = FEATURE_COLS + ["Brew_Date", "prediction"] # Add Brew_Date and prediction
    
    # Filter out any columns that might not exist in df_batch_predictions if FEATURE_COLS is too broad
    # Or, more robustly, select only columns that are present in df_batch_predictions.
    # For this example, we assume all FEATURE_COLS plus Brew_Date and prediction are in df_batch_predictions.
    df_display = df_batch_predictions.select(columns_to_display)
    
    return df_display

def main_app():
    """
    Main function to run the Streamlit application for Brewery Quality Control.
    
    Initializes the UI, loads data and model, and runs the simulation loop
    to display real-time quality predictions.
    """
    # Configure Streamlit page settings
    st.set_page_config(layout="wide", page_title="Control de Calidad de Cervecería")
    st.title("Panel de Control de Calidad de Cervecería")

    # Application sidebar for controls and information
    st.sidebar.header("Controles de Simulación")
    
    # Información sobre la aplicación
    with st.sidebar.expander("ℹ️ Acerca de esta aplicación"):
        st.write("""
        Este panel de control permite monitorear en tiempo real la calidad de los lotes de cerveza en producción. 
        Utiliza un modelo de aprendizaje automático entrenado con datos históricos para predecir la calidad del producto.
        
        **Características principales:**
        - Monitoreo en tiempo real de variables críticas
        - Predicciones automáticas de calidad por lote
        - Sistema de alertas tempranas
        - Visualización de tendencias y patrones
        - Registro automático de incidencias
        
        **Integración con sistemas de producción:**
        - Conexión directa con sensores de línea
        - Sincronización con sistema MES
        - Registro en base de datos central
        - Exportación automática de reportes
        """)
    
    # Guía de interpretación
    with st.sidebar.expander("📊 Guía de interpretación"):
        st.write("""
        **Puntuación de Calidad:**
        - 0.8 - 1.0: Calidad Óptima ✅
          *Producción normal, no se requieren ajustes*
        - 0.6 - 0.8: Calidad Aceptable 🟡
          *Revisar tendencias y considerar ajustes preventivos*
        - < 0.6: Requiere Revisión 🔴
          *Activar protocolo de control de calidad*
        
        **Rangos Aceptables por Variable:**
        - Temperatura: 18-22°C
        - pH: 4.0-6.0
        - Gravedad: 1.010-1.020
        - Tiempo de Fermentación: 72-96h
        
        Los valores se actualizan cada {} segundos con datos de {} lotes.
        """.format(SIMULATION_SLEEP_INTERVAL, SIMULATION_BATCH_SIZE))
    
    # Procedimientos de emergencia
    with st.sidebar.expander("🚨 Procedimientos de emergencia"):
        st.write("""
        **En caso de alerta de calidad:**
        1. Pausar la línea de producción inmediatamente
        2. Tomar muestras manuales del último lote
        3. Contactar al supervisor de calidad
        4. Documentar el incidente en el sistema
        5. Iniciar protocolo de trazabilidad
        
        **Contactos de emergencia:**
        - Supervisor de Calidad: Ext. 1234
        - Mantenimiento: Ext. 5678
        - Laboratorio: Ext. 9012
        - Gerente de Producción: Ext. 4567
        
        **Ubicación de equipos de emergencia:**
        - Kit de muestreo: Armario A-123
        - Documentación de respaldo: Oficina de calidad
        - Equipo de medición portátil: Laboratorio principal
        """)

    # Nueva sección: Mantenimiento y Calibración
    with st.sidebar.expander("🔧 Mantenimiento y Calibración"):
        st.write("""
        **Calendario de Calibración:**
        - Sensores de temperatura: Cada 48 horas
        - Medidores de pH: Calibración diaria
        - Densímetros: Calibración semanal
        
        **Mantenimiento Preventivo:**
        - Limpieza de sensores: Cada 8 horas
        - Verificación de conexiones: Cada 24 horas
        - Backup de datos: Automático cada 4 horas
        
        **Registro de Calibración:**
        Documentar toda calibración en el sistema MES
        usando el formato: CAL-YYYY-MM-DD-EQUIPO
        """)

    # Nueva sección: Normativas y Cumplimiento
    with st.sidebar.expander("📋 Normativas y Cumplimiento"):
        st.write("""
        **Estándares Aplicables:**
        - ISO 9001:2015
        - HACCP
        - Normas específicas de cervecería
        
        **Documentación Requerida:**
        - Registro de lotes
        - Informes de calidad
        - Trazabilidad de ingredientes
        - Registros de temperatura
        
        **Auditorías:**
        - Internas: Mensual
        - Externas: Trimestral
        - Registro sanitario: Anual
        """)

    # Controles de simulación
    simulate_alert_button = st.sidebar.button("Simular Alerta de Calidad")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Monitor de Calidad en Tiempo Real")
        st.write("""
        Este gráfico muestra las predicciones de calidad para cada lote en tiempo real.
        Las fluctuaciones significativas pueden indicar problemas en el proceso de producción.
        
        **Indicadores clave:**
        - Línea verde: Calidad óptima
        - Línea amarilla: Zona de precaución
        - Línea roja: Límite de control
        """)
    
    with col2:
        st.subheader("Parámetros Críticos")
        st.info("""
        🌡️ Temperatura: 18-22°C
        📊 pH: 4.0-6.0
        🔄 Tiempo de Fermentación: 72-96h
        🎯 Gravedad: 1.010-1.020
        ⚖️ Ratio Malta:Lúpulo: 70:30
        """)
        
        st.warning("""
        **Límites de Control:**
        ⚠️ Variación máx. temperatura: ±1.5°C
        ⚠️ Variación máx. pH: ±0.3
        ⚠️ Desviación tiempo: ±4h
        """)

    # Initialize Spark, load model, and data once
    spark = initialize_spark_session()
    model = load_model(MODEL_PATH)
    df_full = load_and_preprocess_data(spark, DATA_PATH)

    # Placeholders for dynamic Streamlit elements
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    table_placeholder = st.empty()
    alert_placeholder = st.empty()

    if simulate_alert_button:
        alert_placeholder.error("""
        🚨 ¡ALERTA DE CALIDAD ACTIVADA!
        
        Último lote requiere revisión inmediata.
        - Verificar parámetros de temperatura
        - Revisar niveles de pH
        - Comprobar ratios de ingredientes
        
        Contacte al supervisor de calidad: Ext. 1234
        """)

    st.sidebar.success("Simulación en curso...") # Indicate simulation is active

    # Initialize chart with an appropriate y-axis for predictions
    initial_chart_data = pd.DataFrame({'Puntuación de Calidad': [0.0, 1.0]}) 
    chart = chart_placeholder.line_chart(initial_chart_data)

    # Main simulation loop
    while True:
        df_display_batch = process_batch(df_full, model, SIMULATION_BATCH_SIZE)
        pdf_batch_display = df_display_batch.toPandas()

        # Update metrics
        with metrics_placeholder.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("Temperatura Promedio", f"{pdf_batch_display['Temperature'].mean():.1f}°C")
            m2.metric("pH Promedio", f"{pdf_batch_display['pH_Level'].mean():.2f}")
            m3.metric("Calidad Predicha", f"{pdf_batch_display['prediction'].mean():.3f}")

        # Update chart
        chart.add_rows(pdf_batch_display[["prediction"]].rename(columns={'prediction': 'Puntuación de Calidad'}))
            
        # Update table with formatted column names
        renamed_cols = {
            'Brew_Date': 'Fecha',
            'Temperature': 'Temperatura',
            'pH_Level': 'Nivel pH',
            'prediction': 'Predicción'
        }
        display_df = pdf_batch_display.rename(columns=renamed_cols)
        table_placeholder.dataframe(display_df)

        time.sleep(SIMULATION_SLEEP_INTERVAL)

if __name__ == "__main__":
    # This ensures main_app() is called only when the script is executed directly
    main_app()
