# Sistema de Control de Calidad para Cervecería 🍺

Este proyecto implementa un sistema completo de control de calidad para una línea de producción de cerveza, utilizando aprendizaje automático para predecir y monitorear la calidad del producto en tiempo real.

## 📋 Tabla de Contenidos
- [Descripción General](#descripción-general)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Pipeline de Datos](#pipeline-de-datos)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Panel de Control](#panel-de-control)
- [Mantenimiento](#mantenimiento)
- [Soporte](#soporte)

## 🎯 Descripción General

El sistema consta de tres componentes principales:
1. Pipeline de datos y preprocesamiento
2. Modelo de predicción de calidad
3. Panel de control en tiempo real

El sistema utiliza Apache Spark para el procesamiento de datos a gran escala y Streamlit para la visualización en tiempo real.

Se puede encontrar una presentacion del proyecto en el archivo `assets/presentacion.pdf`.

## 💻 Requisitos del Sistema

- Python 3.8+
- Java 8+ (requerido para Apache Spark)
- 16GB RAM mínimo recomendado
- Sistema operativo: Linux, macOS, o Windows con WSL2

### Dependencias Principales

Para instalar las dependencias, asegurate de instalarlas con el archivo `requirements.txt`.

## 🚀 Instalación

1. Clone el repositorio:
```bash
git clone https://github.com/guzmandam/DM-ProyectoFinal
cd DM-ProyectoFinal
```

2. Cree un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instale las dependencias:
```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
brewery-quality-control/
├── data/                  # Datos de entrenamiento y producción
├── models/               # Modelos entrenados
├── app.py               # Aplicación de monitoreo en tiempo real
├── main.py              # Pipeline principal de entrenamiento
├── retriever.py         # Módulo de obtención de datos
└── requirements.txt     # Dependencias del proyecto
```

## 🔄 Pipeline de Datos

### Obtención de Datos

1. Los datos se obtienen automáticamente del dataset "ankurnapa/Brewery_sales" usando la clase `Retriever`:
```bash
python retriever.py
```

2. Los datos se guardan en formato Parquet en el directorio `data/`.

### Preprocesamiento

El preprocesamiento incluye:
- Separación de ratios de ingredientes
- Conversión de fechas
- Imputación de valores faltantes
- Normalización de características
- Codificación de variables categóricas

## 🤖 Entrenamiento del Modelo

1. Ejecute el script de entrenamiento:
```bash
python main.py
```

El proceso incluye:
- División temporal de datos (train/validation/test)
- Balanceo de clases
- Entrenamiento de modelo GBT (Gradient Boosted Trees)
- Guardado del modelo en `./models/brewery_pipeline`

### Métricas y Validación

El modelo se evalúa considerando:
- Precisión en la predicción de calidad
- Robustez ante valores atípicos
- Capacidad de generalización temporal

## 📊 Panel de Control

1. Inicie la aplicación de monitoreo:
```bash
streamlit run app.py
```

El panel incluye:
- Visualización en tiempo real de predicciones
- Sistema de alertas de calidad
- Métricas clave de producción
- Procedimientos de emergencia
- Guías de interpretación

### Características Principales

- **Monitoreo en Tiempo Real**: Actualización cada 2 segundos
- **Alertas Automáticas**: Notificación inmediata de problemas de calidad
- **Interfaz Intuitiva**: Diseño optimizado para operadores de producción
- **Registro Automático**: Documentación de incidencias y mediciones

## 🔧 Mantenimiento

### Calibración del Modelo

Se recomienda reentrenar el modelo:
- Mensualmente con nuevos datos
- Cuando cambien las condiciones de producción
- Si se observa drift en las predicciones

### Estándares de Código

- Siga PEP 8 para Python
- Documente todas las funciones y clases
- Mantenga la cobertura de pruebas > 80%

## 🤝 Soporte

Para soporte y consultas:
- 📧 Email: soporte@pasemeporfa.com
- 💬 Slack: #brewery-quality-control
- 📞 Teléfono: +34 900 123 456

---

Desarrollado con ❤️ por el Equipo de Calidad
