# WekaJava - Proyecto de Aprendizaje Automático

Este proyecto está desarrollado en **Java** y utiliza la librería **Weka** para realizar tareas de **aprendizaje automático**, como clasificación y validación cruzada. El objetivo principal es evaluar dos modelos de clasificación, **Decision Stump** y **Naive Bayes**, utilizando un **dataset** proporcionado.

## Descripción del Proyecto

El programa carga el dataset **calificaciones_unidas_nominalizadas.arff**, que contiene datos de calificaciones, para aplicar técnicas de clasificación y validar los resultados utilizando **validación cruzada**.

Este proyecto incluye:

- **Evaluación de modelos de clasificación** con **Decision Stump** y **Naive Bayes**.
- **Validación cruzada** para evaluar la precisión de los modelos.
- **Métricas de rendimiento** para ambos modelos.

## Dataset

El **dataset** utilizado en este proyecto está en formato **ARFF** y contiene información de calificaciones nominalizadas.

- **Nombre del archivo**: `calificaciones_unidas_nominalizadas.arff`
- **Descripción**: Este archivo contiene atributos relacionados con el rendimiento académico de los estudiantes y se utiliza para predecir la categoría o clase de cada estudiante.

### Cargar el Dataset

El archivo ARFF se carga utilizando la librería **Weka** en el siguiente código:

```java
// Cargar el archivo ARFF
File archivo = new File("calificaciones_unidas_nominalizadas.arff");
ARFFLoader loader = new ARFFLoader();
loader.setFile(archivo);
Instances datos = loader.getDataSet();
datos.setClassIndex(datos.numAttributes() - 1);  // Establecer el atributo de clase
