# ParkinVoice: Detección de Parkinson a partir de descriptores vocales

Proyecto de la asignatura Bioinformática y Medicina del Grado en Inteligencia Artificial de la Universidade da Coruña.

## Integrantes del grupo

- Iván Novío París
- Nathan Mariñas Pose

## Objetivo

Analizar la detección de Parkinson a partir de características acústicas de la voz y comparar tres escenarios de entrenamiento:

1. Modelos clásicos entrenados con datos reales.
2. Modelos clásicos entrenados con datos puramente sintéticos generados con CTGAN.
3. Escenario mixto real + sintético para estudiar si el dato sintético puede servir como apoyo experimental.

## Metodología resumida

1. Carga del dataset `pd_speech_features.csv` y eliminación de la columna `id`.
2. División estratificada train/test antes de cualquier selección de variables.
3. Selección de características mediante una combinación de información mutua e importancia de RandomForest.
4. Entrenamiento y comparación de varios modelos clásicos: Logistic Regression, SVM, KNN, RandomForest y Gradient Boosting.
5. Generación de un conjunto puramente sintético con CTGAN usando solo las variables seleccionadas.
6. Evaluación de tres estrategias: real, sintética y mixta.
7. Añadido de una capa de rigor experimental con sensibilidad, especificidad y un bloque de robustez con varias semillas.
8. Exportación de resultados, figuras, informe Markdown y datos estáticos para una app de presentación.

## Cómo ejecutar

### Análisis completo

```bash
python analysis_parkinson.py
```

### Ajustar CTGAN

```bash
python analysis_parkinson.py --ctgan-epochs 100 --robustness-ctgan-epochs 50
```

### Abrir la app estática

Abre `index.html` en el navegador o publícala con GitHub Pages.

## Estructura del repositorio

```text
.
├── analysis_parkinson.py
├── app-data.js
├── app.js
├── index.html
├── styles.css
├── pd_speech_features.csv
├── results/
│   ├── analysis_report.md
│   ├── summary.json
│   ├── real_model_results.csv
│   ├── synthetic_model_results.csv
│   ├── mixed_model_results.csv
│   ├── robustness_summary.csv
│   ├── feature_family_summary.csv
│   └── figures/
├── requirements.txt
└── enunciado.md
```

## Resultados clave

- El entrenamiento con datos reales sigue siendo la referencia principal de despliegue.
- El entrenamiento con datos sintéticos puros conserva parte de la señal, pero empeora claramente en balanced accuracy y ROC AUC.
- La selección de variables está dominada por rasgos TQWT/wavelet y por descriptores MFCC/delta, lo que refuerza la idea de que la información multiescala y dinámica es central en este problema.
- La deriva sintética es especialmente fuerte en variables como `locAbsJitter` y en varias componentes TQWT, lo que ayuda a explicar la caída de rendimiento de CTGAN.
- El escenario mixto se incluye para responder a la pregunta relevante de proyecto: aunque el sintético puro no reemplace al real, todavía puede aportar valor experimental como complemento.

## Enlaces

- App local: [index.html](index.html)
- App desplegada: pendiente de publicar en GitHub Pages o servicio equivalente.
- Zenodo: pendiente de generar el DOI de la release final.
- Presentación del examen: pendiente de añadir el enlace final.

## Archivos principales de salida

- `results/analysis_report.md`: informe completo del análisis.
- `results/summary.json`: resumen estructurado para reutilización.
- `results/feature_family_summary.csv`: agrupación de variables por familias acústicas.
- `results/robustness_summary.csv`: estabilidad de resultados con varias semillas.
- `results/figures/`: figuras listas para memoria y presentación.

## Requisitos

Las dependencias principales están listadas en `requirements.txt`:

- numpy
- pandas
- scipy
- matplotlib
- scikit-learn
- ctgan
- torch

## Nota de entrega

Antes de la entrega final conviene completar los apartados pendientes de enlaces, añadir los nombres reales del grupo y generar una release etiquetada para asociarla con Zenodo.