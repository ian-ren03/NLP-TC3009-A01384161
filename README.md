# PROYECTO FINAL INTEGRADOR: De Cero a Preentrenado: Evaluación Comparativa de Arquitecturas Sequence-to-Sequence en Machine Translation Neuronal

**Integrantes 1: Ian Roberto Rendón Berlanga - A01384161**
**Integrante 2: Gustavo Axel Huerta García - A00574308**

## Objetivo
El objetivo del proyecto es construir cuatro arquitecturas de aprendizaje profundo para la tarea de **Traducción Automática Neuronal (NMT)** del inglés al español (EN \rightarrow ES). A través de este estudio empírico, el equipo analizará el impacto del sesgo inductivo secuencial frente a los mecanismos de autoatención pura desde cero y la ganancia obtenida mediante Transfer Learning y ajuste fino (*Fine-Tuning*) a partir de los modelos preentrenados a gran escala.

## Auditoría de datos

**Corpus:** pares paralelos Inglés–Español de Tatoeba/ManyThings (`spa.txt`).
**Protocolo:** se usó sin modificaciones el bloque oficial de carga, filtrado y partición del enunciado.

| Parámetro | Valor |
|---|---|
| Filtro | ≥ 8 palabras en inglés, ≤ 120 palabras en ambos idiomas |
| Mezcla | determinista, `random.Random(23)` |
| Partición | 80 % Train / 10 % Validation / 10 % Test |
| Pares tras el filtro | 36 413 |

| Partición | Muestras |
|---|---|
| Train | 29 130 |
| Validation | 3 641 |
| Test | 3 642 |

**SHA-256 Test Audit ID:** `dbac04ac5648`
Cada notebook repite el bloque de partición y verifica este ID con un `assert`, así que cualquier cambio en el filtro, la semilla o el archivo hace fallar la ejecución.

### Prevención de fuga de datos

- Los vocabularios (`eng_word2idx`, `spa_word2idx`) se construyen **solo con `train_pairs`**.
- Las palabras de validación y test que no están en el vocabulario de entrenamiento se mapean a `<unk>`.
- `test_pairs` no se usa para entrenar, ajustar hiperparámetros, elegir épocas ni hacer early stopping; solo se usa en la evaluación final.
- La normalización (minúsculas, sin acentos, sin puntuación) usa una única función, `clean_text`, aplicada igual al entrenamiento y al cálculo de BLEU de los cuatro modelos.

### Vocabularios y palabras desconocidas

Vocabularios con frecuencia mínima de 2 (incluyen `<pad>`, `<unk>`, `<sos>`, `<eos>`):

| | Inglés | Español |
|---|---|---|
| Tamaño del vocabulario | 5 453 | 8 027 |
| `<unk>` en train | 1.17 % | 2.71 % |
| `<unk>` en validation | 1.96 % | 4.39 % |
| `<unk>` en test | 2.07 % | 4.37 % |

Las tasas de `<unk>` son mayores en validation y test porque el vocabulario se construye solo con train.

### Limitación conocida: oraciones repetidas entre train y test

Tatoeba contiene varias traducciones al español para una misma oración en inglés. Como resultado, **564 de 3 642 oraciones de test (15.5 %)** tienen una fuente en inglés que también aparece en train (con otra traducción posible al español). La partición no se modificó, por requerimiento del protocolo. Esto puede favorecer a los modelos con mayor capacidad de memorización, por lo que en la evaluación se reporta, además de la tabla oficial, el BLEU sobre el subconjunto de test cuya fuente no aparece en train.
## Secciones del Proyecto
