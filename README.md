# reto_grupo_compilateds

Sistema detector de plagio de código fuente en **Python**, basado en los
fundamentos de [Dolos](https://doi.org/10.1111/jcal.12662): análisis de Árbol de
Sintaxis Abstracta (AST) + algoritmo **Winnowing** + un clasificador de
aprendizaje automático.

## Iteración 2 — qué cambió y por qué

La iteración 1 obtuvo F1 = 0.198 / AUC = 0.534 (azar). El diagnóstico:
las 20 "variantes" por problema del dataset de Kaggle **no son ofuscaciones
unas de otras, son algoritmos distintos** (recursivo vs. iterativo vs. fórmula
de Binet...). Etiquetar "mismo problema = plagio" le pedía al modelo detectar
equivalencia semántica entre algoritmos diferentes, tarea que el marco de
referencia declara explícitamente fuera del alcance (sección 9).

Cambios de la iteración 2:

1. **Dataset ad hoc con transformaciones documentadas** (opción contemplada en
   la sección 5.3 del marco): cada snippet original genera 4 variantes
   plagiadas con `pipeline/transforms.py` (renombrado de identificadores,
   código muerto, `x += y → x = x + y`, inversión de comparaciones,
   reordenamiento de funciones). Plagio = par derivado de la misma fuente.
   La mitad de los negativos son **difíciles**: mismo problema, algoritmo
   distinto.
2. **k = 15 (antes 23)**: los snippets tienen mediana de ~73 tokens; barrido
   empírico sobre k ∈ {5,10,15,23,30} × w ∈ {4,8,16}. Consistente con la
   hipótesis H₁ (k ∈ {15, 23}).
3. **9 características (antes 5)**: se añadieron métricas robustas en archivos
   cortos.
4. **Protocolo del marco**: umbral de decisión ajustado en validación
   (sección 8.2) y comparación de algoritmos de ML (sección 5.1).

**Resultados (test 20%, nunca visto):** precisión 0.985 · recall 0.960 ·
**F1 0.972** · AUC 0.999 — supera las metas del marco (sección 8.3) y la
línea base de Dolos (F1 = 0.865).

## Iteración 3 — nuevo dataset en parquet (`new_data/`)

La iteración 3 adapta el sistema a un nuevo dataset cuya tarea es
**detectar código generado por máquina** (LLMs): cada fila trae un snippet
individual con `code`, `generator`, `language` y `label`
(0 = humano, 1 = generado por máquina). Splits oficiales:
entrenamiento (500k), validación (100k) y muestra de prueba (1k).

- **La arquitectura del modelo no cambia** (misma red basada en el paper
  de Dolos: Dense 32 → Dropout 0.2 → Dense 16 → sigmoide).
- **Cambia la alimentación de datos**: como ya no hay pares de archivos,
  `pipeline/snippet_features.py` calcula 16 características *intrínsecas*
  por snippet con la misma maquinaria (tokenización AST normalizada +
  Winnowing k=15, w=4) más métricas de estilo (comentarios, líneas en
  blanco, identificadores...). Para lenguajes distintos de Python,
  `pipeline/snippet_tokenizer.py` usa un tokenizador léxico genérico
  (mismo enfoque multi-lenguaje de Dolos).

```bash
# 1) Generar las características de cada split (parquet → parquet)
python -m pipeline.build_snippets --input new_data/task_a_training_set_1.parquet \
    --output new_data_features/train_features.parquet --k 15 --w 4 --workers 10
python -m pipeline.build_snippets --input new_data/task_a_validation_set.parquet \
    --output new_data_features/validation_features.parquet --k 15 --w 4 --workers 10
python -m pipeline.build_snippets --input new_data/task_a_test_set_sample.parquet \
    --output new_data_features/test_features.parquet --k 15 --w 4 --workers 10

# 2) Entrenar y evaluar
jupyter notebook model_new_data.ipynb
```

`model_new_data.ipynb` entrena con el split oficial de entrenamiento,
ajusta el umbral de decisión en validación (sección 8.2 del marco) y
reporta sobre validación y prueba: **Precision, Recall, F1 y AUC-ROC**,
la **matriz de confusión** y una **gráfica de barras del F1 contra la
línea base de Dolos (0.865)**. Artefactos: `ai_code_model.keras`,
`scaler_new_data.joblib`, `decision_threshold_new_data.json` y las
figuras en `results/`.

## Arquitectura del pipeline

```
dataset Kaggle (5 problemas × 20 snippets)
     │
     ▼
[0] Generador de dataset ── pipeline/generate_dataset.py
     │   (crea data/: 100 grupos de plagio = original + 4 variantes
     │    ofuscadas con pipeline/transforms.py)
     ▼
archivos .py
     │
     ▼
[1] Tokenizador AST  ── pipeline/ast_tokenizer.py
     │   (módulo `ast` de Python; enmascara identificadores y literales)
     ▼
secuencia de tokens normalizados
     │
     ▼
[2] Winnowing        ── pipeline/winnowing.py
     │   (k-gramas → hash rodante → huellas digitales / fingerprints)
     ▼
huellas por archivo
     │
     ▼
[3] Características   ── pipeline/features.py
     │   (similitud por par de archivos: 9 métricas en [0,1])
     ▼
pairs_features.csv   ── pipeline/build_pairs.py
     │
     ▼
[4] Modelo de ML     ── model.ipynb  (red neuronal en TensorFlow
                        + comparación con RF / SVM / Reg. Logística)
```

## El CSV que consume el modelo

`model.ipynb` importa `FEATURE_COLUMNS` de `pipeline/features.py`, por lo que
notebook y pipeline no pueden desincronizarse. Columnas:

| Columna | Significado | Rango |
|---|---|---|
| `winnowing_similarity` | Huellas compartidas / mín(huellas de cada archivo) (estilo Dolos) | [0,1] |
| `shared_fragment_coverage` | Cobertura promedio de fragmentos compartidos entre ambos archivos | [0,1] |
| `token_overlap_ratio` | Jaccard de los conjuntos de k-gramas (nivel de tokens) | [0,1] |
| `ast_depth_difference` | Diferencia normalizada de profundidad del AST (mayor = más distinto) | [0,1] |
| `fingerprint_jaccard` | Jaccard de las huellas digitales | [0,1] |
| `small_kgram_jaccard` | Jaccard de k-gramas pequeños (k=4); robusto en archivos cortos | [0,1] |
| `node_type_cosine` | Coseno entre histogramas de tipos de nodo AST | [0,1] |
| `token_sequence_ratio` | Similitud de secuencia (difflib) entre cadenas de tokens | [0,1] |
| `length_ratio` | mín(tokens) / máx(tokens) | [0,1] |
| `label` | 1 = plagio (misma fuente, ofuscada), 0 = no plagio | {0,1} |

## Uso

```bash
pip install -r requirements.txt

# 0) Generar el dataset ad hoc (descarga el de Kaggle con kagglehub si hace falta)
python -m pipeline.generate_dataset --output-dir data --variants 4

# 1) Generar el CSV de características
python -m pipeline.build_pairs --data-dir data \
    --output pairs_features.csv --k 15 --w 4 --masking medium

# 2) Entrenar el modelo
jupyter notebook model.ipynb      # detecta y carga pairs_features.csv
```

El entrenamiento guarda tres artefactos: `plagiarism_model.keras`,
`scaler.joblib` y `decision_threshold.json` (umbral + parámetros del pipeline).

Parámetros configurables (variables independientes del marco de referencia):

- `--k` tamaño de k-grama (`5, 10, 15` elegido, `23` default Dolos, `30`)
- `--w` ventana de Winnowing (`4, 8, 16`)
- `--masking` nivel de enmascaramiento: `low` / `medium` / `high`
- `--variants` (generate_dataset) variantes plagiadas por snippet

## Dataset

Base: **Code Similarity Dataset – Python Variants** (Kaggle), el indicado en el
marco de referencia:

> https://www.kaggle.com/datasets/hemajitpatel/code-similarity-dataset-python-variants

Sobre esa base se construye el dataset ad hoc (sección 5.3 del marco):

- **Positivo (1)**: dos archivos del mismo grupo (original + variantes
  ofuscadas de la misma fuente) → C(5,2) = 10 pares/grupo × 100 grupos.
- **Negativo (0)**: dos archivos de grupos distintos, balanceados con los
  positivos. 50% negativos difíciles (mismo problema, algoritmo distinto)
  y 50% fáciles (problemas distintos).

La procedencia de cada archivo generado queda registrada en `data/manifest.csv`.

Alternativas si se quiere ampliar la evaluación:

- **IR-Plag / SOCO** — benchmark clásico usado por Dolos (originalmente Java).
- **POJ-104** (CodeXGLUE clone detection) — 104 problemas, miles de soluciones
  en C/C++; útil para validar la generalización del enfoque a otros lenguajes.
