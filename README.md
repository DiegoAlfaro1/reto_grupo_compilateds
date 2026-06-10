# reto_grupo_compilateds

Sistema detector de plagio de código fuente en **Python**, basado en los
fundamentos de [Dolos](https://doi.org/10.1111/jcal.12662): análisis de Árbol de
Sintaxis Abstracta (AST) + algoritmo **Winnowing** + un clasificador de
aprendizaje automático.

## Arquitectura del pipeline

```
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
     │   (similitud por par de archivos: 5 métricas en [0,1])
     ▼
pairs_features.csv   ── pipeline/build_pairs.py
     │
     ▼
[4] Modelo de ML     ── model.ipynb  (red neuronal en TensorFlow)
```

Las etapas 1–3 son **este parser** (lo que añade este commit); la etapa 4 ya
existía en `model.ipynb`. El parser es el paso previo al entrenamiento: produce
el `pairs_features.csv` que el notebook carga automáticamente.

## El CSV que consume el modelo

`model.ipynb` espera un archivo `pairs_features.csv` con estas columnas
(verificado contra `FEATURE_COLUMNS` del notebook):

| Columna | Significado | Rango |
|---|---|---|
| `winnowing_similarity` | Huellas compartidas / mín(huellas de cada archivo) (estilo Dolos) | [0,1] |
| `shared_fragment_coverage` | Cobertura promedio de fragmentos compartidos entre ambos archivos | [0,1] |
| `token_overlap_ratio` | Jaccard de los conjuntos de k-gramas (nivel de tokens) | [0,1] |
| `ast_depth_difference` | Diferencia normalizada de profundidad del AST (mayor = más distinto) | [0,1] |
| `fingerprint_jaccard` | Jaccard de las huellas digitales | [0,1] |
| `label` | 1 = plagio (mismo problema), 0 = no plagio (problemas distintos) | {0,1} |

## Uso

```bash
pip install -r requirements.txt   # solo necesario para model.ipynb

# 1) Generar el CSV de características a partir del dataset
python -m pipeline.build_pairs --data-dir ./data/python_variants \
    --output pairs_features.csv --k 23 --w 4 --masking medium

# 2) Entrenar el modelo
jupyter notebook model.ipynb      # detecta y carga pairs_features.csv
```

Parámetros configurables (variables independientes del marco de referencia):

- `--k` tamaño de k-grama (`5, 10, 15, 23` default Dolos, `30`)
- `--w` ventana de Winnowing (`4, 8, 16`)
- `--masking` nivel de enmascaramiento: `low` / `medium` / `high`
- `--group-by` cómo agrupar archivos en "problemas": `parent` (default) /
  `grandparent` / `filename`

### Pruebas

```bash
python tests/test_pipeline.py
```

## Dataset recomendado

**Code Similarity Dataset – Python Variants** (Kaggle), el indicado en el marco
de referencia:

> https://www.kaggle.com/datasets/hemajitpatel/code-similarity-dataset-python-variants

Contiene múltiples problemas con ~20 variantes funcionalmente equivalentes por
problema (renombrado de variables, reordenamiento de bloques, `for`↔`while`,
comentarios/código muerto). Etiquetado de pares:

- **Positivo (1)**: dos variantes del *mismo* problema → C(20,2) = 190 pares/problema.
- **Negativo (0)**: dos soluciones de problemas *distintos*, muestreados y
  balanceados con los positivos.

Alternativas si se quiere ampliar la evaluación:

- **IR-Plag / SOCO** — benchmark clásico usado por Dolos (originalmente Java).
- **POJ-104** (CodeXGLUE clone detection) — 104 problemas, miles de soluciones
  en C/C++; útil para validar la generalización del enfoque a otros lenguajes.
- **Generación ad hoc**: aplicar transformaciones documentadas (ofuscación
  controlada) sobre soluciones propias, como contempla el marco de referencia.
