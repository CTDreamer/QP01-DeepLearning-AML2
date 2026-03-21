# QP_01 — Práctica de Deep Learning
**Aprendizaje Automático II · Universidad Politécnica de Madrid**

---

## Estructura del repositorio

```
QP_01/
├── Parte-I/
│   ├── flechas_completo.ipynb          # Notebook principal Parte I
│   ├── requirements.txt
│   ├── dataset/                        # Imágenes de flechas
│   ├── dataset_cleaned.csv             # Rutas + ángulos normalizados
│   ├── resnet_arrow.pth                # Pesos ResNetArrow entrenado
│   ├── vgg16_arrow.pth                 # Pesos VGG16Arrow entrenado
│   └── vgg_arrow.pth                   # Pesos VGGArrow entrenado
└── Parte-II/
    ├── transformer_prediccion_series_temporales.ipynb  # Notebook principal Parte II
    └── requirements.txt
```

---

## Parte I — Predicción de Ángulo de Flechas con CNNs

**Objetivo:** dado un imagen de una flecha pintada en el pavimento, predecir su ángulo de rotación (0°–360°).

### Problema y codificación circular

El ángulo no se predice directamente porque presenta una discontinuidad en 0°/360° que desestabiliza el entrenamiento. En su lugar se codifica como:

$$(\sin\theta,\ \cos\theta)$$

y se recupera el ángulo con `atan2`. La métrica usada es el **MAE Angular Circular** (en grados).

### Modelos entrenados

| Modelo | Arquitectura | Entrada | Parámetros |
|---|---|---|---|
| **VGGArrow** | CNN desde cero (4 bloques conv) | 128×128 | ~1.5 M |
| **VGG16Arrow** | Transfer Learning VGG-16 | 224×224 | ~40 M |
| **ResNetArrow** | Transfer Learning ResNet-18 | 224×224 | ~11 M |

Los modelos de transfer learning se entrenan en **dos fases**: backbone congelado primero, fine-tuning completo después.

### Archivos de pesos

Los ficheros `.pth` contienen los pesos entrenados. El flag `LOAD_PRETRAINED` del notebook controla si se carga o se reentrena:

```python
LOAD_PRETRAINED = True   # carga pesos → salta entrenamiento
LOAD_PRETRAINED = False  # entrena desde cero → guarda pesos
```

---

## Parte II — Predicción de Series Temporales con Transformers

**Objetivo:** diseñar y entrenar una arquitectura Transformer para predecir valores futuros, aplicada a dos dominios: señal sintética y datos reales de bolsa.

### Arquitectura común

Ambos modelos siguen el mismo esquema **Encoder-only**:

```
[batch, seq_len, 1]
   → Linear (input_projection)
   → TransformerEncoder (self-attention × N capas)
   → último token [:, -1, :]
   → Linear (output_layer)
   → [batch, 1]
```

### Tarea 1 — Función Seno

| Parámetro | Valor |
|---|---|
| Datos | 1 000 puntos de `sin(t)`, t ∈ [0, 100] |
| Modelo | `SequenceTransformer` — d_model=32, nhead=4, 2 capas |
| Ventana de contexto | 50 pasos |
| Épocas | 100 |
| Predicción | 200 valores futuros (autorregresivo) |

El seno sirve como validación: si el Transformer no aprende un patrón periódico y determinista, la arquitectura tiene algún problema.

### Tarea 2 — Precio de cierre AAPL

| Parámetro | Valor |
|---|---|
| Datos | Yahoo Finance — AAPL, 2020-01-01 a 2026-01-01 |
| Modelo | `TimeSeriesTransformer` — d_model=64, nhead=4, 2 capas, dropout=0.1 |
| Normalización | `MinMaxScaler` → [0, 1] |
| Ventana de contexto | 60 días |
| Épocas | 50 |
| Predicción | 50 días futuros (~2.5 meses), desnormalizado a USD |

### Predicción autorregresiva

Ambas tareas usan la misma estrategia de inferencia: la predicción de cada paso se concatena a la ventana y se elimina el elemento más antiguo, desplazando el contexto hacia el futuro.

---

## Instalación de dependencias

```bash
# Parte I
pip install -r Parte-I/requirements.txt

# Parte II
pip install -r Parte-II/requirements.txt
```
