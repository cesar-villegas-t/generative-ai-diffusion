# generative-ai-diffusion

[English](README.md) · **Español**

Sistema de generación de imágenes mediante **modelos de difusión basados en scores** (score-based diffusion), implementado en PyTorch. Proyecto de la asignatura Aprendizaje Automático III.

Permite entrenar un modelo sobre **MNIST** o **CIFAR-10**, generar imágenes a partir de ruido (con condicionamiento por clase e imputación de regiones) y evaluar la calidad con **FID** e **Inception Score**.

## Características

- **Procesos de difusión:** `ve` (Variance Exploding), `vp` (Variance Preserving) y `subvp`.
- **Schedules de ruido:** `exponential`, `linear` y `cosine`.
- **Samplers:** Euler-Maruyama (`euler`), Predictor-Corrector (`pc`), Probability Flow (`pf`) y Exponential Integrator (`expint`).
- **Generación condicionada** por etiqueta de clase e **imputación** (inpainting) de una región de la imagen.
- **Métricas:** FID e Inception Score (`torchmetrics`).

## Estructura

| Archivo | Descripción |
|---|---|
| `diffusion_functions.py` | Entrenamiento (`train_model`), generación (`generate_and_plot`) y carga de datasets |
| `diffusion_process.py` | Proceso de difusión gaussiano y función de pérdida |
| `score_model.py` | Red `ScoreNet` que estima el score |
| `noise_schedules.py` | Schedules de ruido |
| `samplers.py` | Samplers para la generación |
| `evaluate_metrics.py` | Carga de checkpoints y cálculo de FID / IS |
| `diffusion_utilities.py` | Utilidades de visualización |
| `checkpoints/` | Modelos ya entrenados (`.pth`) |

## Instalación

```bash
git clone https://github.com/cesar-villegas-t/generative-ai-diffusion.git
cd generative-ai-diffusion
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Los datasets MNIST y CIFAR-10 se descargan automáticamente en `data/` la primera vez que se usan. Se recomienda una GPU con CUDA para entrenar.

## Uso

Hay tres notebooks de ejemplo:

1. **`Entrenamiento_y_Generacion.ipynb`:** entrena un modelo y genera imágenes.
2. **`Evaluacion_modelos.ipynb`:** carga un checkpoint y calcula FID e Inception Score.
3. **`Configuraciones_modelo.ipynb`:** prueba distintos datasets, schedules, samplers e imputación.

Ejemplo rápido:

```python
from diffusion_functions import train_model, generate_and_plot

model = train_model(n_epochs=30, dataset="cifar10", diffusion_type="ve", schedule_type="exponential")
generate_and_plot(model, dataset="cifar10", diffusion_type="ve",
                  schedule_type="exponential", sampler="euler", n_images=10)
```

Cargar un modelo ya entrenado y evaluarlo:

```python
from evaluate_metrics import cargar_modelo, compute_fid_and_iscore

model = cargar_modelo("checkpoints/checkpoint_cifar10_ve_exponential.pth",
                      schedule_type="exponential", diffusion_type="ve")
compute_fid_and_iscore(model, dataset="cifar10", diffusion_type="ve", schedule_type="exponential")
```

Durante el entrenamiento se guarda un checkpoint en `checkpoints/checkpoint_<dataset>_<diffusion_type>_<schedule_type>.pth`.

## Autores

César Villegas Taillefer y Javier Llorente Capa.
