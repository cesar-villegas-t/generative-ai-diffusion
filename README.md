# generative-ai-diffusion

**English** · [Español](README.es.md)

Image generation system based on **score-based diffusion models**, implemented in PyTorch. Project for the course *Aprendizaje Automático III* (Machine Learning III).

Train a model on **MNIST** or **CIFAR-10**, generate images from noise (with class conditioning and region imputation), and evaluate quality with **FID** and **Inception Score**.

## Features

- **Diffusion processes:** `ve` (Variance Exploding), `vp` (Variance Preserving) and `subvp`.
- **Noise schedules:** `exponential`, `linear` and `cosine`.
- **Samplers:** Euler-Maruyama (`euler`), Predictor-Corrector (`pc`), Probability Flow (`pf`) and Exponential Integrator (`expint`).
- **Class-conditional generation** and **imputation** (inpainting) of an image region.
- **Metrics:** FID and Inception Score (`torchmetrics`).

## Structure

| File | Description |
|---|---|
| `diffusion_functions.py` | Training (`train_model`), generation (`generate_and_plot`) and dataset loading |
| `diffusion_process.py` | Gaussian diffusion process and loss function |
| `score_model.py` | `ScoreNet` network that estimates the score |
| `noise_schedules.py` | Noise schedules |
| `samplers.py` | Sampling algorithms |
| `evaluate_metrics.py` | Checkpoint loading and FID / IS computation |
| `diffusion_utilities.py` | Visualization utilities |
| `checkpoints/` | Pre-trained models (`.pth`) |

## Installation

```bash
git clone https://github.com/cesar-villegas-t/generative-ai-diffusion.git
cd generative-ai-diffusion
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

MNIST and CIFAR-10 are downloaded automatically into `data/` on first use. A CUDA-capable GPU is recommended for training.

## Usage

Three example notebooks are included (written in Spanish):

1. **`Entrenamiento_y_Generacion.ipynb`:** train a model and generate images.
2. **`Evaluacion_modelos.ipynb`:** load a checkpoint and compute FID and Inception Score.
3. **`Configuraciones_modelo.ipynb`:** try different datasets, schedules, samplers and imputation.

Quick example:

```python
from diffusion_functions import train_model, generate_and_plot

model = train_model(n_epochs=30, dataset="cifar10", diffusion_type="ve", schedule_type="exponential")
generate_and_plot(model, dataset="cifar10", diffusion_type="ve",
                  schedule_type="exponential", sampler="euler", n_images=10)
```

Load a pre-trained model and evaluate it:

```python
from evaluate_metrics import cargar_modelo, compute_fid_and_iscore

model = cargar_modelo("checkpoints/checkpoint_cifar10_ve_exponential.pth",
                      schedule_type="exponential", diffusion_type="ve")
compute_fid_and_iscore(model, dataset="cifar10", diffusion_type="ve", schedule_type="exponential")
```

During training, a checkpoint is saved to `checkpoints/checkpoint_<dataset>_<diffusion_type>_<schedule_type>.pth`.

## Authors

César Villegas Taillefer and Javier Llorente Capa.

## License

MIT. See [LICENSE](LICENSE).
