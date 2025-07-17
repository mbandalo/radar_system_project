
# Radar Point Cloud Segmentation with VAE-HMM-GNN-RL

**Adaptive Radar Point Cloud Segmentation in Highly Imbalanced Automotive Scenarios**

This project implements a hybrid AI pipeline for per-point radar object segmentation. It combines a variational autoencoder (VAE), hidden Markov model (HMM), graph neural network (GNN), and reinforcement learning (RL) to handle dynamic automotive radar scenes with extreme class imbalance.

---

## Features

- **VAE Encoder** – Compresses radar features into latent representations.
- **HMM Scene Encoder** – Captures temporal dynamics across sequences.
- **GNN Segmenter** – Performs per-point classification with spatial context.
- **Reinforcement Learning Agent (optional)** – Learns adaptive scene-level actions.
- **Robust to Imbalanced Classes** – Weighted loss & sampling to emphasize rare objects.
- **Modular Config (`config.yaml`)** – All hyperparams & paths are easily adjustable.

---

## Dataset

The model uses the [RadarScenes dataset](https://radar-scenes.com/):

- **Classes (12)**:  
  ```
  0: Passenger Cars  
  1: Large Vehicles  
  2: Trucks  
  3: Busses  
  4: Trains  
  5: Bicycles  
  6: Motorcycles  
  7: Pedestrians  
  8: Pedestrian Groups  
  9: Animals  
 10: Other Dynamic  
 11: Static Environment
  ```

Place the dataset under:
```bash
data/RadarScenes/RadarScenes/data/
```

---

## Training

To train the full pipeline:

```bash
python main.py 
```

This will:
1. Train the VAE on sensor-level features.
2. Train the HMM on scene sequences.
3. Train the GNN for segmentation.
4. Run simulation (optional RL).

---

## Evaluation

After training, run evaluation:

```bash
tensorboard --logdir=.
```

Metrics:
- Overall Accuracy
- Macro F1
- Per-class Precision / Recall

Static class (11) is down-weighted automatically to focus on dynamic objects.

---

## Configuration

Key hyperparameters are located in `conf/config.yaml`:

```yaml
scene:
  use_focal_loss: false
  max_class_weight: 5.0
  gnn_hidden_layers: [64, 64]
  early_stopping_patience: 10
```

To ignore static class:

```python
class_weights[11] = 0.1
```

---

## Architecture

```
Raw Radar → VAE → HMM → GNN → Per-Point Class Labels
                                ↘ RL Policy (optional)
```

- AE: `sensor_module/vae.py`  
- HMM: `scene_module/hmm.py`  
- GNN: `scene_module/gnn_point_segmenter.py`  
- RL: `rl_agent/ppo_agent.py`

---
