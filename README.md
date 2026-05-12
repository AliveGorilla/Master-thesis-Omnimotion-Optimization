# Optimized OmniMotion Pipeline for Dense Video Tracking

Optimization of the OmniMotion neural tracking framework for faster, more stable, and computationally efficient dense motion tracking in long video sequences.

---

## Project Overview

This project improves the original OmniMotion architecture by introducing multiple engineering and optimization-level modifications focused on:

- improving training stability,
- accelerating convergence,
- reducing unnecessary computational overhead,
- enhancing long-sequence motion consistency.

The work was developed as part of a Master's thesis focused on neural motion tracking and video understanding.

---

# Business Problem

Dense motion tracking models are computationally expensive and difficult to scale for long video sequences.

In real-world applications such as:

- industrial defect inspection,
- autonomous driving perception,
- robotics,
- sports analytics,
- video surveillance,

tracking instability and slow convergence significantly increase GPU costs and reduce deployment feasibility.

This project focuses on improving the training efficiency and robustness of the OmniMotion framework through architectural and optimization-level modifications.

---

# Key Results

- Simplified loss architecture to reduce computational overhead
- Improved convergence stability through temporal embedding freezing
- Enhanced hard-mining strategy for persistent error correction
- Accelerated training using Tiny-CUDA-NN (TCNN)
- Improved optimization consistency for long video sequences

---

# Architecture

## Pipeline Overview

```text
Input Video Sequence
        ↓
RAFT / Feature Tracking
        ↓
Feature Extraction
        ↓
3D Coordinate Encoding
        ↓
Invertible Neural Network (INN)
        ↓
Hard Mining Optimization
        ↓
Dense Motion Tracking Output
```
Example:

![Architecture](assets/ours.png)

---

# Engineering Optimizations

## 1. Loss Function Simplification

The original OmniMotion implementation used multiple loss functions during optimization.

After empirical evaluation, the smoothness loss was removed to reduce unnecessary computational overhead while preserving tracking quality.

### Impact

- Reduced optimization complexity
- Lower computational cost
- Cleaner convergence behavior

---

## 2. Persistent Hard-Mining Strategy

The original hard-mining implementation randomly selected regions with the highest probability of error.

This project introduced an aggregated hard-mining strategy that accumulates error regions throughout training, allowing the model to focus on consistently problematic areas.

### Impact

- Better error localization
- Improved robustness
- More stable optimization over time

### Implementation

- Aggregated error map tracking
- Persistent high-error region sampling

---

## 3. 3D Coordinate Encoding

The original OmniMotion model did not include explicit 3D coordinate encoding.

A periodic positional encoding mechanism was introduced for 3D coordinates, transforming the input into a higher-dimensional representation.

### Impact

- Improved spatial representation quality
- Better temporal consistency
- Enhanced feature expressiveness

---

## 4. Temporal Embedding Freezing

The original architecture repeatedly retrained temporal embeddings throughout the optimization process.

This project introduced temporal embedding freezing after 40,000 epochs, using the learned representation as a stable global temporal reference.

### Impact

- Improved training stability
- Reduced optimization noise
- Faster convergence behavior

---

## 5. Tiny-CUDA-NN (TCNN) Integration

The project integrated Tiny-CUDA-NN (TCNN) to accelerate neural field computations and improve training efficiency.

### Impact

- Faster GPU training
- Improved runtime performance
- Reduced computational bottlenecks

### Technologies

- CUDA
- Tiny-CUDA-NN
- PyTorch

---

# Performance Improvements

| Optimization | Engineering Benefit |
|---|---|
| Loss simplification | Reduced computational overhead |
| Persistent hard mining | Improved error-region learning |
| Temporal embedding freezing | Increased optimization stability |
| Tiny-CUDA-NN integration | Faster convergence and GPU efficiency |
| 3D coordinate encoding | Improved spatial representation |

---

# Potential Real-World Applications

- Industrial defect detection
- Motion analysis in sports
- Autonomous navigation systems
- Robotics perception
- Video understanding systems
- Long-sequence object tracking

---

# Tech Stack

- Python
- PyTorch
- CUDA
- Tiny-CUDA-NN
- Computer Vision
- Neural Rendering
- Dense Motion Tracking
- RAFT
- DINO
- Neural Networks
- Deep Learning

---

# Original Paper

Links to the original OmniMotion project:

- [Project Page](https://omnimotion.github.io/)
- [Paper](https://arxiv.org/pdf/2306.05422.pdf)
- [Video](https://www.youtube.com/watch?v=KHoAG3gA024)

---

# Installation

The project was tested with:

- Python 3.8
- Torch 1.10.0 + CUDA 11.1
- NVIDIA A100 GPU

```bash
git clone --recurse-submodules https://github.com/qianqianwang68/omnimotion/
cd omnimotion/

conda create -n omnimotion python=3.8
conda activate omnimotion

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
-f https://download.pytorch.org/whl/torch_stable.html

pip install matplotlib tensorboard scipy opencv-python tqdm tensorboardX \
configargparse ipdb kornia imageio[ffmpeg]
```

---

# Training

## 1. Data Preparation

Please follow the preprocessing instructions:

```text
preprocessing/README.md
```

You can also use the processed dataset provided by the original OmniMotion project:

https://omnimotion.cs.cornell.edu/dataset/

---

## 2. Start Training

```bash
python train.py --config configs/default.txt --data_dir {sequence_directory}
```

---

# Thesis

A more detailed explanation of the optimization methods can be found in the thesis paper:

[Master Thesis PDF](https://github.com/AliveGorilla/Master-thesis-Omnimotion-Optimization/blob/main/MasterThesisTlepin.pdf)

---

# Future Improvements

Potential next steps include:

- replacing RAFT with CoTracker,
- upgrading DINO to DINOv2,
- integrating depth-aware tracking,
- optimizing memory efficiency for longer sequences,
- improving real-time inference capabilities.

---

# Acknowledgements

This project is based on the original OmniMotion framework developed by the Cornell Vision and Learning Lab.

Training and testing settings were adapted from the original issue discussions:

https://github.com/qianqianwang68/omnimotion/issues/37
