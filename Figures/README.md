# Neural Field for Fiber Bundle Imaging
<p align="center">
  <img src="Figures/Principle.png" width="1000">
</p>

---

## Overview
Fiber bundle imaging systems enable minimally invasive **in-vivo** imaging but suffer from sampling artifacts (e.g., honeycomb patterns) due to discrete and non-uniform fiber layouts.  

This repository implements an **unsupervised test-time training method** for removing these artifacts from a burst of misaligned frames, without requiring ground truth, calibration, or prior knowledge of fiber geometry. The key idea is to use an image burst to recover information occluded by the fiber mask in individual frames but revealed across multiple frames.

Our method jointly optimizes:
- A **motion network** that models frame-to-frame transformations.  
- A **scene representation network** that reconstructs a super-resolved canonical scene by aggregating information observed across different frames using coordinate-based implicit representation and positional encoding.

---

## Method Summary

Each input frame is modeled as a warped observation of a canonical scene.  
For each pixel coordinate $(x, y, t)$, the reconstructed RGB values are:

$$
\hat{I}(x, y, t) = [\hat{R}, \hat{G}, \hat{B}] = f_\theta\left(\gamma\left(T_{g_\phi}(x, y, t)\right)\right)
$$

where:  
- $T_{g_\phi}$ : motion transformation (e.g., homography or optical flow)  
- $\gamma(\cdot)$ : positional encoding  
- $f_\theta$ : scene representation network  
- Both networks are trained jointly using an $\ell_2$ reconstruction loss  

The optimization minimizes:

$$
\mathcal{L} = \sum_{x, y, t} \left\lVert \hat{I}(x, y, t) - I(x, y, t) \right\rVert^2
$$

where:  
- $I(x, y, t)$ denotes the observed RGB value at pixel $(x, y)$ in frame $t$ (the input image sequence).  
- $\hat{I}(x, y, t)$ denotes the reconstructed RGB value predicted by the neural model.  

Both networks are multilayer perceptrons (MLPs). To mitigate spectral bias and improve the reconstruction of high-frequency details, the spatial coordinates are first passed through the positional encoding $\gamma(\cdot)$ before being mapped to RGB values by $f_\theta$.

---

## Reconstruction Examples
<p align="center">
  <img src="Figures/Reconstruction Examples.png" width="1000">
</p>

---

## Repository Structure
- **`utils.py`** – motion models and positional encoding utilities  
- **`main.ipynb`** – Jupyter notebook for running reconstruction on new bursts  

---

## Usage
1. Place burst frames in a folder, named sequentially as `0_000.png`, `1_000.png`, ..., `{N-1}_000.png`, where $N$ is the number of frames.
2. Open and run `main.ipynb` in Google Colab or locally (A100 GPU recommended).  
3. Select:
   - Motion model (`Homography`, `Optical Flow`, `Optical Flow w/o TV loss`, or `None`)
   - Positional encoding type (`Fourier`, `Hash`, `NeRF`, `SIREN`, or `None`)
   - Key hyperparameters (e.g., encoding scale $\sigma$)

Ablation scripts for motion, encoding, and hyperparameter studies are also provided.

---

## Dataset
The dataset used in this work is publicly available at:  
https://github.com/amirrezavazifeh/Fiber-Bundle-Imaging-Dataset


