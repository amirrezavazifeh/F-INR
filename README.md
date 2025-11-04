# Neural Field Fiber Endoscopy
<p align="center">
  <img src="Principle.png" width="1000">
</p>

---

## Overview
Fiber-optic imaging systems enable minimally invasive **in-vivo** imaging but suffer from sampling artifacts (e.g., honeycomb patterns) due to discrete and non-uniform fiber layouts.  
This repository implements an **unsupervised test-time training method** for removing these artifacts from a burst of misaligned frames — without requiring ground truth, calibration, or prior knowledge of fiber geometry.

Our method jointly optimizes:
- A **motion network** that models frame-to-frame transformations.  
- A **scene representation network** that reconstructs a super-resolved canonical view using coordinate-based implicit representation and positional encoding.

---

## Method Summary
Each input frame is modeled as a warped observation of a canonical scene.  
For each pixel coordinate $(x, y, t)$:

$$
\hat{I}(x, y, t) = f_\theta\left(\gamma\left(T_{g_\phi}(x, y, t)\right)\right)
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

Both networks are multilayer perceptrons (MLPs). To mitigate spectral bias, we apply a positional encoding $\gamma(\cdot)$ to spatial coordinates before mapping to RGB values.

---

## Reconstruction Examples
<p align="center">
  <img src="Reconstruction Examples.png" width="1000">
</p>

---

## Repository Structure
- **`utils.py`** – motion models and positional encoding utilities  
- **`main.ipynb`** – Jupyter notebook for running reconstruction on new bursts  

---

## Usage
1. Place burst frames in a folder, named sequentially
2. Open and run `main.ipynb` in Google Colab or locally (A100 GPU recommended).  
3. Select:
   - Motion model (`homography`, `flow`, `flow_no_tv`, or `none`)
   - Positional encoding type (`Fourier`, `hash`, `NeRF`, `SIREN`, or `none`)
   - Key hyperparameters (e.g., encoding scale $\sigma$)

Ablation scripts for motion, encoding, and hyperparameter studies are also provided.

---

**Repository:** [github.com/amirrezavazifeh/Neural-Field-Fiber-Endoscopy](https://github.com/amirrezavazifeh/Neural-Field-Fiber-Endoscopy)
