# Neural Field Fiber Endoscopy

**Unsupervised Fiber Bundle Artifact Removal Using Neural Fields**

<p align="center">
  <img src="principle.png" width="600">
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
For each pixel coordinate \((x, y, t)\):

\[
\hat{I}(x, y, t) = f_\theta\bigl(\gamma(T_{g_\phi}(x, y, t))\bigr)
\]

where:
- \(T_{g_\phi}\): motion transformation (e.g., homography or optical flow)  
- \(\gamma(\cdot)\): positional encoding  
- \(f_\theta\): scene representation network  
- Both networks are trained jointly using an \(L_2\) reconstruction loss.

---

## Repository Structure

- **`utils.py`** – motion models and positional encoding utilities  
- **`main.ipynb`** – Jupyter notebook for running reconstruction on new bursts  

---

## Usage

1. Place burst frames in a folder, named sequentially:



2. Open and run `main.ipynb` in Google Colab or locally (A100 GPU recommended).  
3. Select:
- Motion model (`homography`, `flow`, or `none`)
- Positional encoding type (`Fourier`, `hash`, etc.)
- Key hyperparameters (e.g., encoding scale `σ`)

Ablation scripts for motion, encoding, and hyperparameter studies are also provided.

---

## Citation

If you use this work, please cite:
> Vazifeh, A.R. *et al.*, “Unsupervised Fiber Bundle Artifact Removal Using Neural Fields,” 2025.

---

**Repository:** [github.com/amirrezavazifeh/Neural-Field-Fiber-Endoscopy](https://github.com/amirrezavazifeh/Neural-Field-Fiber-Endoscopy)
