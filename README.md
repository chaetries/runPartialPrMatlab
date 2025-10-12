
# Partial (3x4) MM Physical Realizability Mask (MATLAB)
----
<img width="740" height="669" alt="Screenshot 2025-10-12 at 15 37 11" src="https://github.com/user-attachments/assets/1f26987c-4ce5-4e5a-b8ab-bbb6b74e69f4" />

**Figure 1.** Visualization of isolated sample regions showing (A) grey-scale intensity, 
(B) CCP ground-truth physical realizability (PR) mask, and (C) machine-learning–predicted PR mask. The predicted PR mask closely reproduces the CCP-derived reference, 
demonstrating accurate spatial mapping of physically realizable polarization states from partial Mueller matrix inputs.

This repository provides MATLAB scripts to generate Physical Realizability (PR) masks
from Mueller Matrix data using a pre-trained ONNX MLP model (or XGBoost model).
The model predicts whether each pixel in the Mueller Matrix is physically realizable (PR = 1, Not PR = 0).

---

## Requirements:

- MATLAB
  - Deep Learning Toolbox
  - Deep Learning Toolbox Converter for ONNX Model Format
 
---

## Dataset Information:
The model was trained using mixed-tissue datasets:

  - Bulk samples in reflection (Cervix, Brain)
  - Microscope thin samples in transmission (Skin, Colon, Brain)

---

## Repository Structure:

model/
  pixel_mlp.onnx              - Pre-trained ONNX MLP model
  MLPFunction.m               - Auto-generated MATLAB ONNX function

sample/
  PPRIM.mat                   - Example partial Mueller Matrix data (3x4)

run_mlp.m                     - Main MATLAB script for PR mask generation
README.md                     - This file

---

## Input Data Format:
The input .mat file must contain one of the following:
  nM      : [H, W, 12]  - normalized partial Mueller matrix
  X_flat  : [N, 12]     - flattened Mueller matrix pixels

---

## Output:
After running PRMask_FromONNX_MLP.m, the same .mat file will include:
  pr_mask               : [H, W] or [N, 1] logical mask
  pr_mask_model         : model file used (.onnx)

Running the Code:
-----------------
1. Ensure model/pixel_mlp.onnx and run_mlp.m  are in your MATLAB path.
2. Make sure your input .mat file (e.g., sample/PPRIM.mat) contains nM or X_flat.
3. Run:
       >> PRMask_FromONNX_MLP
4. The generated mask will be saved back to the same file.

## Citation

---

## Notes


