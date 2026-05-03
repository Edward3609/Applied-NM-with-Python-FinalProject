# Gaussian Beam Simulator

This project simulates laser beam propagation through optical systems using:

- ABCD Matrix Method (Analytical)
- FFT Fresnel Propagation (Numerical)
  - NumPy FFT
  - Custom Cooley–Tukey FFT

## Features

- Beam envelope visualization (side profile)
- Final beam intensity visualization
- Beam width comparison (ABCD vs FFT)
- Wavefront curvature comparison
- Power conservation validation
- Stability check for optical cavities (2+ mirrors)

## How to Run

Run either version:
-Enter initial beam waist radius(example: .002)
-Enter beam's wavelenth(example: 540e-9)
-Enter optical components and their specs(example: mirror "ENTER"
                                                   .2     "ENTER").
-Enter "done" when finished entering componenets.
```bash
python cooley_tukey_simulator.py
