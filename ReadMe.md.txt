# Feature Selection Using BTCOA (MATLAB)

This repository provides the official MATLAB implementation of a hybrid
feature selection framework based on a graph-based filtering stage combined
with the Binary Tangent–Cotangent Optimization Algorithm (BTCOA).

The proposed method is designed for high-dimensional low-sample-size (HDLSS)
datasets and is suitable for large-scale and high-performance computing (HPC)
environments.

---

## Main Contributions
- Hybrid filter–wrapper feature selection framework
- Graph-based feature filtering for dimensionality reduction
- Wrapper-based optimization using BTCOA
- Support for nested cross-validation to avoid selection bias
- Fully reproducible MATLAB implementation

---

## Requirements
- MATLAB R2023a or later
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox (optional)

---

## Repository Structure

Feature-Selection---BTCOA/
│
├── src/
│   ├── FeatureSelectionUsingTanCot.m
│   ├── FilterMethod.m
│   ├── SortPopulation.m
│   ├── Evaluatefeatures.m
│   ├── LoadData.m
│   ├── ManageData.m
│   ├── ManageData_NestedCV.m
│   └── MainCodeFS.m
│
├── preprocessing/
│   └── ManageData_NestedCV.m
│
├── evaluation/
│   ├── Metric.m
│   └── PrecisionRecall.m
│
├── datasets/
│   └── README.md
│
├── results/
│   └── README.md
│
├── run_experiment.m
├── README.md
└── LICENSE

---

## Usage (MATLAB)

1. Clone the repository:
```bash
git clone https://github.com/selin1392/Feature-Selection---BTCOA.git
