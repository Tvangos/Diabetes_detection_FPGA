# Diabetes Classification on FPGA with hls4ml

Hardware acceleration of a neural network for diabetes classification using **Keras, QKeras, TensorFlow Model Optimization, hls4ml, and AMD/Xilinx Vitis HLS**.

The project explores how **pruning, quantization-aware training, and hardware optimization** can transform a small neural network into a resource-efficient FPGA implementation while maintaining its classification performance.

## Overview

The model is trained on the **OpenML Diabetes dataset**, which contains 768 patient records and 8 medical features.

The project follows the complete workflow:

```text
OpenML Diabetes Dataset
          │
          ▼
   Data Preprocessing
          │
          ▼
   Keras Neural Network
          │
          ├───────────────┐
          ▼               ▼
     Baseline        Optimization
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
              Pruning         Quantization
                 │                 │
                 └────────┬────────┘
                          ▼
                    QKeras Model
                          │
                          ▼
                       hls4ml
                          │
                          ▼
                     Vitis HLS
                          │
                          ▼
                    FPGA Hardware
```

## Neural Network

The implemented model is a compact feed-forward neural network designed for binary classification.

```text
Input: 8 features
       │
       ▼
Dense (64)
       │
Batch Normalization
       │
ReLU
       │
Dropout
       │
       ▼
Dense (32)
       │
Batch Normalization
       │
ReLU
       │
Dropout
       │
       ▼
Dense (1)
       │
Sigmoid
       │
       ▼
Diabetes Prediction
```

The eight input features are:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

## Dataset Preprocessing

The OpenML Diabetes dataset contains several zero-valued measurements that are physiologically unrealistic and are treated as missing values.

Zeros in the following features are replaced with the corresponding feature mean:

* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI

The features are then standardized using `StandardScaler`.

The dataset is split into:

* **80% training**
* **20% testing**

A fixed random seed is used to make the experiments reproducible.

Because the dataset contains an approximately 2:1 class imbalance, balanced class weights are applied during training.

## Optimization Techniques

Three main hardware-oriented optimizations are investigated.

### 1. Weight Pruning

Magnitude-based pruning removes weights that contribute relatively little to the model.

A pruning schedule is used to gradually increase sparsity, with a target sparsity of approximately **50%**.

This reduces the number of effective multiplications required by the hardware implementation.

### 2. Quantization-Aware Training

The optimized network uses **QKeras** to train the model while simulating reduced numerical precision during training.

The final optimized architecture uses low-bit fixed-point representations rather than standard 32-bit floating-point arithmetic.

This significantly reduces the hardware cost of arithmetic operations.

### 3. Reuse Factor

The `ReuseFactor` parameter in hls4ml controls the trade-off between:

* FPGA resource utilization
* latency
* parallelism

Different reuse factors were evaluated to find a suitable balance for the target FPGA.

## Model Configurations

Four configurations were evaluated:

| Model   | Pruning | Quantization | Description        |
| ------- | ------- | ------------ | ------------------ |
| Model 1 | ❌       | ❌            | Baseline           |
| Model 2 | ❌       | ✅            | Quantized          |
| Model 3 | ✅       | ❌            | Pruned             |
| Model 4 | ✅       | ✅            | Pruned + Quantized |

Model 4 represents the final hardware-oriented implementation.

## FPGA Implementation

The neural network is converted to synthesizable HLS C++ using **hls4ml** and synthesized using **AMD/Xilinx Vitis HLS**.

The final implementation targets:

**Xilinx Alveo U250**

Target clock period:

**5.0 ns**

The optimized design uses fixed-point arithmetic and a reuse factor of 16.

### Final Hardware Results

| Metric                 | Optimized Model |
| ---------------------- | --------------: |
| Target Clock Period    |         5.00 ns |
| Estimated Clock Period |         3.18 ns |
| Best Latency           |        6 cycles |
| Worst Latency          |        6 cycles |
| Initiation Interval    |         1 cycle |
| BRAM                   |               1 |
| DSP                    |               0 |
| Flip-Flops             |           1,275 |
| LUTs                   |          12,306 |
| URAM                   |               0 |

At a 5 ns clock period, the 6-cycle inference latency corresponds to approximately:

**30 ns per inference**

An **Initiation Interval (II) of 1** means that the architecture can accept a new input every clock cycle once the pipeline is filled.

## Power Analysis

The final FPGA implementation was also evaluated using Vivado power analysis.

| Power Metric        |   Value |
| ------------------- | ------: |
| Total On-Chip Power | 4.157 W |
| Static Power        | 2.966 W |
| Dynamic Power       | 1.191 W |

The optimized model requires **0 DSP slices**. The low-bit arithmetic allows the synthesis tools to map the required operations primarily to FPGA logic resources.

## Results

The optimization process substantially reduces the hardware requirements of the neural network.

Compared with the unoptimized implementation, the final pruned and quantized model achieves:

* **0 DSP usage**
* **12,306 LUTs**
* **1,275 FFs**
* **1 BRAM**
* **6-cycle inference latency**
* **II = 1**
* **3.18 ns estimated clock period**

The ROC/AUC evaluation shows that the model retains useful classification performance throughout the optimization pipeline, with only a small change in classification performance after quantization and pruning.

## Repository Structure

```text
.
├── src/
│   ├── Part1.ipynb
│   ├── Part2.ipynb
│   ├── Part3.ipynb
│   ├── Part4.ipynb
│   ├── bitstream.ipynb
│   ├── callbacks.py
│   └── plotting.py
│
├── results/
│   ├── baseline_util.png
│   ├── util_report.png
│   ├── timing_report.png
│   ├── power_report.png
│   ├── implementation.png
│   ├── implementation_2.png
│   ├── block_diagram.png
│   ├── opt1_*.png
│   ├── opt2_*.png
│   └── opt3_*.png
│
├── Report_DiabetesFPGA_3765_3406.pdf
├── Presentation_Diabetes_3765_3406.pdf
└── README.md
```

## Notebook Workflow

The notebooks correspond to the different stages of the project.

### `Part1.ipynb`

Baseline Keras neural network.

* Dataset loading
* Preprocessing
* Model construction
* Training
* Baseline evaluation
* hls4ml conversion

### `Part2.ipynb`

Quantization experiments.

* Reduced-precision models
* hls4ml fixed-point configuration
* Accuracy evaluation
* Hardware synthesis

### `Part3.ipynb`

Pruning experiments.

* Magnitude-based pruning
* Sparsity evaluation
* Accuracy comparison
* Hardware resource analysis

### `Part4.ipynb`

Combined optimization.

* Quantization-aware training
* Pruning
* Fixed-point hls4ml configuration
* Hardware synthesis
* Final accuracy and ROC comparison

### `bitstream.ipynb`

Further hardware deployment and synthesis workflow for the optimized model, including Vitis HLS synthesis and Vivado analysis.

## Requirements

The software environment requires Python together with the following major packages:

```text
tensorflow
qkeras
tensorflow-model-optimization
hls4ml
scikit-learn
numpy
matplotlib
```

Hardware synthesis requires an appropriate AMD/Xilinx toolchain, including:

* Vitis HLS
* Vivado

The notebooks also assume the corresponding Xilinx tools are available in the system environment.

## Running the Project

Clone the repository:

```bash
git clone https://github.com/Tvangos/Diabetes_detection_FPGA.git
cd Diabetes_detection_FPGA
```

Install the Python dependencies:

```bash
pip install tensorflow qkeras tensorflow-model-optimization hls4ml scikit-learn numpy matplotlib
```

Then open the notebooks in `src/` and execute them in order:

```text
Part1.ipynb
    ↓
Part2.ipynb
    ↓
Part3.ipynb
    ↓
Part4.ipynb
    ↓
bitstream.ipynb
```

> **Note:** Some notebooks expect generated datasets, trained model checkpoints, and hls4ml/Vitis HLS project directories produced during previous stages. These generated artifacts are not included in the repository.

## Tools & Frameworks

* **Python** — Project development and experimentation
* **TensorFlow / Keras** — Neural network training
* **QKeras** — Quantization-aware training
* **TensorFlow Model Optimization Toolkit** — Weight pruning
* **Scikit-learn** — Preprocessing and evaluation
* **hls4ml** — Neural-network-to-HLS conversion
* **Vitis HLS** — High-Level Synthesis
* **Vivado** — FPGA synthesis, implementation, timing and power analysis
* **Xilinx Alveo U250** — Target FPGA platform

## References

* hls4ml: https://github.com/fastmachinelearning/hls4ml
* hls4ml tutorials: https://github.com/fastmachinelearning/hls4ml-tutorial
* OpenML Diabetes Dataset: https://www.openml.org/d/37

## Authors

**Theodoros Vangos** \
**Vasilis Oikonomopoulos**

University project focused on neural-network compression and FPGA acceleration using hls4ml.
