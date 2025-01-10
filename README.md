# Neural Network Verifier

A verifier for neural networks based on DeepPoly convex relaxation, developed for the **Reliable and Trustworthy AI 2024** course. This project evaluates robustness across various architectures and datasets (MNIST, CIFAR-10).

For more details about the underlying methodology, refer to the [DeepPoly paper](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf).

---

## Project Description

This project aims to verify the robustness of trained neural networks against adversarial perturbations within a specified radius in the L∞ norm ball. Specifically, it assesses whether a network maintains consistent output classifications when inputs are subjected to small perturbations, ensuring reliability in uncertain environments.

**Example:**

Consider a neural network trained to recognize handwritten digits from the MNIST dataset. 
The verifier evaluates if the network consistently classifies this image as '7' when subjected to perturbations within a defined L∞ norm radius (e.g., ε = 0.01). If the network's output remains unchanged for all perturbed inputs within this radius, the network is considered robust for this input.

![Adversarial attack. Source: [Research Gate](https://www.researchgate.net/figure/Adversarial-attack-causing-miss-classification-in-MNIST-data-set_fig1_322950125)](./adversarial_example.png)

---

## Features

- Verifies fully connected, convolutional, and skip-connection-based networks.
- Includes 17 pretrained models and sample test cases for evaluation.
- Supports perturbation certification with pixel values normalized between 0 and 1.

---

## Folder Structure

- **`code/`**  
  - `networks.py`: Defines network architectures (uses `MLP.py` and `skip_block.py`).
  - `verifier.py`: Verifier implementation (modify the `analyze` function).
  - `evaluate.sh`: Evaluates all models and test cases.
  - `utils/`: Helper methods for loading and initialization.

- **`models/`**  
  Contains 17 pretrained network weights.

- **`test_cases/`**  
  Includes ground truth (`gt.txt`) and 2 test cases per model.

---

## Setup

Using virtualenv:  
```bash
virtualenv venv --python=python3.10
source venv/bin/activate
pip install -r requirements.txt