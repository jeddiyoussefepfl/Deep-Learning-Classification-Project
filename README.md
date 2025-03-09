# EPFL Deep Learning Classification Project

This repository contains an implementation of various machine learning methods for a classification task, including PCA for dimensionality reduction and multiple deep learning architectures such as a Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and a Vision Transformer (ViT). This project is part of an academic assignment at EPFL.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Acknowledgments](#acknowledgments)


## Overview

The goal of this project is to implement and evaluate different machine learning methods for image classification. The repository includes:
- **Dummy Classifier:** A baseline method that returns random predictions.
- **PCA:** A principal component analysis implementation to reduce feature dimensionality.
- **Neural Networks:**
	- **MLP (Multi-Layer Perceptron):** A fully-connected network for classification.
	- **CNN (Convolutional Neural Network):** A network that uses convolutional layers for feature extraction.
	- **Vision Transformer (ViT):** A transformer-based network for image classification.

The project uses TensorBoard to log training metrics and supports training on CPU or GPU.

## Features

- **Data Processing:** Normalization and optional PCA-based dimensionality reduction.
- **Modular Design:** Multiple architectures (MLP, CNN, ViT) for classification.
- **Training & Evaluation:** Training routines using PyTorch with support for validation splits.
- **Logging:** Integration with TensorBoard for monitoring training progress.
- **Unit Testing:** Tests to validate the implementations.

## Requirements

- Python 3.6+
- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)
- [TorchInfo](https://pypi.org/project/torchinfo/)
- [SciPy](https://scipy.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

## Installation

1. **Clone the Repository:**

	```bash
	git clone https://github.com/yourusername/EPFL-Deep-Learning-Classification-Project.git
	cd EPFL-Deep-Learning-Classification-Project

## Optional 

	python -m venv venv

	source venv/bin/activate  # On Windows: venv\Scripts\activate

	pip install -r requirements.txt

## Usage



The main training and evaluation script is main.py. You can specify various command-line arguments:
- **--data:** Path to the dataset directory.
- **--nn_type:** Type of neural network to use (mlp, cnn, or transformer).
- **--nn_batch_size:** Batch size for training.
- **--use_pca:** Flag to enable PCA dimensionality reduction.
- **--pca_d:** Number of principal components if PCA is used.
- **--lr:** Learning rate for training.
- **--max_iters:** Number of training epochs.
- **--test:** Flag to indicate testing on the entire dataset.

## Example

	python main.py --data dataset --nn_type cnn --nn_batch_size 64 --lr 1e-5 --max_iters 100 --use_pca --pca_d 100

## Testing

To run the unit tests, execute:

	python test_ms2.py

## Acknowledgments

This project was developed as part of the coursework for the EPFL Introduction to Machine Learning class CS-233. Special thanks to the course instructors and TAs.