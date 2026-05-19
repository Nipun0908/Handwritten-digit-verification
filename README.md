# Handwritten-digit-verification
Identifying Handwritten Digits using Logistic Regression in PyTorch

This repository contains a simple PyTorch implementation of a Logistic Regression model trained to classify handwritten digits using the classic MNIST dataset. 

#Project Overview

Logistic Regression, despite its name, is a fundamental linear classification algorithm. When applied to image classification tasks like MNIST, the 2D image matrix is flattened into a 1D vector and passed through a single linear layer. The network maps the features directly to the class probabilities (0-9).

#Dataset
The *MNIST* dataset consists of 60,000 training images and 10,000 testing images of handwritten digits. Each image is a $28 \times 28$ pixels grayscale image, matching an input dimension of $784$.

#Repository Structure

* `model.py`: Defines the `LogisticRegressionModel` class architecture using `torch.nn`.
* `train.py`: Handles dataset downloading, preprocessing, pipeline optimization, model training, and performance evaluation.
* `logistic_regression_mnist.pth`: The serialized weights file generated post-training (created after running script).

#Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/mnist-logistic-regression-pytorch.git](https://github.com/your-username/mnist-logistic-regression-pytorch.git)
   cd mnist-logistic-regression-pytorch

1.	Install the required dependencies:
pip install torch torchvision

#Usage:

To train the model and see evaluation metrics, run the training script:
python train.py

The script will automatically download the MNIST dataset to a local ./data folder, complete 3,000 iterations of stochastic gradient descent (SGD), print loss/accuracy intervals, and save the final model parameters.

#Model Performance:

<img width="826" height="240" alt="Screenshot (389)" src="https://github.com/user-attachments/assets/357d2cda-ab43-4f17-9d35-db14d1d99c32" />




