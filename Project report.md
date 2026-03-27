# 🏠 Project Report: House Price Prediction MLP

## 1. Introduction
This project demonstrates the power and robustness of **Multi-Layer Perceptrons (MLP)**. We implement a neural network from scratch using only **NumPy** to predict house prices. By using Stochastic Gradient Descent with a custom adaptive learning rate logic (Adam-inspired), we show that deep neural networks can be effective even with small datasets.

---

## 2. Concepts & Mathematics

### Architecture
The model uses:
- **Input Layer:** 2 neurons (Square Feet, Bedrooms)
- **Hidden Layers:** 2 layers with 8 neurons each
- **Output Layer:** 1 neuron (Price)

### Activation Function: Leaky ReLU
We chose **Leaky ReLU** to prevent the "dying neuron" problem. Unlike standard ReLU, it allows a small gradient when the input is negative, ensuring all neurons stay active during backpropagation.

**Formula:**
$$f(x) = \begin{cases} x & \text{if } x > 0 \\ 0.01x & \text{if } x \leq 0 \end{cases}$$

**Derivative:**
$$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0.01 & \text{if } x \leq 0 \end{cases}$$

---

## 3. Working Principle
The model operates on a normalized input array ($8 \times 2$). Weights and biases are initialized using **He Initialization** to prevent the vanishing gradient problem, ensuring the model isn't hypersensitive at the start of training.

### Custom Adaptive Optimizer
To escape local minima and find the global minimum, we implemented an adaptive learning rate:
- **Increase:** If weight updates are negligible, the learning rate increases to push the model forward.
- **Decrease:** If weights are "bouncing" (high volatility), the learning rate decays to stabilize convergence.

---

## 4. Key Observations & Results
The model trained over **10,000 iterations**. The custom optimizer played a huge role in smoothing the loss curve and reaching a higher level of accuracy compared to standard SGD.

### Performance Comparison

| Standard SGD (No Adam) | Custom Adaptive Learning Rate |
| :---: | :---: |
| ![Standard Loss](loss_without_adam.png) | ![Adaptive Loss](loss_with_adam.png) |

**Analysis:** The optimized graph converges much faster and remains significantly smoother throughout the training process.

---

## 5. Implementation Details
- **He Initialization:** Prevents vanishing gradients, especially important when using Leaky ReLU.
- **Data Normalization:** Features were scaled (Area / 2000, Bedrooms / 5) to keep input values within a range that prevents gradient explosion.
- **Manual Backpropagation:** Used the Chain Rule and partial derivatives to propagate error from the output layer back through the hidden layers.

---

## 6. Conclusion
This project provided deep insight into the "under the hood" mechanics of Deep MLPs. By building the math from scratch, I successfully modeled a non-linear regression problem and observed the critical importance of optimization algorithms in neural network training.







