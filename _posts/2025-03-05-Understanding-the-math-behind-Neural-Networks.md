---
layout: post
title: "The Elegant Math Behind Neural Networks"
date: 2025-03-05
categories: [machine-learning, neural-networks, mathematics]
tags: [neural-networks, deep-learning, mathematics, toy-model, tutorial]
math: true

description: "A step-by-step walkthrough of the mathematical foundations of neural networks using a simple toy model"
---

When I was trying to learn about neural networks, I found thousands of resources explaining their architecture and how they work conceptually. However, I struggled to find a simple resource that walked through the actual math with a concrete example where I could connect the dots.

It's just like learning how to add 1+3 by hand and understanding *why* 1+3 = 4, instead of just using a calculator and accepting the answer without knowing what happens under the hood.

## Why Understanding the Math Matters

I strongly believe it's super important to know how models work and their limitations to truly benefit from them. I'm not saying you need to understand every detail of a complex model, but understanding a simple toy model is foundational.

You don't typically add 3424234234 + 978779 mentally—you use a calculator. Similarly, you don't need to understand how an entire complex neural network works, but you should understand the basic principles, just like knowing 1+2 = 3.

## Building a Toy Neural Network Model

In this post, I'll build a toy model of a neural network step-by-step on paper. Once you understand this, more complex models will make more sense.

### The Problem Definition

I want to build a TOY model that can classify a car. Specifically, a model that takes an image (just 4 pixels!) as input, processes it, and classifies whether it's a car (1) or not (0).

### Model Architecture

First, we need to decide on the architecture:

* **Input layer**: 4 neurons (1 per pixel)
* **Hidden layer**: 3 neurons (arbitrary choice for simplicity)
* **Output layer**: 1 neuron (car (1) or not (0))

![Neural Network Architecture](/images/neural-network-architecture.png)

### What Do These Layers Do?

**Input Layer**: This is the front door for the data (our 4 pixels) to enter the model. No computation happens here—data is just passed to the next layer.

**Hidden Layer**: This is where the processing happens. The neurons in the hidden layer learn patterns (such as edges, curves, and shapes of the car).

**Output Layer**: This is where prediction happens. It takes the hidden layer patterns (features) and maps them to interpretable results (1 for car, 0 for not).

## Training the Model

### Input Data

Let's start with our input: a 4-pixel grayscale image. We'll flatten the image, converting it to a 1D vector to make the mathematical operations easier.

Here's an example of a flattened 4-pixel grayscale image represented as a column vector:

$$
\mathbf{x} = 
\begin{bmatrix}
0.2 \\
0.5 \\
0.1 \\
0.8
\end{bmatrix}
$$

> **Side note**: Neural networks process inputs as vectors without any spatial awareness. The network treats each element of the vector as an independent input feature without any predetermined relationship to other elements. It doesn't "know" that certain pixels are neighbors or that the data originally had a 2D structure.
>
> This is why Convolutional Neural Networks (CNNs) were developed specifically for image processing - they incorporate spatial awareness by using filters that operate on local regions of the input, preserving spatial relationships.

### Hidden Layer Processing

The hidden layer has 3 neurons and performs two transformations:
1. Linear transformation
2. Non-linear transformation

#### Linear Transformation

Each neuron is built on a linear regression formula:

$$y = wx + b$$

Where:
- $x$ is the input
- $w$ is the weight
- $b$ is the bias
- $y$ is the output

For our hidden layer with 3 neurons and an input of 4 pixels, we need a weight matrix $W_1$ of shape $3 \times 4$ and a bias vector $b_1$ of shape $3 \times 1$.

Let's randomly initialize these (in a real network, these would be learned):

$$
W_1 = 
\begin{bmatrix}
0.1 & 0.2 & -0.1 & 0.1 \\
0.3 & 0.1 & -0.3 & 0.2 \\
-0.2 & 0.3 & 0.4 & 0.1
\end{bmatrix}
$$

$$
b_1 = 
\begin{bmatrix}
0.5 \\
-0.3 \\
0.7
\end{bmatrix}
$$

Now we compute the linear transformation:

$$
z_1 = W_1 \mathbf{x} + b_1
$$

$$
z_1 = 
\begin{bmatrix}
0.1 & 0.2 & -0.1 & 0.1 \\
0.3 & 0.1 & -0.3 & 0.2 \\
-0.2 & 0.3 & 0.4 & 0.1
\end{bmatrix}
\begin{bmatrix}
0.2 \\
0.5 \\
0.1 \\
0.8
\end{bmatrix} +
\begin{bmatrix}
0.5 \\
-0.3 \\
0.7
\end{bmatrix}
$$

$$
z_1 = 
\begin{bmatrix}
(0.1 \times 0.2) + (0.2 \times 0.5) + (-0.1 \times 0.1) + (0.1 \times 0.8) + 0.5 \\
(0.3 \times 0.2) + (0.1 \times 0.5) + (-0.3 \times 0.1) + (0.2 \times 0.8) + (-0.3) \\
(-0.2 \times 0.2) + (0.3 \times 0.5) + (0.4 \times 0.1) + (0.1 \times 0.8) + 0.7
\end{bmatrix}
$$

$$
z_1 = 
\begin{bmatrix}
0.62 \\
-0.01 \\
1.5
\end{bmatrix}
$$

#### Non-linear Transformation (ReLU)

After computing the weighted sum, we apply a ReLU activation function to introduce non-linearity.

**Why do we need this step?**

Without non-linear transformation, we'd only be modeling linear relationships (straight lines), which wouldn't be useful for complex patterns like curves, corners, and gradients in images.

ReLU is a simple function: $\text{ReLU}(x) = \max(0, x)$
- If the input is positive, output is that same number
- If the input is negative or zero, output is zero
  
![ReLU Activation Function](/images/relu-activation.svg)

Think of ReLU as an "on/off switch" for information. Information that aligns with what a neuron is looking for gets passed along (positive values), while information that doesn't match gets filtered out (negative values become zero).

Applying ReLU to our result:

$$
a_1 = \text{ReLU}(z_1) = 
\begin{bmatrix}
\max(0, 0.62) \\
\max(0, -0.01) \\
\max(0, 1.5)
\end{bmatrix} =
\begin{bmatrix}
0.62 \\
0 \\
1.5
\end{bmatrix}
$$

This means our hidden layer's 3 neurons have activations of [0.62, 0, 1.5]. If we passed a different 4-pixel image, say [0.3, 0.1, 0.5, 0.1], the activations would be different.

These activation values represent the "firing strength" of each neuron after processing the input. The pattern of which neurons activate and which don't is how the network encodes its "understanding" of the input.

### Output Layer Processing

The output layer takes the hidden layer activations and processes them similarly in two steps:
1. Linear transformation
2. Output activation function

#### Linear Transformation

With a single output neuron, we need a weight vector $W_2$ of shape $1 \times 3$ and a bias scalar $b_2$:

$$
W_2 = 
\begin{bmatrix}
0.5 & -0.3 & 0.8
\end{bmatrix}
$$

$$
b_2 = 0.2
$$

The linear transformation is:

$$
z_2 = W_2 a_1 + b_2
$$

$$
z_2 = 
\begin{bmatrix}
0.5 & -0.3 & 0.8
\end{bmatrix}
\begin{bmatrix}
0.62 \\
0 \\
1.5
\end{bmatrix} + 0.2
$$

$$
z_2 = (0.5 \times 0.62) + (-0.3 \times 0) + (0.8 \times 1.5) + 0.2 = 1.51
$$

#### Output Activation Function (Sigmoid)

The result, 1.51, doesn't mean much in our binary classification context. We need to transform it into a probability using the sigmoid function:

$$\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}$$

Applying sigmoid:

$$\text{sigmoid}(1.51) = \frac{1}{1 + e^{-1.51}} \approx 0.82$$

So our model predicts there's an 82% chance the image is a car!

![Forward Propagation Flow](/images/forward-propagation.svg)

## What's Next in the Training Process?

We've just done one forward pass with randomly initialized weights. In a real training process, we would:

1. **Calculate the error**: Compare the model's prediction (0.82) to the actual label (assuming 1 for car), and determine how far off we are.

2. **Backpropagation**: Use gradient descent to calculate how much each weight and bias contributes to the error.

3. **Update weights and biases**: Adjust the parameters to reduce the error.

4. **Repeat**: Pass new images through the updated model, calculate new predictions and losses, and continue until the model performs satisfactorily.

![Neural Network Learning Process](/images/learning-process.svg)

This toy example demonstrates the fundamental mathematics behind neural networks. While real-world neural networks contain many more layers and neurons, the basic principles remain the same:

1. Forward propagation through linear transformations
2. Application of non-linear activation functions
3. Error calculation
4. Backpropagation to update weights

Understanding these basics will help you grasp how more complex architectures like CNNs, RNNs, and Transformers function at their core.

In my next post, I'll cover the backpropagation algorithm in detail and show how the model actually learns from its mistakes.
