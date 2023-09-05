# Neural Networks in R

## Description
Implementation and analysis of a 1-layer artificial neural network in R, designed to classify bird species based on wing length and body weight. Utilizes a softmax activation function for the output layer to accommodate one-hot encoded responses. Both tanh and ReLU activations are evaluated in the hidden layer. The model is trained using a cross-entropy loss function and R's 'nlm' optimizer. L1 regularization is also employed, and the optimal \( \nu \) (L1 penalty multiplier) is determined through validation analysis.

## Dataset
The dataset, `Hawks_Data_2023.txt`, consists of 148 observations featuring the following variables:
- `SpecA`, `SpecB`, `SpecC`: One-hot encoded labels indicating species (A, B, or C).
- `Wing`: Wing length measured in meters.
- `Weight`: Body weight in kilograms.
