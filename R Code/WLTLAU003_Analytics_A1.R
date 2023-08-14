rm(list = ls())

# (a)

# Function: softmax_vector
# Description: Apply Softmax to a vector (single training example).
#
# Arguments:
#   z: A vector with a numeric value for each neuron to be included in the 
#      softmax calculation, i.e. the output layer
#
# Returns:
#   A vector of the same size as z with softmax applied to each value
softmax_vector = function(z){
  exp(z)/sum(exp(z))
}

# Function: softmax_matrix
# Description: Apply softmax to a matrix (N training examples)
#
# Arguments:
#   z_matrix: A q (no. nodes) X N (no. training examples) matrix
#
# Returns:
#   Matrix of same dimensions as z_matrix with softmax applied to each column
softmax_matrix = function(z_matrix){
  res = matrix(0, nrow = dim(z_matrix)[1], ncol = dim(z_matrix)[2])
  for (j in 1:dim(z_matrix)[2]){  
    vec      = z_matrix[, j]        
    res[, j] = softmax_vector(vec)
  }
  return(res)
}


# (b)


# Read in the data:
dat = read.table('Hawks_Data_2023.txt',h= T)

X   = as.matrix(dat[,4:5], ncol=2)
Y   = as.matrix(dat[,1:3], ncol = 3)

# Function: tanh
# Description: Hyperbolic tangent function
tanh = function(z){
  # Hyperbolic tangent function
  (exp(z) - exp(-z))/(exp(z) + exp(-z))
}
  

# Function: neural_net
# Description: 
#
# Arguments:
#   X: Input matrix (N x p)
#   Y: Output matrix (N x q)
#   theta: A parameter vector (all of the parameters)
#   nu: Regularisation parameter 
# Returns:
#   
neural_net = function(X, Y, theta, nu)
{
  # Relevant dimensional variables:
  N     = dim(X)[1]
  p     = dim(X)[2]
  q     = dim(Y)[2]
  # Number of nodes (fixed at 8)
  m     = 8
  
  # Populate weight-matrix and bias vectors:
  index = 1:(p*m)
  W1    = matrix(theta[index], p, m) 
  index = max(index)+1:(m*q)
  W2    = matrix(theta[index], m, q)
  index = max(index)+1:(m)
  b1    = matrix(theta[index], m, 1)
  index = max(index)+1:(q)
  b2    = matrix(theta[index], q, 1)
  
  # Updating equations (matrix form)
  # mxN (8x148)
  b1    = matrix(rep(b1, N), nrow=m, ncol=N)
  
  # qxN (3x148)
  b2    = matrix(rep(b2, N), nrow=q, ncol = N)
  
  # pxN (2x148)
  a0    = matrix(t(X), p, N)

  # mxN (8x148)   
  a1    = apply(t(W1)%*%a0 +  b1, c(1,2), tanh)

  # qxN (3x148)        
  z2    = t(W2)%*%a1 + b2
  a2    = softmax_matrix(z2)
  
  # Cross-entropy error function for multi-class problems (q-dimensional response)
  cross_entropy = -sum(Y * log(t(a2))) / N
  # Cross-entropy error with L1 penalty applied
  L1 = (cross_entropy + nu * (sum(W1)+sum(W2))) / N
  
  # Return predictions and error:
  return(list(out=a2, cross_entropy=cross_entropy, L1 = L1))
}


# Create a color vector for each Y value
colors <- c("red", "green", "blue")[apply(Y, 1, which.max)]

# Create a scatter plot
plot(X[, 1], X[, 2], col = colors, pch = 16, xlab = "Wing", ylab = "Weight", main = "Scatter Plot")
legend("topright", legend = c("Y1", "Y2", "Y3"), col = c("red", "green", "blue"), pch = 16)

# Network parameters
p = 2
q = 3
m = 8
npars = p*m+m*q+m+q
theta = runif(npars,-1,1)
nu = 0.2

res = neural_net(X,Y,theta, 0.2)
res

  
  









