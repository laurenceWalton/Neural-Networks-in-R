rm(list = ls())

# (a)

# Function: softmax_matrix
# Description: Apply softmax to a matrix (N training examples)
#
# Arguments:
#   z_matrix: A q (no. nodes) X N (no. training examples) matrix
#
# Returns:
#   Matrix of same dimensions as z_matrix with softmax applied to each column
softmax_matrix <- function(z_matrix){
  
  softmax_vector = function(z){
    exp(z)/sum(exp(z))
  }
  
  res = matrix(0, nrow = dim(z_matrix)[1], ncol = dim(z_matrix)[2])
  for (j in 1:dim(z_matrix)[2]){  
    vec      = z_matrix[, j]        
    res[, j] = softmax_vector(vec)
  }
  return(res)
}


# Function: log_softmax_matrix
# Description: Apply log of the softmax to a matrix (N training examples),
# avoiding over/under flow of values from normal softmax
#
# Arguments:
#   z_matrix: A p (no. features) X N (no. training examples) matrix
#
# Returns:
#   Matrix of same dimensions as z_matrix with log softmax applied to each column
log_softmax_matrix = function(z_matrix){
  
  log_softmax_vector = function(z){
    # Subtract the maximum value
    x = z - max(z)
    # Equation rearranging 
    log_softmax_z = x - log(sum(exp(x)))
    return(log_softmax_z)
  }
  
  # Apply the log_softmax_vector function to each column of z_matrix
  result_matrix = apply(z_matrix, 2, log_softmax_vector)
  
  return(result_matrix)
}


# (b)


# Function: tanh
# Description: Hyperbolic tangent function
tanh = function(z){
  # Do NOT use the bare calculation commented out - you may spend very long thinking 
  # the softmax is causing over/under flow while it was tanh all along
  #(exp(z) - exp(-z))/(exp(z) + exp(-z))
  base::tanh(z)
}

# Function: neural_net
#
# Arguments:
#   X: Input matrix (N x p)
#   Y: Output matrix (N x q)
#   theta: A parameter vector (all of the parameters)
#   nu: Regularisation hyperparameter 
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
  
  # Use log_softmax_matrix instead of softmax_matrix
  a2    = softmax_matrix(z2)
  log_a2 = log_softmax_matrix(z2)
  
  # Cross-entropy error function for multi-class problems (q-dimensional response)
  cross_entropy <- -sum(t(Y) * log_a2)/N
  
  # Debugging
  if (is.na(cross_entropy)){
    browser()
  }
  # Cross-entropy error with L1 penalty applied
  L1 <- cross_entropy + nu/N * (sum(abs(W1)) + sum(abs(W2)))
                            
  # Return predictions and error:
  return(list(out=a2, lsm=log_a2, z2=z2, cross_entropy=cross_entropy, L1=L1))
}


#
# (c)
#

set.seed(2023)

# Read in the data:
dat = read.table('Hawks_Data_2023.txt',h= T)
X   = as.matrix(dat[,4:5], ncol=2) 
Y   = as.matrix(dat[,1:3], ncol = 3)
N   = dim(dat)[1]

# Split data into training and test sets 
set = sample(1:N, 0.5*N, replace=FALSE)
X_train       = matrix(X[set,], ncol=2)
Y_train       = matrix(Y[set,], ncol = 3)

X_validation  = matrix(X[-set,], ncol=2)
Y_validation  = matrix(Y[-set,], ncol = 3)

# Return the error with L1 penalty applied when fitting
obj <- function(pars) {
  res <- neural_net(X_train, Y_train, pars, nu)
  return(res$L1)
}

# Network parameters
p = 2
q = 3
m = 8
npars = p*m+m*q+m+q

seq       = 30
val_error = rep(NA, seq)
lams      = exp(seq(-11, 1, length.out=seq))

for (i in 1:seq){
  nu      = lams[i]
  theta   = runif(npars,-1,1)
  res_opt = nlm(obj, theta, iterlim=250)
  
  res_val = neural_net(X_validation, Y_validation, res_opt$estimate, 0) 
  
  val_error[i] = res_val$cross_entropy
  print(paste0('Val_Run_',i))
}

plot(val_error ~ lams, type = 'b', pch=19, lty=2, xlab = "nu (ν)", ylab = "Cross-Entropy Error", col = 4, lwd = 2, ylim=c(0,1), main="Validation Set Error vs Nu (L1-regularisation hyperparameter)")
# Vertical line at the value of lams that corresponds to the minimum val_error
abline(v=lams[which.min(val_error)], col=2, lty=2)
# Horizontal line at minimum val_error
abline(h=min(val_error), col=2, lty=2)

# Minimum validation error
min(val_error)
# Corresponding nu
lams[which.min(val_error)]


#
# (d)
#

# Function: reLU
# Description: Apply reLU to a vector (single training example). Uses pmax(z,0) to apply the max-function element-wise
reLU = function(z){
  pmax(z,0)
}

# Function: neural_net_reLU
# Description: Identical implementaion as question (b) but with reLU activation
# for hidden layer
neural_net_reLU = function(X, Y, theta, nu)
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
  a1    = apply(t(W1)%*%a0 +  b1, c(1,2), reLU)
  
  # qxN (3x148)        
  z2    = t(W2)%*%a1 + b2
  
  # Use log_softmax_matrix instead of softmax_matrix
  a2    = softmax_matrix(z2)
  log_a2 = log_softmax_matrix(z2)
  
  # Cross-entropy error function for multi-class problems (q-dimensional response)
  cross_entropy <- -sum(t(Y) * log_a2)/N
  
  # Debugging
  if (is.na(cross_entropy)){
    browser()
  }
  # Cross-entropy error with L1 penalty applied
  L1 <- cross_entropy + nu/N * (sum(abs(W1)) + sum(abs(W2)))
  
  # Return predictions and error:
  return(list(out=a2, lsm=log_a2, z2=z2, cross_entropy=cross_entropy, L1=L1))
}

# Return the error with L1 penalty applied when fitting - in this case use modified NN
obj_reLU <- function(pars) {
  res <- neural_net_reLU(X_train, Y_train, pars, nu)
  return(res$L1)
}

# Network parameters
p = 2
q = 3
m = 8
npars = p*m+m*q+m+q

seq       = 30
val_error_reLU = rep(NA, seq)
lams      = exp(seq(-11, 1, length.out=seq))

for (i in 1:seq){
  nu      = lams[i]
  theta   = runif(npars,-1,1)
  res_opt = nlm(obj_reLU, theta, iterlim=250)
  
  res_val_reLU = neural_net_reLU(X_validation, Y_validation, res_opt$estimate, 0) 
  
  val_error_reLU[i] = res_val_reLU$cross_entropy
  print(paste0('Val_Run_',i))
}

plot(val_error_reLU ~ lams, type = 'b', pch=19, lty=2, xlab = "nu (ν)", 
     ylab = "Cross-Entropy Error", col = 9, lwd = 2, ylim=c(0,1), 
     main="Validation Set Error vs Nu (L1-regularisation hyperparameter)")
# NOTE: to super impose the val_error with tanh NN, must run that code in prev.
# question first to obtain val_error vector. Ensure all params the same (iterlim, seq)
lines(val_error ~ lams, type = 'b', pch=19, lty=2, col=4, lwd=2)
legend("topright", col=c(4, 9), legend=c("tanh", "reLU"), pch=19)
