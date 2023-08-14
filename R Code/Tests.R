# ------------------------ TESTS for (a) ------------------------ 

# Create a vector for a single training example
z_vector = matrix(c(0.1, 0.9, 0.7), nrow=1, ncol=3)
# Create a 3x4 matrix of the same z values for each column (4 training examples w/ same values)
z_matrix = matrix(c(0.1, 0.9, 0.7, 0.1, 0.9, 0.7, 0.1, 0.9, 0.7, 0.1, 0.9, 0.7), nrow = 4, ncol = 3, byrow = TRUE)

res_vector = softmax_vector(z_vector)
res_matrix = softmax_matrix(z_matrix)
print("Dimensions for res_matrix:")
dim(res_matrix)

# Define a tolerance level for the comparison
tolerance <- 1e-6 

# Test softmax_vector 
if (abs(sum(res_vector) - 1) < tolerance) {
  print("The values add up to approximately 1.")
} else {
  print("The values do not add up to approximately 1.")
}

# Test softmax_matrix
for (i in 1:dim(res_matrix)[2])
  if (abs(sum(res_matrix[,i]) - 1) < tolerance) {
    print("The values add up to approximately 1.")
  } else {
    print("The values do not add up to approximately 1.")
  }

# --------------------------------------------------------------- 




# ----  TESTS for (b) ----

# -----------------------------------------------------