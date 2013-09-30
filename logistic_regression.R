# Example of Batch Gradient Descent for Logistic Regression in R
# Data Set: 'German Credit' from UCI machine learning repository
# Data Set can be downloaded from: http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
#
# Author: Alexandre Vilcek - Oracle
# Version: 1.0

# load input data into memory
# you should replace <path-to-file> with the path where you downloaded the input file (german.data-numeric)
data <- as.matrix(read.table("/home/avilcek/DataSets/UCI/CreditRisk/german.data-numeric"))

# rescale the output to be 0 or 1 (needed for binary classification using the logistic function) instead of 1 or 2
data[,ncol(data)] <- data[,ncol(data)]-1

# split data into train and test sets (X), and also extract the correspondent vectors of observed values (y)
# first 20% for testing / remaining 80% for training
# add x0=1 to all rows of X (convention)
# scale data to achieve faster convergence
index <- seq(1,nrow(data)*0.2)
X_test <- cbind(1,scale(data[index,1:ncol(data)-1],center=F))
y_test <- as.matrix(data[index,ncol(data)])
X_train <- cbind(1,scale(data[-index,1:ncol(data)-1],center=F))
y_train <- as.matrix(data[-index,ncol(data)])

# initialize parameters vector theta
theta <- as.matrix(rep(0,ncol(data)))

# hypothesis function
hypot <- function(z) {
  1/(1+exp(-z))
}

# gradient of cost function
gCost <- function(t,X,y) {
  1/nrow(X)*(t(X)%*%(hypot(X%*%t)-y))
}

# cost function optimization through batch gradient descent (training)
# alpha = learning rate
# steps = iterations of gradient descent
# tol = convergence criteria
# convergence is measured by comparing L2 norm of current gradient and previous one
alpha <- 0.3
tol <- 1e-6
step <- 1
while(T) {
  p_gradient <- gCost(theta,X_train,y_train)
  theta <- theta-alpha*p_gradient
  gradient <- gCost(theta,X_train,y_train)
  if(abs(norm(gradient,type="F")-norm(p_gradient,type="F"))<=tol) break
  step <- step+1
}

# hypothesis testing
# counts the predictions from the test set classified as 'good' and ' bad' credit and compares with the actual values
y_pred <- hypot(X_test%*%theta)
result <- xor(as.vector(round(y_pred)),as.vector(y_test))
corrects = length(result[result==F])
wrongs = length(result[result==T])
cat("steps: ",step,"\n")
cat("corrects: ",corrects,"\n")
cat("wrongs: ",wrongs,"\n")
cat("accuracy: ",corrects/length(y_pred),"\n")

# The execution of the code above should give a result like this:
# steps:  1330 
# corrects:  151 
# wrongs:  49 
# accuracy:  0.755
# The results above mean that the algorithm converged in 4 steps of Batch Gradient Descent and got 75.5% of the test dataset correct (the test dataset was not used in the training step).

# Final remarks:
# this is a very simple example that could be enhanced in several ways
# for exemple: by introducing a regularization factor in the cost function and by performing feature selection before training
