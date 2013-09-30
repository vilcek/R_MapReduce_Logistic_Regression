# Example of Batch Gradient Descent for Logistic Regression in R expressed with MapReduce
# Data Set: 'German Credit' from UCI machine learning repository
# Data Set can be downloaded from: http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
#
# This example builds up onto 'logistic_regression.R'
# It breaks part of the computation of the gradient in partial sums that can be expressed by MapReduce
# It uses the Oracle R Connector for Hadoop (ORCH) - http://docs.oracle.com/cd/E41604_01/doc.22/e41238/orch.htm#CFHFIGAI - as the framework
# The ORCH framework allow an integration of the R environment with the Hadoop environment, providing an API for HDFS access and an API for the MapReduce framework in Hadoop
# To run this example it is required that you have an R environment and a Hadoop environment both with R and ORCH installed and communicating through the network
#
# Author: Alexandre Vilcek - Oracle
# Version: 1.0

# load ORCH libraries
library("ORCH")

# load input data into HDFS
# you should replace <path-to-file> with the path where you downloaded the input file (german.data-numeric)
# you should also replace <path-to-dfs> with the path on hdfs where the data will be stored
#data_test <- hdfs.put(data=read.table("/media/sf_DataSets/UCI/CreditRisk/german.data-numeric")[1:200,],dfs.name="/user/oracle/german.data-numeric/test",overwrite=T,rownames=T)
#data_train <- hdfs.put(data=read.table("/media/sf_DataSets/UCI/CreditRisk/german.data-numeric")[201:1000,],dfs.name="/user/oracle/german.data-numeric/train",overwrite=T,rownames=T)
data <- hdfs.put(data=read.table("<path-to-file>"),dfs.name="<path-to-dfs>",overwrite=T,rownames=T)

# split data into train and test sets: first 20% for testing / remaining 80% for training
# add x0=1 to all rows of X (convention)
# rescale the output to be 0 or 1 (needed for binary classification using the logistic function) instead of 1 or 2
# all this data preprocessing is performed in a distributed fashion (when working with enough data) by mappers running in the Hadoop cluster
dataPrep <- function(data,size,type) {
  map_in_format <- "vector"
  data_prep <- hadoop.run(
    data,
    mapper = function(k,v) {
      v <- as.numeric(v)
      if(type=="test") {
        if(v[length(v)]<=size) {
          v[length(v)-1] <- v[length(v)-1]-1
          v <- append(1,v)
          orch.keyval(k,v[1:length(v)-1])
        }
      }
      if(type=="train") {
        if(v[length(v)]>size) {
          v[length(v)-1] <- v[length(v)-1]-1
          v <- append(1,v)
          orch.keyval(k,v[1:length(v)-1])
        }
      }
    },
    export = orch.export(size,type),
    config = new("mapred.config",
                job.name = "data.preparation",
                map.split = 1,
                map.input = map_in_format,
                map.tasks = 1,
                verbose= T)
  )
  data_prep
}

# prepare test and train datasets
data_test <- dataPrep(data,200,"test")
data_train <- dataPrep(data,200,"train")

# initialize parameters vector theta
#theta <- as.matrix(rep(0,ncol(hdfs.get("/user/oracle/itau/X_train"))-2))
theta <- as.matrix(rep(0,25))

# hypothesis function
hypot <- function(z) {
  1/(1+exp(-z))
}

# gradient of cost function
gCost <- function(t,X,y) {
  1/nrow(X)*(t(X)%*%(hypot(X%*%t)-y))
}

train <- function(data_train,theta,hypot,gCost) {
  # define input and output formats for the mappers and reducers
  map_in_format <- "data.frame"
  map_out_schema <- data.frame(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
  reduce_in_format <- "data.frame"
  reduce_out_schema <- map_out_schema
  # runs one step of a batch gradient computation in a subset of the input data defined by the Hadoop cluster
  gradient <- hadoop.run(
    data_train,
    # compute partial results
    mapper = function(k,v) {
      X <- as.matrix(v[,1:25])
      y <- as.matrix(v[,26])
      p_gradient <- gCost(theta,X,y)
      orch.keyval(1,p_gradient)
    },
    # aggregate partial results
    reducer = function(k,v) {
      orch.keyval(k,sapply(v,sum))
    },
    # external variables needed for the MapReduce execution
    export = orch.export(theta,hypot,gCost,alpha,tol),
    # configuration parameters for the MapReduce job
    config = new("mapred.config",
                 job.name = "logistic.regression.train",
                 map.split = 0,
                 map.input = map_in_format,
                 map.output = map_out_schema,
                 reduce.input = reduce_in_format,
                 reduce.output= reduce_out_schema,
                 map.tasks = 1,
                 reduce.tasks = 1,
                 verbose=T)
  )
  # output of one MapReduce iteration
  t(as.matrix(hdfs.get(gradient)))
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
  cat("step: ",step,"\n")
  p_gradient <- train(data_train,theta,hypot,gCost)
  theta <- theta-alpha*p_gradient
  gradient <- train(data_train,theta,hypot,gCost)
  if(abs(norm(gradient,type="F")-norm(p_gradient,type="F"))<=tol) break
  step <- step+1
}

# hypothesis testing
# counts the predictions from the test set classified as 'good' and ' bad' credit and compares with the actual values
data_test <- as.matrix(hdfs.get(data_test))
X_test <- data_test[,1:25]
y_test <- data_test[,26]
y_pred <- hypot(X_test%*%theta)
result <- xor(as.vector(round(y_pred)),as.vector(y_test))
corrects = length(result[result==F])
wrongs = length(result[result==T])
cat("corrects: ",corrects,"\n")
cat("wrongs: ",wrongs,"\n")
cat("accuracy: ",corrects/length(y_pred),"\n")

# The execution of the code above should give a result like this:
# step:  1 
# step:  2 
# step:  3 
# ...
# step:  1330 
# corrects:  151 
# wrongs:  49 
# accuracy:  0.755
# The results above mean that the algorithm converged in 4 steps of Batch Gradient Descent and got 75.5% of the test dataset correct (the test dataset was not used in the training step).

# Final remarks:
# this is a very simple example that could be enhanced in several ways
# for exemple: by introducing a regularization factor in the cost function and by performing feature selection before training
