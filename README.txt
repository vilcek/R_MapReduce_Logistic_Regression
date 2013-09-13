Example code to implement Logistic Regression with R running on Hadoop

These R programs are examples of implementations of Logistic Regression where the optimization (error minimization) of the Cost Function is performed through Batch Gradient Descent. The first example (logistic_regression.R) implements that through common single-threaded R code. The second example (mr_logistic_regression.R) shows how to rewrite the first example to implement the same algorithm expressed in MapReduce code, using the R MapReduce framework provided by ORCH (Oracle R Connector for Hadoop).

Therefore, to run the first example you need only to provide the dataset and R instaled on your machine. For running the second example, you need to have access to a Hadoop cluster with R and ORCH installed on its nodes. The easiest way to accomplish this for tests purposes is to assemble a cluster in pseudo-distributed mode in a desktop.

Both examples use the 'German Credit Data' dataset, which can be obtained from the UCI Machine Learning repository in http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
To run the examples you will need to download only the 'german.data-numeric' file.

Using that dataset, the examples show how to use Logistic Regression to predict good and bad credit.

The execution of the first example (logistic_regression.R) should give a result like this:
steps: 4
corrects: 143
wrongs: 57
accuracy: 0.715

The results above mean that the algorithm converged in 4 steps of Batch Gradient Descent and got 71.5% of the test dataset correct.

The execution of the second example (mr_logistic_regression.R) should give a result like this:
step: 1
step: 2
step: 3
step: 4
corrects: 143
wrongs: 57
accuracy: 0.715

It is worth noting that this is a fairly simple implementation that could probably achieve better results by applying a regularization term to the cost function as well as a feature selection pre-processing.

For more information on how to assemble a Hadoop cluster in pseudo-distributed mode:
http://hadoop.apache.org/docs/stable/single_node_setup.html

For more information about R:
http://cran.us.r-project.org/

For more information about Oracle R Connector for Hadoop:
http://docs.oracle.com/cd/E41604_01/doc.22/e41238/orch.htm#CFHFIGAI
http://docs.oracle.com/cd/E41604_01/doc.22/e41238/start.htm#CHDJIBGI

