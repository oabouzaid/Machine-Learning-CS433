## Project 2 Machine Learning @ EPFL ##

**Kaggle Team Name:**
<br />
Logistic Depression

**Team Members:**
<br />
Oussama Abouzaid (oussama.abouzaid@epfl.ch)
<br />
Gaurav Pasari (gaurav.pasari@epfl.ch)
<br />
Karunya Tota (karunya.tota@epfl.ch)

**Instructions to Run:**
<br />
* To reproduce the results of our code and create a blended results submission, type **'python3 run.py'**. This generates our predictions by calculating the weights for all of our models (global_mean, global_median, user_mean, user_median, item_mean, item_median, als, sgd). Each model's predictions are pickeled in the 'predictions' directory. However, if a particular model does not exist, our program will rerun the function and generate the predictions for that model. 

* In order to run a particular model, type **'python3 run.py -m <model name(s) separated by commas>'**

* For help, type **'python3 run.py -h'**


**Estimated Runtime:**
<br />
~70 minutes

**Important Files**
* 'run.py' contains all of our code for parsing command line arguments and running the selected models. If no choice of models is given, the default option is to run all 8 methods and generate predictions. This file also performs the operations to read in training and testing data and generate splits on the training dataset.

* 'helpers.py' contains all of the helpers provided to us for this project. 

* 'data_helpers.py' contains our own helpers for creating a csv submission file using the testing dataset and generated predictions. It also contains the functionality to generate a split on our training set.

* The 'models' directory contains each of the following files:

	* 'means.py' contains the implementations and generates predictions for the global_mean, user_mean, and item_mean methods
	* 'medians.py' contains the implementations and generates predictions for the for global_median, user_median, and item_median methods
	* 'als.py' contains the implementation and generates predictions for the Alternating Least Squares algorithm
	* 'sgd.py' contains the implementation and generates predictions for the Stochastic Gradient Descent algorithm

* 'blender.py' takes all of the predictions from every model for the training data (using the 'train_predictions.csv') and calculates the best weights for each method using the least squares algorithm. Then, the script generates predictions for the testing dataset using these weights and writes output to a file titled 'output_blended.csv' in the 'predictions' directory.


**Libraries:**

We have used the following libraries to implement our methods:
* [Numpy](http://www.numpy.org/)
* [Scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)
* [Sklearn](http://scikit-learn.org/stable/)