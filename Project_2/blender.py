import numpy as np
from math import sqrt

def blend(train_data, predictions):
	print("=======================================================")
	print("Blending models")

	joint_file = "./data/train_predictions.csv"

	# create_predictions(train_data, predictions, joint_file)

	trainX, trainY, methods = load_predictions(joint_file)
	
	rmse, weights = least_squares(trainX, trainY)

	print("Blended RMSE: {s}".format(s=rmse))
	print("Weights: " + " ".join(str(x) for x in weights))

	coeffs = {}

	for i in range(len(methods)):
		coeffs[methods[i]] = weights[i]

	print("Coefficients: " + str(coeffs))

	return coeffs


def create_predictions(train_data, predictions, joint_file):
	"""
	Create a joint data file including the training data and their
	corresponing predictions
	"""

	# write header row
	file = open(joint_file, "w")
	file.write("Id,Prediction")

	# write each method name to header row
	models = list(predictions.keys())

	for i in range(len(models)):
		file.write("," + models[i])

	file.write("\n")

	rows, cols = train_data.nonzero()
	idx = list(zip(rows, cols))

	# write each true rating and predicted rating
	for movie, user in idx:
		line = "r" + str(movie) + "_c" + str(user) + ","
		rating_value = str(train_data[movie, user])
		file.write(line + rating_value)

		for model in models:
			file.write("," + str(predictions[model][movie, user]))

		file.write("\n")

	file.close()


def load_predictions(data_file):
	"""
	Load the training ratings, predictions, and method names
	"""
	input = np.genfromtxt(data_file, delimiter=",", skip_header=1)
	
	trainX = input[:, 1]
	trainX = np.expand_dims(trainX, axis=1)

	trainY = input[:, 2:]

	file = open(data_file,'r')
	methods = file.readline().strip()
	methods = methods.split(",")[2:]

	return trainX, trainY, methods

def least_squares(trainX, trainY):
	"""
	Calculate the least squares solution to generate weights
	for each method
	"""
	w = np.linalg.solve(trainX.T.dot(trainX), trainX.T.dot(trainY))    
	rmse = sqrt(np.average((trainY - np.dot(trainX, w)) * (trainY - np.dot(trainX, w))))

	weights = np.ravel(w)

	return rmse, weights