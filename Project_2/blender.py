import os.path
from math import sqrt
import numpy as np

from helpers import *

JOINT_FILE = "./predictions/train_predictions.csv"


def blend(train_ratings, predictions):
	create_predictions(train_ratings, predictions)

	trainX, trainY, models = load_predictions(JOINT_FILE)
	
	blended_rmse, weights = least_squares(trainX, trainY)

	print("Blended RMSE: {s}".format(s=blended_rmse))

	coeffs = {}
	for i in range(len(models)):
		coeffs[models[i]] = weights[i]

	print("Coefficients: " + str(coeffs))

	return coeffs


def create_predictions(train_ratings, predictions):
	"""
	Create a joint data file including the training data and their
	corresponing predictions
	"""

	global JOINT_FILE

	# if predictions are already created, skip
	if os.path.exists(JOINT_FILE):
		return

	file = open(JOINT_FILE, "w")

	# write header row
	header = "Id,Prediction," + ",".join(predictions.keys())
	file.write(header)
	file.write("\n")

	rows, cols = train_ratings.nonzero()
	idx = list(zip(rows, cols))

	# write each true rating and predicted rating
	models = list(predictions.keys())
	
	for item, user in idx:
		line = []

		user_item_id = "r" + str(item + 1) + "_" + "c" + str(user + 1)
		rating = str(train_ratings[item, user])

		line.append(user_item_id)
		line.append(rating)

		for model in models:
			model_predictions = str(predictions[model][item, user])
			line.append(model_predictions)

		line = ",".join(line)

		file.write(line)
		file.write("\n")

	file.close()


def load_predictions(data_file):
	"""
	Load the training ratings, predictions, and method names
	"""
	input = np.genfromtxt(data_file, delimiter=",", skip_header=1)
	
	trainY = input[:, 1]
	trainY = np.expand_dims(trainY, axis=1)

	trainX = input[:, 2:]

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
	blended_rmse = sqrt(np.average((trainY - np.dot(trainX, w)) * (trainY - np.dot(trainX, w))))
	weights = np.ravel(w)

	return blended_rmse, weights


def blend_predictions(coeffs, predictions, test_ratings):
	"""
	Create the blended predictions for the test data
	"""
	
	# # generate predictions matrix for the test ratings
	num_items = test_ratings.shape[0]
	num_users = test_ratings.shape[1]
	models = list(predictions.keys())
	pred_blended = sp.lil_matrix((num_items, num_users))

	for item in range(num_items):
		for user in range(num_users):
			score = 0.0
			for i in range(len(models)):
				score += coeffs[models[i]] * predictions[models[i]][item, user]
			score = max(score, 1.0)
			score = min(score, 5.0)
			pred_blended[item, user] = score

	return pred_blended
