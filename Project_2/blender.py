from helpers import *


JOINT_FILE = "./predictions/train_predictions.csv"


def blend(train_ratings, predictions):
	create_predictions(train_ratings, predictions)

	trainX, trainY, models = load_predictions(joint_file)
	
	blended_rmse, weights = least_squares(trainX, trainY)

	print("Blended RMSE: {s}".format(s=rmse))
	print("Weights: " + " ".join(str(x) for x in weights))

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
	line = []
	for item, user in idx:
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
	blended_rmse = sqrt(np.average((trainY - np.dot(trainX, w)) * (trainY - np.dot(trainX, w))))
	weights = np.ravel(w)

	return blended_rmse, weights