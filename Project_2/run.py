import getopt
import os.path
import pickle
import sys

from blender import *
from data_helpers import *
from helpers import *
from models.means import *


SELECTED_MODEL = "all_methods"
PREDICTIONS_PATH = "./predictions/"
PICKLE_PATH = "./pickles/"
TRAIN_DATA = "./data/data_train.csv"
TEST_DATA = "./data/sample_submission.csv"

models = {
	"global_mean": calculate_global_mean,
	"user_mean": calculate_user_mean,
	"item_mean": calculate_item_mean,
}


def get_predictions(model, train, test, test_ratings):
	return models[model](train, test, test_ratings)


def accept_parameters(argv):
	global SELECTED_MODEL, NUM_FOLDS

	try:
		opts, args = getopt.getopt(argv, "hm:",["model="])
	except getopt.GetoptError:
		print("[help]: run.py -m <model>")
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print ("[help]: python run.py -m <model>")
			print("available models: [global_mean, user_mean, item_mean]")
			print("default: all")
			sys.exit()
		elif opt in ("-m", "--model"):
			if arg not in models and arg != "all":
				print("available models: [global_mean, user_mean, item_mean]")
				print("default: all")
				sys.exit()
			SELECTED_MODEL = arg


def load_data_from_pkl():
	"""
	Pickles training and testing data
	"""
	train_pkl_path = PICKLE_PATH + "train.pkl"
	test_pkl_path = PICKLE_PATH + "test.pkl"
	train_split_pkl_path = PICKLE_PATH + "train_split.pkl"
	test_split_pkl_path = PICKLE_PATH + "test_split.pkl"

	if os.path.exists(train_pkl_path):
		training_data_pkl = open(train_pkl_path, "rb")
		testing_data_pkl = open(test_pkl_path, "rb")

		print("Loading train data from path = {}".format(TRAIN_DATA))
		train_ratings = pickle.load(training_data_pkl)
		print("Loading test data from path = {}".format(TEST_DATA))
		test_ratings = pickle.load(testing_data_pkl)

		train_pkl = open(train_split_pkl_path, "rb")
		test_pkl = open(test_split_pkl_path, "rb")

		train = pickle.load(train_pkl)
		test = pickle.load(test_pkl)

	else:
		print("Loading train data from path = {}".format(TRAIN_DATA))
		train_ratings = load_data(path_dataset=TRAIN_DATA)
		print("Loading test data from path = {}".format(TEST_DATA))
		test_ratings = load_data(path_dataset=TEST_DATA)

		train, test = split_data(ratings=train_ratings)

		train_data_pkl = open(train_pkl_path, "wb")
		test_data_pkl = open(test_pkl_path, "wb")

		pickle.dump(train_ratings, train_data_pkl)
		pickle.dump(test_ratings, test_data_pkl)

		train_pkl = open(train_split_pkl_path, "wb")
		test_pkl = open(test_split_pkl_path, "wb")

		pickle.dump(train, train_pkl)
		pickle.dump(test, test_pkl)
	
	return train_ratings, test_ratings, train, test

def run_model(SELECTED_MODEL, train, test, test_ratings, use_pkls=True):
	"""
	Runs the selected method
	"""
	output_file = ""
	prediction_pkl_path = PICKLE_PATH + "predictions.pkl"
	predictions = {}

	if os.path.exists(prediction_pkl_path):
		predictions_pkl = open(prediction_pkl_path, "rb")
		predictions = pickle.load(predictions_pkl)

	else:
		run_models = models.keys() if SELECTED_MODEL == "all_methods" else [SELECTED_MODEL]

		for model in run_models:
			print("=======================================================")
			print("Getting predictions for model = {}".format(model))
			output_file = PREDICTIONS_PATH+"output_"+model+".csv"
			predictions[model] = get_predictions(model, train, test, test_ratings)
			create_csv_submission(TEST_DATA, output_file, predictions[model])

		predictions_pkl = open(prediction_pkl_path, "wb")
		pickle.dump(predictions, predictions_pkl)

	return predictions

def main(argv):
	accept_parameters(argv)
	start = time.time()

	print("=======================================================")
	print("Starting Recommender")
	print("=======================================================")

	train_ratings, test_ratings, train, test = load_data_from_pkl()
	
	print("Running {}".format(SELECTED_MODEL))

	predictions = run_model(SELECTED_MODEL, train, test, test_ratings, use_pkls=True)
	
	print("=======================================================")
	print("Blending models")

	coeffs = blend(train, predictions)
		
	print("=======================================================")
	print("Finished Running Recommender")
	end = time.time()
	print('Elapsed time: {s} minutes'.format(s=str((end - start)/60.0)))
	print("=======================================================")

if __name__ == '__main__':
	main(sys.argv[1:])