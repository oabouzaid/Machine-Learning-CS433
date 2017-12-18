import getopt
import os.path
import pickle
import sys

from blender import *
from data_helpers import *
from helpers import *
from models.means import *


models = {
	"global_mean": calculate_global_mean,
	"user_mean": calculate_user_mean,
	"item_mean": calculate_item_mean,
}

PICKLE_PATH = "./pickles/"
PREDICTIONS_PATH = "./predictions/"
TRAIN_DATA = "./data/data_train.csv"
TEST_DATA = "./data/sample_submission.csv"


def get_predictions(model, train, test, test_ratings):
	return models[model](train, test, test_ratings)


def help_and_exit(err):
	print(err)
	print("[help]: python run.py -m <model name(s) separated by commas>")
	print("available models: {}".format(", ".join(models.keys())))
	print("default: all")
	sys.exit()


def get_selected_models(argv):
	try:
		opts, args = getopt.getopt(argv, "m:")
	except getopt.GetoptError as err:
		help_and_exit(str(err))

	for opt, arg in opts:
		if opt == "-m":
			selected_models = arg.split(",")
		else:
			help_and_exit("option {} not recognized".format(opt))
	
	for selected_model in selected_models:
		if selected_model not in models:
			help_and_exit("model {} not in available models".format(selected_model))

	return selected_models


def load_data_from_pkl():
	"""
	Pickle train and test data
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


def run_model(selected_models, train, test, test_ratings, use_pkls=True):
	"""
	Run the selected model
	"""
	output_file = ""
	prediction_pkl_path = PICKLE_PATH + "predictions.pkl"
	predictions = {}

	if os.path.exists(prediction_pkl_path):
		print("=======================================================")
		print("Loading predictions")
		predictions_pkl = open(prediction_pkl_path, "rb")
		predictions = pickle.load(predictions_pkl)
	else:
		for selected_model in selected_models:
			print("=======================================================")
			print("Running model = {}".format(model))
			predictions[selected_model] = get_predictions(
				selected_model,
				train,
				test,
				test_ratings)
			output_file = PREDICTIONS_PATH + "output_" + selected_model + ".csv"
			create_csv_submission(TEST_DATA, output_file, predictions[selected_model])

		predictions_pkl = open(prediction_pkl_path, "wb")
		pickle.dump(predictions, predictions_pkl)

	return predictions


def main(argv):
	print("=======================================================")
	print("Starting Recommender")
	print("=======================================================")

	selected_models = get_selected_models(argv)
	print("Selected models: {}".format(", ".join(selected_models)))

	start = time.time()

	train_ratings, test_ratings, train, test = load_data_from_pkl()

	predictions = run_model(selected_models, train, test, test_ratings, use_pkls=True)
	
	print("=======================================================")
	print("Blending models")
	print("=======================================================")

	coeffs = blend(train, predictions)
		
	print("=======================================================")
	print("Finished Running Recommender")
	print("=======================================================")

	end = time.time()
	print('Elapsed time: {s} minutes'.format(s=str((end - start)/60.0)))


if __name__ == '__main__':
	main(sys.argv[1:])