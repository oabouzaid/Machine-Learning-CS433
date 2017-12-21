import getopt
import os.path
import pickle
import sys

from blender import *
from data_helpers import *
from helpers import *
from models.als import *
from models.means import *
from models.medians import *


models = {
	"global_mean": calculate_global_mean,
	"global_median": calculate_global_median,
	"user_mean": calculate_user_mean,
	"user_median": calculate_user_median,
	"item_mean": calculate_item_mean,
	"item_median": calculate_item_median,
	"als": calculate_als
}

PICKLE_PATH = "./pickles/"
PREDICTIONS_PATH = "./predictions/"
TRAIN_DATA = "./data/data_train.csv"
TEST_DATA = "./data/sample_submission.csv"


def get_predictions(model, train, test, test_ratings):
	return models[model](train=train, test=test, test_ratings=test_ratings)


def run_models(selected_models, train, test, test_ratings):
	"""
	Run the selected model
	"""
	predictions = {}

	for model in selected_models:
		pred_pkl_path = PICKLE_PATH + "pred" + "_" + model + ".pkl"

		if os.path.exists(pred_pkl_path):
			print("Loading " + model + " model...")
			current_pred = pickle.load(open(pred_pkl_path, "rb"))
		else:
			current_pred = get_predictions(
				model=model,
				train=train,
				test=test,
				test_ratings=test_ratings)
			pickle.dump(current_pred, open(pred_pkl_path, "wb"))

		predictions[model] = current_pred

	return predictions


def get_data():
	"""
	Pickle train and test data
	"""
	train_pkl_path = PICKLE_PATH + "train.pkl"
	test_pkl_path = PICKLE_PATH + "test.pkl"
	train_split_pkl_path = PICKLE_PATH + "train_split.pkl"
	test_split_pkl_path = PICKLE_PATH + "test_split.pkl"

	if os.path.exists(train_pkl_path):
		print("Loading train data from path = {}".format(train_pkl_path))
		train_ratings = pickle.load(open(train_pkl_path, "rb"))
	else:
		print("Loading train data from path = {}".format(TRAIN_DATA))
		train_ratings = load_data(path_dataset=TRAIN_DATA)
		pickle.dump(train_ratings, open(train_pkl_path, "wb"))

	if os.path.exists(test_pkl_path):
		print("Loading test data from path = {}".format(test_pkl_path))
		test_ratings = pickle.load(open(test_pkl_path, "rb"))
	else:
		print("Loading test data from path = {}".format(TEST_DATA))
		test_ratings = load_data(path_dataset=TEST_DATA)
		pickle.dump(test_ratings, open(test_pkl_path, "wb"))

	if os.path.exists(train_split_pkl_path) and os.path.exists(test_split_pkl_path):
		print("Loading train_split_data from path = {}".format(train_split_pkl_path))
		print("Loading test_split_data from path = {}".format(test_split_pkl_path))
		train = pickle.load(open(train_split_pkl_path, "rb"))
		test = pickle.load(open(test_split_pkl_path, "rb"))
	else:
		print("Splitting training data into 0.90 train and 0.10 test")
		train, test = split_data(ratings=train_ratings)
		pickle.dump(train, open(train_split_pkl_path, "wb"))
		pickle.dump(test, open(test_split_pkl_path, "wb"))

	
	return train_ratings, test_ratings, train, test


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

	selected_models = list(models.keys())

	for opt, arg in opts:
		if opt == "-m":
			selected_models = arg.split(",")
	
	for selected_model in selected_models:
		if selected_model not in models:
			help_and_exit("model {} not in available models".format(selected_model))

	return selected_models


def main(argv):

	print("=======================================================")
	print("Starting Recommender")
	print("=======================================================")

	selected_models = get_selected_models(argv)
	print("Selected models: {}".format(", ".join(selected_models)))

	start = time.time()

	train_ratings, test_ratings, train, test = get_data()

	predictions = run_models(selected_models, train, test, test_ratings)
	
	print("=======================================================")
	print("Blending models")
	print("=======================================================")

	coeffs = blend(train_ratings, predictions)
	blended_path =  PREDICTIONS_PATH+"output_blended.csv"
	
	print("=======================================================")
	print("Generating predictions at " + blended_path)
	print("=======================================================")
	
	predictions_blended = blend_predictions(coeffs, predictions, test_ratings)
	create_csv_submission(TEST_DATA, blended_path, predictions_blended)
		
	print("=======================================================")
	print("Finished Running Recommender")
	print("=======================================================")

	end = time.time()
	print('Elapsed time: {s} minutes'.format(s=str((end - start)/60.0)))


if __name__ == '__main__':
	main(sys.argv[1:])