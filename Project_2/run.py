import sys
import getopt
import pickle

from data_helpers import *
from helpers import *
from models.means import *
from blender import *


SELECTED_MODEL = "all_methods"
PREDICTIONS_PATH = "./predictions/"
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
			sys.exit()
		elif opt in ("-m", "--model"):
			if arg not in models and arg != "all":
				print("available models: [global_mean, user_mean, item_mean]")
				print("default: all")
				sys.exit()
			SELECTED_MODEL = arg


def main(argv):
	accept_parameters(argv)

	print("=======================================================")
	print("Starting Recommender")
	print("=======================================================")

	print("Loading train data from path = {}".format(TRAIN_DATA))
	train_ratings = load_data(path_dataset=TRAIN_DATA)
	print("Loading test data from path = {}".format(TEST_DATA))
	test_ratings = load_data(path_dataset=TEST_DATA)

	train, test = split_data(ratings=train_ratings)

	train_data_pkl = open("./pickles/train_split.pkl", "wb")
	test_data_pkl = open("./pickles/test_split.pkl", "wb")

	pickle.dump(train_ratings, train_data_pkl)
	pickle.dump(test_ratings, test_data_pkl)

	train_pkl = open("./pickles/train_split.pkl", "wb")
	test_pkl = open("./pickles/test_split.pkl", "wb")

	pickle.dump(train, train_pkl)
	pickle.dump(test, test_pkl)
	
	# UNCOMMENT AFTER PICKLING ONCE
	# training_data_pkl = open("./pickles/train_data.pkl", "rb")
	# testing_data_pkl = open("./pickles/test_data.pkl", "rb")

	# print("Loading train data from path = {}".format(TRAIN_DATA))
	# train_ratings = pickle.load(training_data_pkl)
	# print("Loading test data from path = {}".format(TEST_DATA))
	# test_ratings = pickle.load(testing_data_pkl)

	# train_pkl = open("./pickles/train_split.pkl", "rb")
	# test_pkl = open("./pickles/test_split.pkl", "rb")

	# train = pickle.load(train_pkl)
	# test = pickle.load(test_pkl)

	print("Running {}".format(SELECTED_MODEL))

	predictions = {}
	output_file = ""

	run_models = models.keys() if SELECTED_MODEL == "all_methods" else [SELECTED_MODEL]

	for model in run_models:
		print("=======================================================")
		print("Getting predictions for model = {}".format(model))
		predictions[model] = get_predictions(model, train, test, test_ratings)
		output_file = PREDICTIONS_PATH+"output_"+model+".csv"
		create_csv_submission(TEST_DATA, output_file, predictions[model])

	blend(train, predictions)
		
	print("=======================================================")
	print("Finished Running Recommender")
	print("=======================================================")

if __name__ == '__main__':
	main(sys.argv[1:])