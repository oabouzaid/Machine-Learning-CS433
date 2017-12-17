from data_helpers import *
from helpers import *
from models.means import *


NUM_FOLDS = 5
PREDICTIONS_PATH = "./predictions/"
TRAIN_DATA = "./data/data_train.csv"
TEST_DATA = "./data/sample_submission.csv"

models = {
	# "global_mean": calculate_global_mean,
	# "user_mean": calculate_user_mean,
	"item_mean": calculate_item_mean
}


def get_predictions(model, train_folds, test_folds):
	return models[model](train_folds, test_folds)


def main():
	print("=======================================================")
	print("Starting Recommender")
	print("=======================================================")

	print("Loading train data from path = {}".format(TRAIN_DATA))
	train = load_data(path_dataset=TRAIN_DATA)
	print("Loading test data from path = {}".format(TEST_DATA))
	test = load_data(path_dataset=TEST_DATA)

	print("Splitting data into k = {} folds".format(NUM_FOLDS))
	train_folds, test_folds = split_data_into_folds(ratings=train, num_folds=NUM_FOLDS)

	for i in range(len(train_folds)):
		print(train_folds[i].shape)
		print(test_folds[i].shape)


	predictions = {}
	for model in models:
		print("=======================================================")
		print("Getting predictions for model = {}".format(model))
		predictions[model] = get_predictions(model, train_folds, test_folds)

	print("=======================================================")
	print("Finished Running Recommender")
	print("=======================================================")


if __name__ == '__main__':
	main()