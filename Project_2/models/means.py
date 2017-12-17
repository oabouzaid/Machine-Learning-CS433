from helpers import *


def baseline_global_mean(train, test):
    """baseline method: use the global mean."""
    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_mean_train = nonzero_train.mean()

    # find the non zero ratings in the test
    nonzero_test = test[test.nonzero()].todense()

    # predict the ratings as global mean
    mse = calculate_mse(nonzero_test, global_mean_train)
    rmse = np.sqrt(1.0 * mse / nonzero_test.shape[1])
    print("test RMSE of baseline using the global mean: {v}.".format(v=rmse))
    
    return global_mean_train, rmse[0]


def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    mse = 0
    num_items, num_users = train.shape

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = train[:, user_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]
        
        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_mean = nonzeros_train_ratings.mean()
        else:
            continue
        
        # find the non-zero ratings for each user in the test dataset
        test_ratings = test[:, user_index]
        nonzeros_test_ratings = test_ratings[test_ratings.nonzero()].todense()
        
        # calculate the test error 
        mse += calculate_mse(nonzeros_test_ratings, user_train_mean)
        rmse = np.sqrt(1.0 * mse / test.nnz)
    
    print("test RMSE of the baseline using the user mean: {v}.".format(v=rmse))
    
    return user_train_mean, rmse[0]


def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    mse = 0
    num_items, num_users = train.shape
    
    print(train.shape)
    print(test.shape)

    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[item_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean = nonzeros_train_ratings.mean()
        else:
            continue
        
        # find the non-zero ratings for each movie in the test dataset
        test_ratings = test[item_index, :]
        nonzeros_test_ratings = test_ratings[test_ratings.nonzero()].todense()
        
        # calculate the test error 
        mse += calculate_mse(nonzeros_test_ratings, item_train_mean)
    rmse = np.sqrt(1.0 * mse / test.nnz)
    print("test RMSE of the baseline using the item mean: {v}.".format(v=rmse))

    return item_train_mean, rmse[0]


def calculate_global_mean(training_folds, test_folds):
    pred_global_mean = []
    rmse = 0.0
    start = time.time()

    for i in range(len(training_folds)):
        pred, curr_rmse = baseline_global_mean(training_folds[i], test_folds[i])
        pred_global_mean.append(pred)
        rmse += curr_rmse

    average_rmse = rmse/len(training_folds)
    average_global_mean = np.mean(pred_global_mean)
    
    print('Average Folds RMSE: ' + str(average_rmse[0]))
    print('Average Folds Global Mean: ' + str(average_global_mean))

    end = time.time()
    print('Elapsed time: ' + str((end - start)/60.0))
    
    return average_rmse, average_global_mean


def calculate_user_mean(training_folds, test_folds):
    pred_user_mean = []
    rmse = 0.0
    start = time.time()

    for i in range(len(training_folds)):
        pred, curr_rmse = baseline_user_mean(training_folds[i], test_folds[i])
        pred_user_mean.append(pred)
        rmse += curr_rmse
        
    average_rmse = rmse/len(training_folds)
    average_user_mean = np.mean(pred_user_mean)

    print('Average Folds RMSE: ' + str(average_rmse[0]))
    print('Average Folds User Mean: ' + str(average_user_mean))

    end = time.time()
    print('Elapsed time: ' + str((end - start)/60.0))
    
    return average_user_mean


def calculate_item_mean(training_folds, test_folds):
    pred_item_mean = []
    rmse = 0.0
    start = time.time()

    print(len(training_folds))
    print(len(test_folds))

    for i in range(len(training_folds)):
        pred, curr_rmse = baseline_item_mean(training_folds[i], test_folds[i])
        pred_item_mean.append(pred)
        rmse += curr_rmse
        
    average_rmse = rmse/len(training_folds)
    average_item_mean = np.mean(pred_item_mean)

    print('Average Folds RMSE: ' + str(average_rmse[0]))
    print('Average Folds Item Mean: ' + str(average_item_mean))

    end = time.time()
    print('Elapsed time: ' + str((end - start)/60.0))
    
    return average_item_mean