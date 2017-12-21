from helpers import *


def calculate_global_median(train, test, test_ratings):
    """ Use the global median as every prediction."""

    # find the nonzero train ratings
    train_nz = train[train.nonzero()]
    
    # calculate the global median of train_nz
    global_median = np.median(train_nz.todense(), axis=1)[0, 0]
    
    # find the training-test (0.1 split from train) ratings
    test_nz = test[test.nonzero()].todense()

    # calculate rmse between the training-test and the global median
    mse = calculate_mse(test_nz, global_median)
    rmse = np.sqrt(1.0 * mse/test_nz.shape[1])
    
    print("Test RMSE using the Global Median: {s}".format(s=rmse))
    
    # generate predictions matrix for the test ratings
    num_users = test_ratings.shape[0]
    num_items = test_ratings.shape[1]
    pred_global_median = sp.lil_matrix((num_users, num_items))

    for user in range(num_users):
        for item in range(num_items):
            pred_global_median[user, item] = global_median
        
    return pred_global_median


def calculate_user_median(train, test, test_ratings):
    """ Use the user medians as predictions. """
    mse = 0
    medians = []
    num_users, num_items = train.shape
    
    for user in range(num_users):
        # find all nonzero ratings for the present user
        user_train = train[user, :]
        nz_train = user_train[user_train.nonzero()]
        
        # calculate the user's average rating
        if nz_train.shape[1] != 0:
            user_median = np.median(nz_train.todense(), axis=1)[0, 0]
        else:
            user_median = 0.0
            
        medians.append(user_median)
        
        # find the test ratings from the train-test split
        user_test = test[user, :]
        nz_test = user_test[user_test.nonzero()].todense()
        
        # calculate the test error 
        mse += calculate_mse(nz_test, user_median)
        
    rmse = np.sqrt(1.0 * mse/test.nnz)
    print("Test RMSE using the User Median: {s}".format(s=rmse))
    
    # generate predictions matrix for the test ratings
    num_users = test_ratings.shape[0]
    num_items = test_ratings.shape[1]
    pred_user_median = sp.lil_matrix((num_users, num_items))
    
    for user in range(num_users):
        for item in range(num_items):
            pred_user_median[user, item] = medians[user]
        
    return pred_user_median


def calculate_item_median(train, test, test_ratings):
    """ Use the item medians as predictions. """
    mse = 0
    medians = []
    num_users, num_items = train.shape
    
    for item in range(num_items):
        # find all nonzero ratings for the item
        item_train = train[:, item]
        nz_train = item_train[item_train.nonzero()]
        
        # calculate the item's average rating
        if nz_train.shape[1] != 0:
            item_median = np.median(nz_train.todense(), axis=1)[0, 0]
        else:
            item_median = 0.0
            
        medians.append(item_median)
        
        # find the test ratings from the train-test split
        item_test = test[:, item]
        nz_test = item_test[item_test.nonzero()].todense()
        
        # calculate the test error 
        mse += calculate_mse(nz_test, item_median)
   
    rmse = np.sqrt(1.0 * mse/test.nnz)
    print("Test RMSE using the Item Median: {s}".format(s=rmse))
    
    # generate predictions matrix for the test ratings
    num_users = test_ratings.shape[0]
    num_items = test_ratings.shape[1]
    pred_item_median = sp.lil_matrix((num_users, num_items))
    
    for user in range(num_users):
        for item in range(num_items):
            pred_item_median[user, item] = medians[item]
        
    return pred_item_median