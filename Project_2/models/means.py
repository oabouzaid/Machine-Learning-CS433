from helpers import *


def calculate_global_mean(train, test, test_ratings):
    """ Use the global mean as every prediction."""

    # find the nonzero train ratings
    train_nz = train[train.nonzero()]
    
    # calculate the global mean of train_nz
    global_mean = train_nz.mean()
    
    # find the training-test (0.1 split from train) ratings
    test_nz = test[test.nonzero()].todense()

    # calculate rmse between the training-test and the global mean
    mse = calculate_mse(test_nz, global_mean)
    rmse = np.sqrt(1.0 * mse/test_nz.shape[1])
    
    print("Test RMSE using the Global Mean: {s}".format(s=rmse))
    
    # generate predictions matrix for the test ratings
    num_items = test_ratings.shape[0]
    num_users = test_ratings.shape[1]
    pred_global_mean = sp.lil_matrix((num_items, num_users))

    for item in range(num_items):
        for user in range(num_users):
            pred_global_mean[item, user] = global_mean
        
    return pred_global_mean


def calculate_user_mean(train, test, test_ratings):
    """ Use the user means as predictions. """
    mse = 0
    means = []
    num_items, num_users = train.shape
    
    for user in range(num_users):
        # find all nonzero ratings for the present user
        user_train = train[:, user]
        nz_train = user_train[user_train.nonzero()]
        
        # calculate the user's average rating
        if nz_train.shape[0] != 0:
            user_mean = nz_train.mean()
        else:
            user_mean = 0.0
            
        means.append(user_mean)
        
        # find the test ratings from the train-test split
        user_test = test[:, user]
        nz_test = user_test[user_test.nonzero()].todense()
        
        # calculate the test error 
        mse += calculate_mse(nz_test, user_mean)
        
    rmse = np.sqrt(1.0 * mse/test.nnz)
    print("Test RMSE using the User Mean: {s}".format(s=rmse))
    
    # generate predictions matrix for the test ratings
    num_items = test_ratings.shape[0]
    num_users = test_ratings.shape[1]
    pred_user_mean = sp.lil_matrix((num_items, num_users))
    
    for item in range(num_items):
        for user in range(num_users):
            pred_user_mean[item, user] = means[user]
        
    return pred_user_mean


def calculate_item_mean(train, test, test_ratings):
    """ Use the item means as predictions. """
    mse = 0
    means = []
    num_items, num_users = train.shape
    
    for item in range(num_items):
        # find all nonzero ratings for the item
        item_train = train[item, :]
        nz_train = item_train[item_train.nonzero()]
        
        # calculate the item's average rating
        if nz_train.shape[0] != 0:
            item_mean = nz_train.mean()
        else:
            item_mean = 0.0
            
        means.append(item_mean)
        
        # find the test ratings from the train-test split
        item_test = test[item, :]
        nz_test = item_test[item_test.nonzero()].todense()
        
        # calculate the test error 
        mse += calculate_mse(nz_test, item_mean)
   
    rmse = np.sqrt(1.0 * mse/test.nnz)
    print("Test RMSE using the Item Mean: {s}".format(s=rmse))
    
    # generate predictions matrix for the test ratings
    num_items = test_ratings.shape[0]
    num_users = test_ratings.shape[1]
    pred_item_mean = sp.lil_matrix((num_items, num_users))
    
    for item in range(num_items):
        for user in range(num_users):
            pred_item_mean[item, user] = means[item]
        
    return pred_item_mean