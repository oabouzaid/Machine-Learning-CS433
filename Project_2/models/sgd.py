from helpers import *
from models.als import *


def calculate_sgd(train, test, test_ratings):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    num_features = 15   # K in the lecture notes
    lambda_users = [.1]  #[29.5, 30.5, 31.0, 31.5, 32] #31.8553 was best for ALS
    lambda_items = [.007]  #[19.5, 20.0, 20.5, 21.0, 21.5] #20.05672522 was best for ALS
    num_epochs = 20     # number of full passes through the train set
    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    for lambda_user in lambda_users:
        for lambda_item in lambda_items:
            print("[PARAMS]: lambda_user = {} , lambda_item = {}, num_features = {}"\
                      .format(lambda_user, lambda_item, num_features))
            user_features, item_features = init_MF(train, num_features)
    
            # find the non-zero ratings indices 
            nz_row, nz_col = train.nonzero()
            nz_train = list(zip(nz_row, nz_col))
            nz_row, nz_col = test.nonzero()
            nz_test = list(zip(nz_row, nz_col))

            print("learn the matrix factorization using SGD...")
            for it in range(num_epochs):        
                # shuffle the training rating indices
                np.random.shuffle(nz_train)
        
                # decrease step size
                gamma /= 1.2
        
                for d, n in nz_train:
                    # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
                    item_info = item_features[:, n]
                    user_info = user_features[:, d]
                    err = train[d, n] - user_info.T.dot(item_info)
    
            # calculate the gradient and update
                    item_features[:, n] += gamma * (err * user_info - lambda_item * item_info)
                    user_features[:, d] += gamma * (err * item_info - lambda_user * user_info)

                rmse = compute_error(train, user_features, item_features, nz_train)
                print("iter: {}, RMSE on training set: {}.".format(it, rmse))
                rmse_b = compute_error(test, user_features, item_features, nz_test)
                print("RMSE on test data: {}.".format(rmse_b))  
        
                errors.append(rmse)

    # evaluate the test error
            # rmse = compute_error(test, user_features, item_features, nz_test)
            # print("RMSE on test data: {}.".format(rmse))  

    num_users = test_ratings.shape[0]
    num_items = test_ratings.shape[1]
    
    pred_sgd = sp.lil_matrix((num_users, num_items))
    
    for user in range(num_users):
        for item in range(num_items):
            user_info = user_features[:, user]
            item_info = item_features[:, item]
            pred_sgd[user, item] = user_info.T.dot(item_info)
    
    return pred_sgd