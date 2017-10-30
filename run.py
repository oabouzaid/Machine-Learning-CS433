from proj1_helpers import *
from implementations import *


def error(y, tx, w):
    '''
    Calculates the error in the current prediction.
    '''
    return y - np.dot(tx, w)


def compute_loss(y, tx, w):
    '''
    Calculates the loss using MSE.
    '''
    N = y.shape[0]
    e = error(y, tx, w)
    factor = 1/(2*N)
    loss = (np.dot(np.transpose(e), e)) * factor
    return loss


def compute_gradient(y, tx, w):
    '''
    Computes the gradient of the MSE loss function.
    '''
    N = y.shape[0]
    e = error(y, tx, w)
    factor = -1/N
    grad = (np.dot(np.transpose(tx), e)) * factor
    loss = compute_loss(y, tx, w)
    return grad, loss


def compute_stoch_gradient(y, tx, w):
    '''
    Computes the stochastic gradient from a few examples of n and their corresponding y_n labels.
    '''
    N = y.shape[0]
    e = error(y, tx, w)
    factor = -1/N
    grad = (np.dot(np.transpose(tx), e)) * factor
    loss = compute_loss(y, tx, w)
    return grad, loss


def sigma(x):
    '''
    Calculates sigma using the given formula.
    '''
    return np.exp(x)/(1+np.exp(x))


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    '''
    Generates a minibatch iterator for a dataset.
    Takes as input two iterables - the output desired values 'y' and the input data 'tx'.
    Outputs an iterator which gives mini-batches of batch_size matching elements from y and tx.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use:
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        do something
    '''
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            
def compare_prediction(w_train, x, y):
    '''
    Calculates the accuracy by comparing the predictions with given test data.
    '''
    pred = predict_labels(w_train, x)
    N = len(pred)
    count = 0.0
    for i in range(len(pred)):
        if pred[i] == y[i]:
            count += 1
    return count/N


def split_data(tx, ty, ratio, seed=1):
    '''
    Split the training data by ratio.
    '''
    np.random.seed(seed)
    split_idxs = [i for i in range(len(tx))]
    
    # Shuffle the indicies randomly
    np.random.shuffle(split_idxs)
    tx_shuffled = tx[split_idxs]
    ty_shuffled = ty[split_idxs]
    
    # Split by ratio
    split_pos = int(len(tx) * ratio)
    x_train = tx_shuffled[:split_pos]
    x_test = tx_shuffled[split_pos:]
    y_train = ty_shuffled[:split_pos]
    y_test = ty_shuffled[split_pos:]
    
    return x_train, y_train, x_test, y_test 


def build_poly(x, degree):
    '''
    Builds a polynomial of the given degree and appends it to the given matrix.
    '''
    x_ret = x
    for i in range(2, degree+1):
        x_ret = np.c_[x_ret, np.power(x, i)]
    return x_ret


def standardize(x):
    '''
    Standardizes the matrix by subtracting mean of each column and then dividing by standard deviation.
    '''
    res = x.copy()
    for i in range(0, res.shape[1]):
        if i == 22:
            continue
            
        # Calculate mean and standard deviation without including NaN values
        mean = np.nanmean(res[:,i])
        std = np.nanstd(res[:,i])
        
        # Change mean and standard deviation if column has all NaN values
        if np.isnan(mean):
            mean = 0
        if np.isnan(std) or std == 0:
            std = 1
        
        # Replaces NaN values with mean and divides by standard deviation
        for j in range(len(res[:,i])):
            if np.isnan(res[j][i]):
                res[j][i] = mean
            else:
                res[j][i] -= mean
            res[j][i] /= std
    return res


def replace_999_with_nan(x):
    '''
    Replaces -999 (undefined values) with NaN.
    '''
    res = x.copy()
    res[res == -999.0] = np.nan
    return res


def one_hot_encoding(x):
    '''
    Converts categorical data for PRI_jet_num (column at index 22) to use one hot encoding.
    '''
    res = x.copy()
    col = res[:,22]
    b = np.zeros((res.shape[0], 4))
    for i in range(len(b)):
        a = int(col[i])
        b[i][a] = 1
    res = np.delete(res, 22, 1)
    res = np.hstack((res, b))
    return res


def get_buckets(x):
    '''
    Splits the dataset into 8 buckets.
    Based on 4 values (0, 1, 2, 3) of PRI_jet_num and 2 values (defined or -999) of DER_mass_MMC.
    '''
    result = []
    for i in range(0, 4):
        # Get all rows where PRI_jet_num equals i
        x_jet = x[x[:,22] == i]
        
        # Get all rows where DER_mass_MMC defined and undefined
        xi_defined = x_jet[x_jet[:,0] != -999.0]
        xi_undefined = x_jet[x_jet[:,0] == -999.0]
        
        result.append(xi_defined)
        result.append(xi_undefined)
    return result


def get_id_buckets(x):
    '''
    Splits the set of ids into 8 buckets as above.
    Used for sorting the predictions based on event id.
    '''
    result = []
    for i in range(0, 4):
        x_jet = x[x[:,1] == i]
        xi_defined = x_jet[x_jet[:,-1] != -999.0]
        xi_undefined = x_jet[x_jet[:,-1] == -999.0]
        result.append(xi_defined)
        result.append(xi_undefined)
    return result
