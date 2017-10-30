from proj1_helpers import *
from implementations import *


# Constants for importing training and testing data
TRAINING_DATA = '/Users/Gaurav/Desktop/train.csv'
TEST_DATA = '/Users/Gaurav/Desktop/test.csv'

# Constants used to clean data
SPLIT_PERCENT = 0.80
DEGREE = 13
LAMBDA_ = 0.9

            
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


print("Loading data from train = " + TRAINING_DATA + " test = " + TEST_DATA)

# Load csv training and testing data
ty, tx, ids_train = load_csv_data(TRAINING_DATA, sub_sample = False)
fy, fx, ids_test = load_csv_data(TEST_DATA, sub_sample = False)

print("Completed loading data.")

# Backup of the imported training and testing data
orig_tx = tx.copy()
orig_ty = ty.copy()
orig_fx = fx.copy()


print("Partition into buckets, clean data, standardize, build polynomial, add intercept.")

# Split data into 80-20 and use 80 for training and 20 to check model accuracy
x_train, y_train, x_test, y_test = split_data(orig_tx.copy(), orig_ty.copy(), SPLIT_PERCENT, seed=1)

# Split final test data to make predictions on into buckets
fx_train = orig_fx.copy()
fx_buckets = get_buckets(fx_train)

# Append y values as column to later divide y into buckets corresponding with x values
x_train = np.column_stack((x_train, y_train))
x_test = np.column_stack((x_test, y_test))

# Split training x into buckets
buckets = get_buckets(x_train)

# Split testing y into buckets corresponding to x values
y_buckets = []
for i in range(len(buckets)):
    y_buckets.append(buckets[i][:,-1])
    buckets[i] = np.delete(buckets[i], -1, 1)

# Split testing x into buckets
test_buckets = get_buckets(x_test)

# Split testing y into buckets corresponding to x values
test_y_buckets = []
for i in range(len(test_buckets)):
    test_y_buckets.append(test_buckets[i][:,-1])
    test_buckets[i] = np.delete(test_buckets[i], -1, 1)

# Replace -999 with NaN and standardize columns for each bucket
for b in range(len(buckets)):
    buckets[b] = replace_999_with_nan(buckets[b])
    buckets[b] = standardize(buckets[b])
    # buckets[b] = one_hot_encoding(buckets[b])
    test_buckets[b] = replace_999_with_nan(test_buckets[b])
    test_buckets[b] = standardize(test_buckets[b])
    # test_buckets[b] = one_hot_encoding(test_buckets[b])
    fx_buckets[b] = replace_999_with_nan(fx_buckets[b])
    fx_buckets[b] = standardize(fx_buckets[b])
    # fx_buckets[b] = one_hot_encoding(fx_buckets[b])
    
# Build polynomial of given degree for each bucket
for b in range(len(buckets)):
    buckets[b] = build_poly(buckets[b], DEGREE)
    test_buckets[b] = build_poly(test_buckets[b], DEGREE)
    fx_buckets[b] = build_poly(fx_buckets[b], DEGREE)

# Add column of ones for intercept for each bucket
for b in range(len(buckets)):
    buckets[b] = np.column_stack((np.ones((buckets[b].shape[0], 1)), buckets[b]))
    test_buckets[b] = np.column_stack((np.ones((test_buckets[b].shape[0], 1)), test_buckets[b]))
    fx_buckets[b] = np.column_stack((np.ones((fx_buckets[b].shape[0], 1)), fx_buckets[b]))

print("Applying ridge regression on each bucket.")


# Calculate weights for each bucket separately
weights = []
for i in range(len(buckets)):
    w_rr, loss_rr = ridge_regression(y_buckets[i], buckets[i], LAMBDA_)
    weights.append(w_rr)

# Compare predictions for each bucket using its corresponding weights found earlier
correct_predictions = 0
len_data = 0
for i in range(len(buckets)):
    rr_accuracy = compare_prediction(weights[i], test_buckets[i], test_y_buckets[i])
    correct_predictions += (rr_accuracy * len(test_buckets[i]))
    len_data += len(test_buckets[i])

total_accuracy = correct_predictions / len_data

print("Total Accuracy = " + str(total_accuracy) + " Degree = " + str(DEGREE) + " Lambda = " + str(LAMBDA_))


# Create new array with Id, PRI_jet_num, and DER_mass_MMC for reordering predictions
ids_array = ids_test
pri_jet_num_col = orig_fx[:,22]
der_mass_mmc_col = orig_fx[:,0]
ids_array = np.column_stack((ids_array, pri_jet_num_col))
ids_array = np.column_stack((ids_array, der_mass_mmc_col))

# Divide Id into 8 buckets similar to input data
id_buckets = get_id_buckets(ids_array)

# Make predictions for each bucket using weights calculated by training on each bucket
final_y = predict_labels(weights[0], fx_buckets[0])
final_y = np.column_stack((final_y, id_buckets[0]))
for i in range(1, len(weights)):
    y_pred = predict_labels(weights[i], fx_buckets[i])
    y_pred = np.column_stack((y_pred, id_buckets[i]))
    final_y = np.concatenate((final_y, y_pred))
    
# Sort predictions based on Id
final_y = final_y[final_y[:,1].argsort()]

# Select only prediction values
final_y = final_y[:,0]

# Create output file containing predictions
create_csv_submission(ids_test, final_y, "output.csv")
print("Created output.csv with shape = " + str(final_y.shape))
