from helpers import *


def init_MF(train, num_features):
	""" 
	TODO: REVIEW AND CHANGE
	Source: Lab 10 Solutions
	Init the parameter for matrix factorization 
	"""

	num_users, num_items = train.shape
	user_features = np.random.rand(num_features, num_users)
	item_features = np.random.rand(num_features, num_items)

	# get the sum of every item's ratings
	item_sum = train.sum(axis=0)
	item_nnz = train.getnnz(axis=0)
	
	# set the first item features to the sum of the ratings divided by the number of nonzero items
	for item_index in range(num_items):
		item_features[0, item_index] = item_sum[0, item_index]/item_nnz[item_index]
		
	return user_features, item_features


def update_user_feature(train, item_features, num_features, lambda_user, nz_user_itemindices):
	"""
	Update user feature matrix
	"""
	num_users = train.shape[0]
	user_features = np.zeros((num_features, num_users))

	for user, items in nz_user_itemindices:
		W = item_features[:, items]
		
		A = W @ W.T + lambda_user * sp.eye(num_features)
		b = W @ train[user, items].T
		
		x = np.linalg.solve(A, b)
		
		user_features[:, user] = np.copy(x.T)
		
	return user_features


def update_item_feature(train, user_features, num_features, lambda_item, nz_item_userindices):
	"""
	Update item feature matrix
	"""
	num_items = train.shape[1]
	item_features = np.zeros((num_features, num_items))

	for item, users in nz_item_userindices:
		W = user_features[:, users]
		
		A = W @ W.T + lambda_item * sp.eye(num_features)
		b = W @ train[users, item]
		
		x = np.linalg.solve(A, b)
		
		item_features[:, item] = np.copy(x.T)
		
	return item_features


def compute_error(data, user_features, item_features, nz):
	""" 
	Source: Lab 10 Solutions
	Compute the MSE between predictions and nonzero train elements
	"""
	mse = 0
	pred = np.dot(user_features.T, item_features)
	
	for row, col in nz:
		mse += np.square((data[row, col] - pred[row, col]))
		
	return np.sqrt(mse/len(nz))


def calculate_als(train, test, test_ratings, seed=988, num_features=8, m_iter=30, lambda_user=31.8553, lambda_item=20.05672522, change=1, stop_criterion=1e-4):
	"""
	Use Alternating Least Squares (ALS) algorithm to generate predictions
	"""
	error_list = [0]
	itr = 0
	
	# set seed
	np.random.seed(seed)

	# initialize the matrix factorization
	user_features, item_features = init_MF(train, num_features)
	
	# group the indices by row or column index
	nz_train, nz_user_itemindices, nz_item_userindices = build_index_groups(train)
	
	print("Starting ALS")
	while change > stop_criterion and itr < m_iter:
		# update user features & item features
		user_features = update_user_feature(train, item_features, num_features, lambda_user, nz_user_itemindices)
		item_features = update_item_feature(train, user_features, num_features, lambda_item, nz_item_userindices)

		rmse = compute_error(train, user_features, item_features, nz_train)
		print("RMSE on training set: {}.".format(rmse))
	
		change = np.fabs(rmse - error_list[-1])
		error_list.append(rmse)
		itr += 1
	
	# evaluate the test error
	nnz_row, nnz_col = test.nonzero()
	nnz_test = list(zip(nnz_row, nnz_col))
	rmse = compute_error(test, user_features, item_features, nnz_test)
	print("Test RMSE after running ALS: {s}".format(s=rmse))

	num_users = test_ratings.shape[0]
	num_items = test_ratings.shape[1]
	pred_als = sp.lil_matrix((num_users, num_items))
	
	for user in range(num_users):
		for item in range(num_items):
			item_info = item_features[:, item]
			user_info = user_features[:, user]
			pred_als[user, item] = user_info.T.dot(item_info)
		
	return pred_als