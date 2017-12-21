from helpers import *


def init_MF(train, num_features):
	""" 
	Source: Lab 10 Solutions
	Init the parameter for matrix factorization 
	"""

	num_users, num_items = train.shape
	user_features = np.random.rand(num_features, num_users)
	item_features = np.random.rand(num_features, num_items)

	# Sum up each ratings for each movie 
	item_sum = train.sum(axis=0)
	item_nnz = train.getnnz(axis=0)
	
	# set the first item features to the sum of the ratings divided by the number of nonzero items
	for item_index in range(num_items):
		item_features[0, item_index] = item_sum[0, item_index]/item_nnz[item_index]
		
	return user_features, item_features


def update_user_feature(train, item_features, num_features, lambda_user, nz_user_itemindices):
	"""
	Helper function that updates the user-feature matrix Z
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
	Update item-feature matrix W
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


def calculate_als(train, test, test_ratings, seed=988, num_features=8, m_iter=10, lambda_user=1., lambda_item=0.007, change=1, stop_criterion=1e-4):
	"""
	Use Alternating Least Squares (ALS) algorithm to generate predictions
	"""
	error_list = [0]
	itr = 0
	np.random.seed(seed)

	# Initialize W and Z with random small numbers
	user_features, item_features = init_MF(train, num_features)
	
	# Group the indices by row or column index
	nz_train, nz_user_itemindices, nz_item_userindices = build_index_groups(train)
	
	while change > stop_criterion and itr < m_iter:
		
		# Update W and Z
		user_features = update_user_feature(train, item_features, num_features, lambda_user, nz_user_itemindices)
		item_features = update_item_feature(train, user_features, num_features, lambda_item, nz_item_userindices)

		rmse = compute_error(train, user_features, item_features, nz_train)
		print("RMSE on training set: {}.".format(rmse))
	
		change = np.fabs(rmse - error_list[-1])
		error_list.append(rmse)
		itr = itr + 1
	
	# Calculate the RMSE
	nnz_row, nnz_col = test.nonzero()
	nnz_test = list(zip(nnz_row, nnz_col))
	rmse = compute_error(test, user_features, item_features, nnz_test)
	print("Test RMSE after running ALS: {s}".format(s=rmse))

	num_users = test_ratings.shape[0]
	num_items = test_ratings.shape[1]
	pred_als = sp.lil_matrix((num_users, num_items))
	
	# Multiply the 2 matrices to get X 
	for user in range(num_users):
		for item in range(num_items):
			item_info = item_features[:, item]
			user_info = user_features[:, user]
			pred_als[user, item] = user_info.T.dot(item_info)
		
	return pred_als