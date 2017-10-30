import numpy as np
import matplotlib.pyplot as plt


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''
    Linear regression using gradient descent.
    '''
    w = initial_w
    for n_iter in range(max_iters):
        grad, loss = compute_gradient(y, tx ,w)
        w = w - (gamma * grad)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    '''
    Linear regression using stochastic gradient descent.
    '''
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - (gamma * grad)
            loss = compute_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    '''
    Least squares regression using normal equations.
    '''
    gram = np.dot(np.transpose(tx),tx)
    gram = np.linalg.inv(gram)
    
    w = np.dot(gram,np.transpose(tx))
    w = np.dot(w, y)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    '''
    Ridge regression using normal equations.
    '''
    N = tx.shape[1]
    a = np.dot(np.transpose(tx), tx) + (lambda_ * np.identity(N))
    b = np.dot(np.transpose(tx), y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''
    Logistic regression using gradient descent.
    '''
    w = initial_w
    for n_iter in range(max_iters):
        yx = np.dot(y, tx)
        yxw = np.dot(yx, w)
        log = np.log(1 + np.exp(np.dot(tx, w)))
        loss = (log - yxw).sum()
        
        # Update rule
        sig = sigma(np.dot(tx, w))
        sig = sig - y
        grad = np.dot(np.transpose(tx), sig)
        w = w - (gamma * grad)
    return w, loss


def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    '''
    Regularized logistic regression using gradient descent.
    '''
    w = initial_w
    for n_iter in range(max_iters):
        yx = np.dot(y, tx)
        yxw = np.dot(yx, w)
        log = np.log(1 + np.exp(np.dot(tx, w)))
        
        # Add the 'penalty' term
        loss = (log - yxw).sum() - (lambda_/2)* np.square((np.linalg.norm(w)))
        
        # Update rule
        sig = sigma(np.dot(tx, w))
        sig = sig - y
        grad = np.dot(np.transpose(tx), sig) + (2 * lambda_*w)
        w = w - (gamma * grad)
    return w, loss


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
