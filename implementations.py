import numpy as np
import matplotlib.pyplot as plt


def error(y, tx, w):
    return y - np.dot(tx, w)

def compute_loss(y, tx, w):
    """Calculates the loss using MSE."""
    N = y.shape[0]
    e = error(y, tx, w)  
    loss = (np.dot(np.transpose(e), e))* (1/(2*N))   
    return loss

def compute_gradient(y, tx, w):
    """Computes the gradient of the MSE loss function"""
    N = y.shape[0]
    e = error(y, tx, w)
    grad = (np.dot(np.transpose(tx), e)) * (-1/N)
    loss = compute_loss(y, tx, w)
    return grad, loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    N = y.shape[0]
    e = error(y, tx, w)
    grad = (np.dot(np.transpose(tx), e)) * (-1/N)
    loss = compute_loss(y, tx, w)
    return grad, loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
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


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w.copy() # ........ !!!
    for n_iter in range(max_iters):
        grad, loss = compute_gradient(y, tx ,w)
       #Update rule
        w = w - gamma * grad  
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss  


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w.copy()
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """calculate the least squares solution."""     
    gram = np.dot(np.transpose(tx),tx)
    gram = np.linalg.inv(gram)
    
    w = np.dot(gram,np.transpose(tx))
    w = np.dot(w, y) 
    v = np.dot(tx, w)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    N = y.shape[0]
    gram = np.dot(np.transpose(tx),tx)
    i = (np.identity(N))*(2*lambda_*N)
    gram = gram + i
    gram = np.linalg.inv(gram)
    w = np.dot(gram,np.transpose(tx))
    w = np.dot(w, y) 
    loss = compute_loss(y, tx, w)
    return w, loss


def sigma(x):
    return np.exp(x)/(1+np.exp(x))


#Logistic regression using gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    #if y.min == -1:
    #    y = (y>0).astype(np.float64)
    w = initial_w
    for n_iter in range(max_iters):
        yx = np.dot(y, np.transpose(tx))
        yxw = np.dot(yx, w)
        log = np.log(1 + np.exp(np.dot(np.transpose(tx),w)))
        loss = (log - yxw).sum()
        #Update rule
        sig = sigma(np.dot(xt, w))
        sig = sig - y
        grad = np.dot(np.transpose(xt), sig)
        w = w - gamma * grad 
        
        ## at the last iteration, should the gradient be updated before or after the the update rule ???? 
    return w, loss 


#Regularized logistic regression using gradient descent
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    #'a0Case y.min == -1
    w = initial_w
    for n_iter in range(max_iters):
        yx = np.dot(y, np.transpose(tx))
        yxw = np.dot(yx, w)
        log = np.log(1 + np.exp(np.dot(np.transpose(tx),w)))
        loss = (log - yxw).sum() - (lambda_/2)* np.square((np.linalg.norm(w)))   ## Add the 'penalty' term
        #Update rule
        sig = sigma(np.dot(xt, w))
        sig = sig - y
        grad = np.dot(np.transpose(xt), sig) + 2 * lambda_*w
        w = w - gamma * grad 
        
    return w, loss 