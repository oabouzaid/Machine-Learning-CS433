import numpy as np
import matplotlib.pyplot as plt

from run import *

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
