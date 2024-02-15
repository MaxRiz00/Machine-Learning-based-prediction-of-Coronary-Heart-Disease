"""Implementations of different ML methods."""

import numpy as np


def compute_loss_mse(y, tx, w):
    """Computes the MSE loss function."""

    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e ** 2)


def compute_gradient_mse(y, tx, w):
    """Computes the gradient of the MSE loss function."""

    e = y - tx.dot(w)
    return - 1 / len(y) * tx.T.dot(e)


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


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Computes the gradient descent algorithm for the MSE loss function."""

    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and update w
        w = w - gamma * compute_gradient_mse(y, tx, w)

    loss = compute_loss_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Computes the stochastic gradient descent algorithm for the MSE loss function."""

    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            w = w - gamma * compute_gradient_mse(minibatch_y, minibatch_tx, w)

    loss = compute_loss_mse(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns loss, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """

    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))

    loss = compute_loss_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Calculate the ridge regression solution.
       returns loss, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    lambda_prime = 2 * len(y) * lambda_

    w = np.linalg.solve(tx.T.dot(tx) + lambda_prime * np.eye(tx.shape[1]), tx.T.dot(y))

    loss = compute_loss_mse(y, tx, w)

    return w, loss


def sigmoid(t):
    """Apply sigmoid function on t."""

    return 1.0 / (1 + np.exp(-t))


def compute_loss_logistic(y, tx, w):
    """Compute the cost by negative log likelihood."""

    N = y.shape[0]
    sigma = sigmoid(tx @ w)
    loss = -1 / N * np.sum(y.T @ np.log(sigma) + (1 - y.T) @ np.log(1 - sigma))

    return loss


def compute_gradient_logistic(y, tx, w):
    """ Compute the gradient of loss."""

    N = y.shape[0]
    sigma = sigmoid(tx @ w)
    gradient = 1 / N * tx.T @ (sigma - y)

    return gradient


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Calculate the logistic regression solution using gradient descent.
       returns loss, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: scalar.
        gamma: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """

    w = initial_w

    for n_iter in range(max_iters):
        w = w - gamma * compute_gradient_logistic(y, tx, w)

    loss = compute_loss_logistic(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Calculate the regularized logistic regression solution using gradient descent.
       returns loss, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: scalar.
        gamma: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """

    w = initial_w

    for n_iter in range(max_iters):
        w = w - gamma * (compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w)

    loss = compute_loss_logistic(y, tx, w)

    return w, loss


def stoch_logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Calculate the logistic regression solution using stochastic gradient descent.
       returns loss, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: scalar.
        gamma: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """

    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            w = w - gamma * compute_gradient_logistic(minibatch_y, minibatch_tx, w)

    loss = compute_loss_logistic(y, tx, w)

    return w, loss


def stoch_reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Calculate the regularized logistic regression solution using stochastic gradient descent.
       returns loss, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: scalar.
        gamma: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """

    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            w = w - gamma * (compute_gradient_logistic(minibatch_y, minibatch_tx, w) + 2 * lambda_ * w)

    loss = compute_loss_logistic(y, tx, w)

    return w, loss
