import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    delta = 1
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    scores = X.dot(W)
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    mask = np.ones(scores.shape, dtype=bool)  # 500 x 10
    mask[np.arange(mask.shape[0]), y] = False
    margins = np.maximum(0, scores.reshape(mask.shape[0], -1)
                         - scores[~mask].reshape(-1, 1) + delta)
    margins[~mask] = 0
    loss = np.sum(margins)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    # X Dim = 500 x 3073
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    X_count = np.ones(margins.shape)  # 500 x 10
    X_count[margins == 0] = 0
    y_count = np.sum(X_count, axis=1)
    X_count[np.arange(num_train), y] = -y_count
    dW = X.transpose().dot(X_count)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
