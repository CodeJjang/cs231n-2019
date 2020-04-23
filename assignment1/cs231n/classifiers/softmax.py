from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
      curr_scores = W.T.dot(X[i])
      prob = np.exp(curr_scores[y[i]]) / np.exp(curr_scores).sum()
      loss += -1 * np.log(prob)
      
      for j in range(num_classes):
        prob = np.exp(curr_scores[j]) / np.exp(curr_scores).sum()
        dW[:, j] += (prob - (j == y[i])) * X[i]
      
    
    loss /= num_train
    loss += reg * (W ** 2).sum().sum()

    dW /= num_train
    dW += 2 * W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    scores_exp = np.exp(scores)
    scores_sum = scores_exp.sum(axis = 1)
    probs = scores_exp[np.arange(num_train), y] / scores_sum
    loss = -1 * np.log(probs).sum()

    probs = scores_exp / scores_sum.reshape(-1, 1)
    probs[np.arange(num_train), y] -= 1
    dW += X.T.dot(probs)
    
    
    loss /= num_train
    loss += reg * (W ** 2).sum().sum()

    dW /= num_train
    dW += 2 * W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
  
