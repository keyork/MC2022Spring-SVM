import numpy as np

class LinearSVM(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate, reg, num_iters,
            batch_size=100, verbose=True):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None
      
      random_idxs = np.random.choice(num_train, batch_size)
      X_batch = X[random_idxs]
      y_batch = y[random_idxs]
      
      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update 
      self.W -= learning_rate * grad  

      # if verbose and it % 100 == 0:
      #   print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.argmax(np.dot(X, self.W), axis=1)
    
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Question 1: Implement the regulated hinge loss function and its gradient. 
    The computation process should be better fully-vectorized, i.e., without loop over the samples.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss: a single float
    - gradient dW: an array of the same shape as W
    """

    #############################################################################
    #                             BEGIN OF YOUR CODE                            #
    #############################################################################

    batch_size, _ = X_batch.shape

    pred_score = np.dot(X_batch, self.W)
    crt_rows = np.arange(batch_size).reshape(batch_size, 1)
    crt_cols = y_batch.reshape(batch_size, 1)
    crt_score = pred_score[crt_rows, crt_cols]

    margin = np.maximum(0, pred_score-crt_score+1)
    margin[crt_rows, crt_cols] = 0

    loss = 0.5*reg*np.sum(self.W*self.W) + (np.sum(margin)) / batch_size

    margin[margin>0] = 1
    margin[crt_rows, crt_cols] = -np.sum(margin, axis=1).reshape(batch_size, 1)

    dW = np.dot(np.transpose(X_batch), margin) / batch_size + reg * self.W

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


# # test
# if __name__ == '__main__':
#     # dim: 10, batch_size: 100, cls_num: 8
#     svm = LinearSVM()
#     svm.W = 0.001 * np.random.randn(10, 8)
#     X_batch = np.random.randn(100, 10)
#     y_batch = np.random.randint(0, 7, (100, 1))
#     reg = 0.1
#     svm.loss(X_batch, y_batch, reg)
    