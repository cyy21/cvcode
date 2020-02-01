

import numpy as np



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
    num_sample=X.shape[0]
    num_class=W.shape[1]
    score_row=np.zeros(num_class)
    for i in range(num_sample):
        score_row=np.dot(X[i],W)
        print(score_row)
        #大的数加起来大，分母大，小的除以他就会0，为了避免这种情况就减去最大值
        score_row-=np.max(score_row)
        loss-=np.log(np.exp(score_row[y[i]])/np.sum(np.exp(score_row)))
        for j in range(num_class):
            row_exp_sum=np.sum(np.exp(score_row))
            print(row_exp_sum)
            dW[:,j]+=np.reshape(np.exp(score_row[j])*X[i,:]/row_exp_sum,(-1,1))
            if j==y[i]:
                dW[:,j]-=np.reshape(X[i,:],(-1,1))

    loss+=reg*np.sum(W*W)
    dW+=2*reg*W
    

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
    num_sample=X.shape[0]
    num_class=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score=np.dot(X,W)
    score=score-np.max(score,axis=1,keepdims=True)
    
    a=np.sum(np.exp(score),axis=1)
    b=np.exp(score[range(num_sample),y])
    
    loss-=np.sum(np.log(b/a))
    tmp=np.ones_like(score)
    e_xi_w=np.reshape(np.exp(score[range(num_sample),y]),(-1,1))
    tmp=tmp*e_xi_w
  
    
    tmp/=np.reshape(np.sum(np.exp(score),axis=1),(-1,1))
    tmp[range(num_sample),y]-=1
    dW=np.dot(X.T,tmp)

    loss+=reg*np.sum(W*W)
    dW+=2*reg*W
    loss/=num_sample
    dW/=num_sample
    loss+=reg*np.sum(W*W)
    dW+=2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #num_samples = X.shape[0]

    #num_classes = W.shape[1]

    #score = X.dot(W) # N by C

    #prob = score - np.max(score, axis=1, keepdims=True)

    #prob = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True) # N by C, Mat of P

    #loss = np.sum( -1 * np.log( prob[range(num_samples),y] ) )

    #prob[range(num_samples),y] -= 1 # j == y[i] , dw = (P_ij - 1)Xi

    #dW = X.T.dot(prob) # (D by N)(N by C) = D by C



    #loss /= num_samples

    #dW /= num_samples



    #loss += reg * np.sum(W * W)

    #dW += 2 * reg * W
    return loss, dW
