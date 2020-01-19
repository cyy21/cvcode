import numpy as np
from collections import Counter



class KNeareastNeighbor():
    def __init__(self):
        pass
    def train(self,x,y):
        self.x_train=x
        self.y_train=y
    #def compute_dist_no_loop(self,x):
    #    pass
    #def compute_dist_one_loop(self,x):
    #    pass
    #def compute_dist_two_loop(self,x):
    #    pass
    def predict(self,x):
        num_test=x.shape[0]
        num_train=self.x_train.shape[0]
        dists=np.zeros((num_test,num_train))
        test_sum=np.sum(np.power(x,2),axis=1).reshape(-1,1)
        train_sum=np.sum(np.power(self.x_train,2),axis=1).reshape(1,-1)
        dotsum=np.dot(x,self.x_train.transpose(1,0))
        dists=test_sum+train_sum-2*dotsum


               
        return np.sqrt(dists)
    def predict_label(self,dists,k=1):
        num_test=dists.shape[0]
        y_pred=np.zeros(num_test)
        old_index=list(range(dists.shape[1]))
        for i in range(num_test):
            closest_y=np.argsort(dists[i])
            
            closest_y=closest_y[0:k]
            
            #closest中的是距离最近的图像的下标
            tmp_pred=self.y_train[closest_y]
            
            
            y_pred[i]=np.argmax(np.bincount(tmp_pred))
            
        return y_pred
            
            



