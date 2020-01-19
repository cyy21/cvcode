import numpy as np
import matplotlib.pyplot as plt
import random
import imageio
import pickle
import os
from load_data import load_cifar10
from PIL import Image
cifar10_dir='D:\\assignment1\\cifar-10-batches-py'
x_train,y_train,x_test,y_test=load_cifar10(cifar10_dir)

#for 画图
#print('Training data shape: ', x_train.shape)
#print('Training labels shape: ', y_train.shape)
#print('Test data shape: ', x_test.shape)
#print('Test labels shape: ', y_test.shape)
#classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#num_classes = len(classes)

#sample_per_class=7
#for i in range(0,num_classes):
#    idx=np.flatnonzero(y_train==i)
#    idx=np.random.choice(idx,sample_per_class,replace=False)

#    for j in range(0,len(idx)):
#        plt_idx=i*sample_per_class+j+1
#        plt.subplot(num_classes,sample_per_class,plt_idx)
#        plt.imshow(x_train[idx[j]].astype('uint8'))
#        #plt.axis('off')
#        if j==0:
#            plt.ylabel(classes[i])
#plt.show()

#画图结束
num_training=5000
mask=list(range(0,num_training))
x_train=x_train[mask]
y_train=y_train[mask]

num_test=500
mask=list(range(num_test))
x_test=x_test[mask]
y_test=y_test[mask]
x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_test=x_test.reshape((x_test.shape[0],-1))
print(x_train.shape,x_test.shape)



from knearstneighbor import KNeareastNeighbor
knn=KNeareastNeighbor()
knn.train(x_train,y_train)
x_test_dist=knn.predict(x_test)


x_test_pred=knn.predict_label(x_test_dist,k=5)

num_correct=np.sum(x_test_pred==y_test)
accuracy=float(num_correct)/num_test
print('k=5:accuracy:{:.2f}:{}/{}'.format(accuracy,num_correct,num_test))

x_test_pred=knn.predict_label(x_test_dist)
num_correct=np.sum(x_test_pred==y_test)
accuracy=float(num_correct)/num_test
print('k=1:accuracy:{:.2f}:{}/{}'.format(accuracy,num_correct,num_test))
