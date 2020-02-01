import numpy as np
import matplotlib.pyplot as plt
import random
import imageio
import pickle
import os
from load_data import load_cifar10
from PIL import Image
import time

cifar10_dir='D:\\assignment1\\cifar-10-batches-py'
x_train,y_train,x_test,y_test=load_cifar10(cifar10_dir)


x_train=np.reshape(x_train,(x_train.shape[0],-1))

x_test = np.reshape(x_test, (x_test.shape[0], -1))

#print(mean_image[:10])
#plt.figure(figsize=(4,4))
#plt.imshow(mean_image.reshape((32,32,3)).astype('uint8'))
#plt.show()


#x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
#x_val = np.hstack([x_val, np.ones((x_val.shape[0], 1))])
#x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
#x_dev = np.hstack([x_dev, np.ones((x_dev.shape[0], 1))])


num_sample=x_train.shape[0]
num_class=10
din=x_train.shape[1]
dout=10
H=100
w1=np.random.randn(din,H)*0.0001
w2=np.random.randn(H,dout)*0.0001
loss=0.0
def f(a,num,y,reg,w1,w2):
    a=np.exp(a)
    a_correct=a[range(num),y]
    a_correct=a_correct.reshape((-1,1))
    a_row_sum=np.sum(a,axis=1)
    a_row_sum=np.reshape(a_row_sum,(-1,1))
    
    loss=0.0
    loss-=np.sum(np.log(a_correct/a_row_sum))
    loss/=num
    grad_a=np.zeros_like(a)
    
    grad_a=a/a_row_sum
    grad_a[range(num),y]-=1
    loss+=reg*(np.sum(w1*w1)+np.sum(w2*w2))
    return loss,grad_a


num_trainn=x_train.shape[0]
reg=2.5e4

for t in range(200):
    y1=np.dot(x_train,w1)
    y1_relu=np.maximum(y1,0)
    y2=np.dot(y1_relu,w2)
    loss,grad_y2=f(y2,y2.shape[0],y_train,reg,w1,w2)
    grad_w2=np.dot(y1_relu.T,grad_y2)
    grad_y1=np.dot(grad_y2,w2.T)*(y1>0)
    
    grad_w1=np.dot(x_train.T,grad_y1)
    grad_w1/=num_trainn
    grad_w2/=num_trainn
    grad_w1+=2*reg*w1
    grad_w2+=2*reg*w2
    w1-=(5e-7)*grad_w1
    w2-=(5e-7)*grad_w2
    print("di{}loss {}".format(t,loss))



y1=np.dot(x_train,w1)
y1_relu=np.maximum(y1,0)
y2=np.dot(y1_relu,w2)
y_pred=np.argmax(y2,axis=1)
acc=np.mean(y_pred==y_train)
print(acc)
y1=np.dot(x_test,w1)
y1_relu=np.maximum(y1,0)
y2=np.dot(y1_relu,w2)
y_pred=np.argmax(y2,axis=1)
acc=np.mean(y_pred==y_test)
print(acc)


    
    




















#from softmaxclassfier import *
#W=np.random.randn(3073,10)*0.0001
#from linearclass import  LinearClassifier

#sft=LinearClassifier()
#loss=sft.train(x_train,y_train, learning_rate=1e-7, reg=2.5e4,num_iters=1500, verbose=True)
#y_pred=sft.predict(x_test)
#acc=np.mean(y_pred==y_test)
#print(acc)

############################### svm loss ###########################
#from svm import svm_loss_naive
#from svm import svm_loss_vectorized
#W=np.random.randn(3073,10)*0.0001

#from linearclass import  LinearClassifier
#ssvm=LinearClassifier()

#loss=ssvm.train(x_train,y_train, learning_rate=1e-7, reg=2.5e4,num_iters=1500, verbose=True)
#y_pred=ssvm.predict(x_test)
#acc=np.mean(y_pred==y_test)
#print(acc)

#ssvm1=LinearClassifier()
#loss=ssvm1.train(x_train,y_train, learning_rate=5e-5, reg=5e4,num_iters=1500, verbose=True)
#y_pred=ssvm1.predict(x_test)
#acc=np.mean(y_pred==y_test)
#print(acc)


































#################################### the knn part #################################
#for 画图
#print('training data shape: ', x_train.shape)
#print('training labels shape: ', y_train.shape)
#print('test data shape: ', x_test.shape)
#print('test labels shape: ', y_test.shape)
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
#num_training=5000
#mask=list(range(0,num_training))
#x_train=x_train[mask]
#y_train=y_train[mask]

#num_test=500
#mask=list(range(num_test))
#x_test=x_test[mask]
#y_test=y_test[mask]
#x_train=np.reshape(x_train,(x_train.shape[0],-1))
#x_test=x_test.reshape((x_test.shape[0],-1))
#print(x_train.shape,x_test.shape)



#from knearstneighbor import KNeareastNeighbor
#knn=KNeareastNeighbor()
#knn.train(x_train,y_train)
#x_test_dist=knn.predict(x_test)


#x_test_pred=knn.predict_label(x_test_dist,k=5)

#num_correct=np.sum(x_test_pred==y_test)
#accuracy=float(num_correct)/num_test
#print('k=8:accuracy:{:.2f}:{}/{}'.format(accuracy,num_correct,num_test))

#x_test_pred=knn.predict_label(x_test_dist)
#num_correct=np.sum(x_test_pred==y_test)
#accuracy=float(num_correct)/num_test
#print('k=1:accuracy:{:.2f}:{}/{}'.format(accuracy,num_correct,num_test))
#num_folds=5
#k_choices=[1,3,5,8,10,12,15,20,50,100]

#x_train_folds=np.array_split(x_train,num_folds)
#y_train_folds=np.array_split(y_train,num_folds)

#k2accuracy={}
#len_per_fold=np.ceil(x_train.shape[0]/num_folds)
#for i in k_choices:
#    list_acc=np.zeros(num_folds)
#    for j in range(num_folds):
#        cur_x_test=np.array(x_train_folds[j])
#        cur_y_test=np.array(y_train_folds[j])
#        start=int(j*len_per_fold)
#        end=int((j+1)*len_per_fold)
#        cur_x_train=np.delete(x_train,np.s_[start:end],0)
#        cur_y_train=np.delete(y_train,np.s_[start:end],0)
#        knn.train(cur_x_train,cur_y_train)
#        dists=knn.predict(cur_x_test)
#        y_test_pred=knn.predict_label(dists,k=i)
        
#        num_correct = np.sum(y_test_pred == y_train_folds[j])
#        accuracy = float(num_correct) / len_per_fold
#        list_acc[j] = accuracy
#    k2accuracy[i]=list_acc
#for k in sorted(k2accuracy):
#    for accuracy in k2accuracy[k]:
#        print('k = %d, accuracy = %f' % (k, accuracy))

#for k in k_choices:
#    accuracies = k2accuracy[k]
#    plt.scatter([k] * len(accuracies), accuracies)
#acc_mean=np.array([np.mean(v) for k,v in sorted(k2accuracy.items())])
#acc_std=np.array([np.std(v) for k,v in sorted(k2accuracy.items())])
#plt.errorbar(k_choices,acc_mean,yerr=acc_std)
#plt.title('cross validation')
#plt.xlabel('k')
#plt.ylabel('acc')
#plt.show()


################################# end of knn part ################################