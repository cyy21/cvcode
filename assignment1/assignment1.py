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

print('Training data shape: ', x_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
plt.imshow(x_train[0].astype('uint8'))
plt.show()
#knn