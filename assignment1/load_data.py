import pickle
import os
import numpy as np
def load_cifar10batch(filename):
    f=open(filename,'rb')
    dict=pickle.load(f,encoding='bytes')

    
    x=dict[b'data']
    y=dict[b'labels']
    y=np.array(y)
    x=x.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
    f.close()
    return x,y
def load_cifar10(ROOT):
    xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(ROOT,'data_batch_{}'.format(b))
        x,y=load_cifar10batch(f)
        xs.append(x)
        ys.append(y)
    xtr=np.concatenate(xs)
    ytr=np.concatenate(ys)
    del x,y
    xte,yte=load_cifar10batch(os.path.join(ROOT,'test_batch'))
    return xtr,ytr,xte,yte

        
