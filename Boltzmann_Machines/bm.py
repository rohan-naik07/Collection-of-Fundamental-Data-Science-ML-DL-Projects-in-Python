import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')

train_set=pd.read_csv('ml-100k/u1.base',delimiter='\t')
train_set=np.array(train_set,dtype='int')
test_set=pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set=np.array(test_set,dtype='int')

nb_users= int(max(max(train_set[:,0]),max(test_set[:,0])))
nb_movies= int(max(max(train_set[:,1]),max(test_set[:,1])))

# users in rows and movies in columns
def convert(data):
    new_data=[]
    for id_users in range(1,nb_users+1):
        id_movies= data[:,1][data[0]==id_users]
        id_ratings= data[:,2][data[0]==id_users]
        ratings=np.zeros(nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(ratings)
    return new_data
train_set=convert(train_set)
test_set=convert(test_set)

train_set=torch.FloatTensor(train_set)
test_set=torch.FloatTensor(test_set)

train_set[train_set==0]=-1
train_set[train_set==1]=0
train_set[train_set==2]=-0
train_set[train_set>=3]=1

test_set[test_set==0]=-1
test_set[test_set==1]=0
test_set[test_set==2]=-0
test_set[test_set>=3]=1

class RBM():
    def __init__(self,nv,nh):
        self.W=torch.randn(nh,nv)
        self.a=torch.randn(1,nh) # bias for hidden nodes
        self.b=torch.randn(1,nv) # bias for visible nodes

    def sample_h(self,x):
        wx=torch.mm(x,self.W.t())
        activation=wx+self.a.expand_as(wx) #bias applied to each line of mini batch
        ph_given_v=torch.sigmoid(activation)
        return ph_given_v,torch.bernoulli(ph_given_v)

    def sample_v(self,y):
        wy=torch.mm(y,self.W)
        activation=wy+self.b.expand_as(wy) #bias applied to each line of mini batch
        pv_given_h=torch.sigmoid(activation)
        return pv_given_h,torch.bernoulli(pv_given_h)

    def train(self,v0,vk,ph0,phk):
        self.W+=torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)
        self.b+=torch.sum((v0-vk),0)
        self.a+=torch.sum((ph0-phk),0)

nv=len(train_set[0])
nh= 100 # no of features we want to detect
batch_size=100
model=RBM(nv=nv,nh=nh)
no_of_epochs=10

for epoch in range(1,no_of_epochs+1):
    train_loss=0
    s=0.
    for id_user in range(0,nb_users-batch_size,batch_size):
        vk=train_set[id_user:id_user+batch_size] 
        v0=train_set[id_user:id_user+batch_size]
        ph0,_= model.sample_h(v0)
        for k in range(10):
            _,hk=model.sample_h(vk) # features from movies
            _,vk=model.sample_v(hk) # movies from extracted features
            vk[v0 < 0] = v0[v0<0]
        phk,_=model.sample_h(vk)  
        model.train(v0,vk,ph0,phk)
        train_loss+=torch.mean(torch.abs(v0[v0 >= 0]-vk[v0 >=0]))
        s+=1.
    print('loss after epoch '+str(epoch) +' is '+str(train_loss/s))


test_loss=0
s=0.
for id_user in range(0,nb_users):
    v=train_set[id_user:id_user+1] 
    vt=test_set[id_user:id_user+1]
    if len(vt[vt>=0]) >0:
        _,h=model.sample_h(v) # features from movies
        _,v=model.sample_v(h) # movies from extracted features
        test_loss+=torch.mean(torch.abs(v0[v0 >= 0]-vk[v0 >=0]))
        s+=1.
print('loss is '+str(test_loss/s))
