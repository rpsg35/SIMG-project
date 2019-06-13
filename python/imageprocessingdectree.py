
# importing libraries

from PIL import Image
import numpy as np
import math
import cv2
import time






#import image


i=Image.open('kuaui_hawaii.tiff')
i1=Image.open("khsmall.png")
iar=np.asarray(i)
iar1=np.asarray(i1)
print(iar1.shape)



#using same pixel data from imageprocessingbayes.py
#adding a 4th value to pixel vector which denotes its class as 0:Coast, 1:field, 2:water, 3:house, 4:road, 5:forest, 6:building


import copy 
trainingset=[]
for i in range(3351,3356):
    for j in range(2107,2117):
        n=iar[i][j]
        nx=np.append(n,0)
        trainingset.append(nx)
for i in range(1425,1430):
    for j in range(7065,7075):
        n=iar[i][j]
        nx=np.append(n,1)
        trainingset.append(nx)
for i in range(6248,6253):
    for j in range(404,414):
        n=iar[i][j]
        nx=np.append(n,2)
        trainingset.append(nx)
for i in range(3620,3630):
    for j in range(3089,3094):
        n=iar[i][j]
        nx=np.append(n,3)
        trainingset.append(nx)
for i in range(3367,3372):
    for j in range(4387,4397):
        n=iar[i][j]
        nx=np.append(n,4)
        trainingset.append(nx)
for i in range(1052,1057):
    for j in range(3876,3886):
        n=iar[i][j]
        nx=np.append(n,5)
        trainingset.append(nx)
for i in range(2023,2028):
    for j in range(268,278):
        n=iar[i][j]
        nx=np.append(n,6)
        trainingset.append(nx)

print(len(trainingset))
for row in trainingset:
    print (row)

       
    


#function testsplit to split data into 2 different parts, depending upon threshhold value and data-index
#giniindex function to calculate gini index value of split as described here https://en.wikipedia.org/wiki/Gini_coefficient

from math import log
def testsplit(index,value,data):
    left=[]
    right=[]
    for row in data:
        if (row[index]>value):
            right.append(row)
            
            
        else:
            left.append(row)
            
    groups=[left,right]
    return groups
def giniindex(groups,classes):
    te=0
    
    for group in groups:
        score=1
        size=len(group)
        
        for classval in classes:
            
            if(size==0):
                prob=0
            else:    
                prob=[row[-1] for row in group].count(classval)/size    
            
                score=score-(prob*prob)
        te=te+score
    return te


#createsplit will loop through all pixel values and will return an object node with the following values- 'groups' - 2 arrays of pixel vectors ; 'index' - index which provided best gini coefficient - value under the 'index' column which will provide the best gini coefficient

def createsplit(data):
    bestgini=9999
    classes=[0,1,2,3,4,5,6]
    bestindex=None
    bestval=None
    
    
    for row in data:
        
            
            
        
        for i in range(0,len(row)-1):
            groups=testsplit(i,row[i],data)
            gn=giniindex(groups,classes)
            
            if(gn<bestgini):
                bestindex=i
                bestval=row[i]
                bestgini=gn
                #print('value is ',bestgini,'at ',bestindex,'and ',bestval)
                
                
    bestsplit=testsplit(bestindex,bestval,data)
    node={'groups':bestsplit,'index':bestindex,'value':bestval}
    
    
    return node
    

# to_terminal function will give the maximum class instances present in an array of  pixel+class concatenated  vectors.


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# split function will split a tree node if needed.  


def split(node, max_depth, min_size, depth):
    left, right = node['groups'] 
    print('left = ',left)
    print('right =',right)
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    
    if depth >= max_depth:
        print('depth exceeded')
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        print('left size small')
        node['left'] = to_terminal(left)
    else:
        node['left'] = createsplit(left)
        
        split(node['left'], max_depth, min_size, depth+1)
        
    if len(right) <= min_size:
        print('right size small')
        node['right'] = to_terminal(right)
    else:
        node['right'] = createsplit(right)
        
        split(node['right'], max_depth, min_size, depth+1)


# function to predict , input param  row is  a pixel vector


def predict(node, row):
    if row[node['index']] <= node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# function to build  tree from complete data base

def build_tree(train, max_depth, min_size):
    root = createsplit(train)
    
    split(root, max_depth, min_size, 0)
    return root


# here we built tree from our training sample with max depth 6 and minimum size required to split a node equal to 5.


tre=build_tree(trainingset,6,5)
print(tre)



# analysing percentwise distribution of classes


coastn=0
fieldn=0
watern=0
housen=0
roadn=0
forestn=0
buildingn=0
j=0
t1=time.time()
for row in iar1:
    for pix in row:
        if(j%100==0):
            print(j)
            print(time.time()-t1)
        j=j+1
        if(predict(tre,pix)==0):
            coastn=coastn+1
        if(predict(tre,pix)==1):
            fieldn=fieldn+1
        if(predict(tre,pix)==2):
            watern=watern+1
        if(predict(tre,pix)==3):
            housen=housen+1
        if(predict(tre,pix)==4):
            roadn=roadn+1
        if(predict(tre,pix)==5):
            forestn=forestn+1
        if(predict(tre,pix)==6):
            buildingn=buildingn+1



#mapping process with giving same pixel value for each class

sum=coastn+fieldn+watern+housen+roadn+forestn+buildingn
print ('% of forest in image is',forestn*100/sum)
print ('% of field in image is',fieldn*100/sum)
print ('% of road in image is',roadn*100/sum)
print ('% of house in image is',housen*100/sum)
print ('% of water in image is',watern*100/sum)
print ('% of coast in image is',coastn*100/sum)
print ('% of building in image is',buildingn*100/sum)










fieldmap=np.array([165, 191, 126])
coastmap=np.array([232, 224, 178])
watermap=np.array([30, 48, 206])
housemap=np.array([191, 43, 26])
roadmap=np.array([122, 115, 115])
forestmap=np.array([19, 66, 32])
buildingmap=np.array([129, 130, 110])
maparray=np.zeros([720,720,3])
for i in range(720):
    for j in range(720):
        getclass=predict(tre,iar1[i][j])
        if (getclass==0):
            maparray[i][j]+=fieldmap
        elif (getclass==1):
            maparray[i][j]+=coastmap
        elif (getclass==3):
            maparray[i][j]+=watermap
        elif (getclass==2):
            maparray[i][j]+=housemap
        elif (getclass==4):
            maparray[i][j]+=roadmap
        elif (getclass==5):
            maparray[i][j]+=forestmap
        elif (getclass==6):
            maparray[i][j]+=buildingmap
    print(i) 



# saving map


cv2.imwrite("map2.png",maparray)







