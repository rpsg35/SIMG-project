
# importing libraries


from PIL import Image
import numpy as np
import math

#import image

i1=Image.open('kuaui_hawaii.tiff')
iar=np.asarray(i1)
iar.setflags(write=1)
#print (iar.shape)


#collecting pixel data for different classes and creating seperate 2d array for each class


coast= []
for m in range(3351,3356):
    for n in range(2107,2117):
        coast.append(iar[m][n])

    




field= []
for m in range(1425,1430):
    for n in range(7065,7075):
        field.append(iar[m][n])



water= []
for m in range(6248,6253):
    for n in range(404,414):
        water.append(iar[m][n])
        




house= []
for m in range(3620,3630):
    for n in range(3089,3094):
        house.append(iar[m][n])
        




road= []
for m in range(3367,3372):
    for n in range(4387,4397):
        road.append(iar[m][n])
        




forest= []
for m in range(1052,1057):
    for n in range(3876,3886):
        forest.append(iar[m][n])
        




building= []
for m in range(2023,2028):
    for n in range(268,278):
        building.append(iar[m][n])
        
print(building[5])


# gaussian function -  returns 1d vector inputs taken - (pixel vector, class mean vector , standard deviation vector) 


from math import exp
def gaussian(x,m,d):
    val=[1,1,1]    
    fc=((1/(2*3.14*d*d)*np.exp((-0.5)*((x-m)/d)**2)))
    val=(val*fc)
    v2=1
    for i in range(3):
        v2=v2*fc[i]
    return v2








# function to calculate SD and mean for each training class


import math
def clsmean(cls):
    val=[0,0,0]
    for eachpix in cls:
        val+=eachpix
    return(val/len(cls))
def clsdev(cls):
    val=[0,0,0]
    meanval=clsmean(cls)
   
    for eachpix in cls:
        val+=(eachpix-meanval)**2
    
    le=len(cls)
    val2=[0,0,0]
    
    val2+=val/le
    dev=[0,0,0]
    
    dev+=np.sqrt(val2)
        
    return dev





        
        
        


# stacking all classes together with their sd and  mean in difference 
#pocarray is vector of length 7 having same value (1/7) in all indexes since all class arrays have equal length.
#function classifpix gives probability score of a pixel wrt to a class using gaussian naive bayes distribution explained here https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes


classes=[field,coast,water,house,road,forest,building]
devarray=[clsdev(eachclass) for eachclass in classes]
meanarray=[clsmean(eachclass) for eachclass in classes]
sumofclasses=sum(len(eachclass) for eachclass in classes)
pocarray=[(len(eachclass)/sumofclasses) for eachclass in classes]
def classifpix(pix,k):
    sum=0
    for i in range(len(devarray)):
        sum+=((pocarray[i])*gaussian(pix,meanarray[i],devarray[i]))
    val=(pocarray[k]*gaussian(pix,meanarray[k],devarray[k]))/sum
    return val
#the following function will classify pixel comparing all its probability scores among all classes.

              




def mapclassify(pix):
    probarray=[]
    for eachclass in classes:
        val=classifpix(pix,eachclass)
        probarray.append(val)
    classno=np.argmax(probarray)
    return classno



#vv written while process to check speed 


import time


i=0
t1=time.time()
for eachrow in iar:
    for eachpix in eachrow:
        classify(eachpix)
        if (i%100==0):
            print (i)
            
            print(time.time()-t1)
        i+=1
        
        
   

a = np.array([1, 2, 3])
classify(a)


# used shrinked version of the same image for mapping because the original one was large and taking long time without any gains.


import cv2
filename = "kuaui_hawaii.tiff"
oriimage = cv2.imread(filename)

newimage = cv2.resize(oriimage,(720,720))
cv2.imwrite("khsmall.png",newimage)

   
  

i2=Image.open('khsmall.png')
iar2=np.asarray(i2)
iar2.setflags(write=1)
print (iar2.shape)


# analysing percentwise distribution of classes


fieldn=0
coastn=0
watern=0
housen=0
roadn=0
forestn=0
buildingn=0
j=0
t1=time.time()
for eachrow in iar2:
    for eachpix in eachrow:
        if(predictclass(eachpix)==0):
            fieldn+=1
        if(predictclass(eachpix)==1):
            coastn+=1
        if(predictclass(eachpix)==2):
            watern+=1
        if(predictclass(eachpix)==3):
            housen+=1
        if(predictclass(eachpix)==4):
            roadn+=1
        if(predictclass(eachpix)==5):
            forestn+=1
        if(predictclass(eachpix)==6):
            buildingn+=1
        
        
    
        
    

    






classsum=518400
perforest=forestn/classsum*100
perwater=watern/classsum*100
perroad=roadn/classsum*100
perfield=fieldn/classsum*100
percoast=coastn/classsum*100
perbuilding=buildingn/classsum*100
perhouse=housen/classsum*100



print ('% of forest in image is',perforest)
print ('% of field in image is',perfield)
print ('% of road in image is',perroad)
print ('% of house in image is',perhouse)
print ('% of water in image is',perwater)
print ('% of coast in image is',percoast)
print ('% of building in image is',perbuilding)


#mapping process with giving same pixel value for each class


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
        getclass=predictclass(iar2[i][j])
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


cv2.imwrite("map.png",maparray)






