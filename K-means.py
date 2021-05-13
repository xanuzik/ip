import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

def run():
    
    #read image and get rgb values of all pixels
    img = cv2.imread("123.png")
    arrimg = np.array(img)

    features=[]
    row=arrimg.shape[0]
    col=arrimg.shape[1]
    for i in range(0, row):
        for j in range(0, col):
            r = arrimg[i, j, 2]
            g = arrimg[i, j, 1]
            b = arrimg[i, j, 0]
            features.append([r, g, b])

    #reverse red and blue channel of the oringinal image to prepare for result output
    img = cv2.imread('123.png', cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    kclasses=[]  # kmeans >>> kclasses
    
    #randomly define 5 seed vectors, since k=5
    for i in range (0,5):
        a=random.randint(0,255)
        b=random.randint(0,255)
        c=random.randint(0,255)
        kclasses.append([a,b,c])
    print(f"Select seed vector as {kclasses}\n Start testing the availabilty of the vector \n")
    
    #calculate distance between each pixel to each seed vectors
    dist=[]
    pclass=[] #point classes>> 0,1,2,3,4
    for feature in features:
        feature=np.array(feature)
        for kclass in kclasses:
            kclass=np.array(kclass)
            #calulate distance to five seed vectors,append these 5 distances to a single list
            dist.append(np.linalg.norm(kclass-feature))
        #find the smallest distance, 
        #if the smallest one is the first one in the list, which is dist[0], then this pixel categorized to class 0, et cetera.
        #pixels then are classified as 0,1,2,3,4, stored in list pclass[]
        pclass.append(dist.index(min(dist)))
        dist=[]
    
    #check if there is a class contains no pixel, if yes, reselect seed vector. 
    for i in range (0,5):
        if pclass.count(i) == 0:
            print("Some class contains no pixel, reselecting seed vector\n")
            return False
        else:
            continue            
    print(f"The seed vector has been decied as \n {kclasses}\n, now start iteration.\n")
                    
#计算每个向量和类质心的距离
#生成5个距离值分别对应klcasses[0-4]
#5个距离值放入dist[]
#dist[]最小的值的次序，则为这个向量的类
#将类依次存入plcass[]
#就有了features[]和plcass[]一一对应的关系

#start 10 times iteration
    classdict={}
    
    j=0
    while (j<25):
        dist=[]
        pclass=[]
        myindex = 0
        for feature in features:
            feature=np.array(feature)
            for kclass in kclasses:
                kclass=np.array(kclass)
                dist.append(np.linalg.norm(kclass-feature))
            pclass.append(dist.index(min(dist)))
            dist=[]

        class0=[]
        class1=[]
        class2=[]
        class3=[]
        class4=[]
        i=0


        while (i<len(features)):
            if pclass[i]==0:
                class0.append(features[i])
            elif pclass[i]==1:
                class1.append(features[i])
            elif pclass[i]==2:
                class2.append(features[i])
            elif pclass[i]==3:
                class3.append(features[i])
            else:
                class4.append(features[i])
            i=i+1
            
        print(f"After {j} time(s) of iteration, the numbers of pixels in each class \n class1: {len(class0)},\n class2: {len(class1)}, \n class3: {len(class2)},\n class4: {len(class3)},\n class5: {len(class4)}\n")
        
        #calculate the new seed vector for next iteration
        arrclass0=np.array(class0)
        arrclass1=np.array(class1)
        arrclass2=np.array(class2)
        arrclass3=np.array(class3)
        arrclass4=np.array(class4)
        a=(np.mean(arrclass0,axis=0))
        b=(np.mean(arrclass1,axis=0))
        c=(np.mean(arrclass2,axis=0))
        d=(np.mean(arrclass3,axis=0))
        e=(np.mean(arrclass4,axis=0))
        kclasses=[[a[0],a[1],a[2]],[b[0],b[1],b[2]],[c[0],c[1],c[2]],[d[0],d[1],d[2]],[e[0],e[1],e[2]]]
        print(f"Seed vector after {j+1} times of iteration is \n {kclasses} \n")
        j=j+1

    #interation ends
    #if the pixel has been decided as class0, then fill the pixel with the RGB value of class0, if pixel class1, fill it with RGB value of class1, et cetera.
    i=0
    newpfeatures=[]
    while (i<len(features)):
        if pclass[i]==0:
            newpfeatures.append(kclasses[0])
        elif pclass[i]==1:
            newpfeatures.append(kclasses[1])
        elif pclass[i]==2:
            newpfeatures.append(kclasses[2])
        elif pclass[i]==3:
            newpfeatures.append(kclasses[3])
        else:
            newpfeatures.append(kclasses[4])
        i=i+1

    #define a 0 matrix of 243 rows, 320 colums, and 3 units in each element
    output=np.zeros(shape=(row,col,3))
    loop=0
    for hang in range(0,row):
        for lie in range(0,col):
            for rgb in range(0,3):
                #map RGB value of each class to the correspoding elements in the matrix 1 by 1.
                output[hang][lie][rgb]=int(newpfeatures[loop][rgb])
            loop=loop+1

    
    #plot input and output
    plt.figure(figsize=(15,15))
    plt.subplot(121), plt.imshow(img), plt.title('input')
    plt.subplot(122), plt.imshow(output.astype('uint8'), 'gray'), plt.title('K-means output, k=5')
    plt.show()
#     classdict['time']=j
#     classdict['1']=len(class0)
#     classdict['2']=len(class1)
#     classdict['3']=len(class2)
#     classdict['4']=len(class3)
#     classdict['5']=len(class4)
#     print(pd.DataFrame(classdict))
    
flag = run()
while flag==False:
    flag = run()
