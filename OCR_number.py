#!/usr/bin/env python
# coding: utf-8

# In[9]:


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os
import imutils
from PIL import Image
import math as m
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 10.0)

def bilateral_norm(img):
    img = cv2.bilateralFilter(img, 9, 15, 30)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
def histogram_norm(img):
    img = bilateral_norm(img)
    add_img = 255 - cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # Filtrage en image dont les pixels de premier plan sont blancs
    img = 255 - img #filtrage en noir 
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255  # normalisation      
    hist, bins = np.histogram(img.ravel(), 256, [0,256]) # img.ravel pour transformer l'image en un vecteur
                                                         #  
    img = img.astype(np.uint8) # 

    ret,thresh4 = cv2.threshold(img,np.argmax(hist)+10,255,cv2.THRESH_TOZERO)
    return add_img
    return cv2.add(add_img, thresh4, dtype=cv2.CV_8UC1)
def cropp(img):
    h,w = img.shape
    top=0
    down=0
    left=0
    right=0
    
    halt = False
    for i in range(h):
        if halt:
            break
        for j in range(w):
            if img[i,j] == 0:
                halt = True
                top = i-1
                break
                
    halt = False
    for i in reversed(range(h)):
        if halt:
            break
        for j in range(w):
            if img[i,j] == 0:
                halt = True
                down = i+1
                break
    
    halt = False
    for i in range(w):
        if halt:
            break
        for j in range(h):
            if img[j,i] == 0:
                halt = True
                left = i-1
                break
                
    halt = False
    for i in reversed(range(w)):
        if halt:
            break
        for j in range(h):
            if img[j,i] == 0:
                halt = True
                right = i+1
                break
                
    if (top < 0): top = 0
    if (down < 0): down = 0
    if (left < 0): left = 0
    if (right < 0): right = 0
        
    #print('Top: ', top)
    #print('Down: ', down)
    #print('Left: ', left)
    #print('Right: ', right)
    
    return img[top:down, left:right]
    
def find_indexe(s):
    index=[]
    i=0
    while i<=len(s)-1:
        cnt_0=0
        j=0
        #if i==0 and s[i]==0:
            #index.append(cnt_0)
            #i+=1
            
        if s[i]!=0 and i<=len(s)-1:
            i+=1
        else:
            j=i-1
            while s[i]==0 and i<=len(s)-1:
                if i==len(s)-1:
                    cnt_0+=1
                    i+=1
                    break
                else:
                    cnt_0+=1
                    i+=1
            index.append(m.ceil(cnt_0/2)+j)
                
                
    return index
        
import statistics as st
def binary_graph(thresh):
    # Binary graph
    (h, w) = thresh.shape[:2] # h est le nombre de pixels totale par colonne et w le nombre de pixels totale par ligne
    sumCols = []
    for j in range(w):  
        col = thresh[0:h, j:j+1] # y1:y2, x1:x2
        sumCols.append(h-(np.sum(col)/255))
    hist=np.zeros(len(sumCols))
    #mpx=np.min(sumCols)
    mpx=0
    for pix in range(len(sumCols)):
        if sumCols[pix]>mpx:
            hist[pix]=1
        else:
            continue
    return hist
def binary_index(hist):
    # Find index
    import math as m
    indexes=find_indexe(hist)
    return indexes

def segment1(img, indexes):
    if len(indexes)==0:
        return 1
    else:
        #pw=int(np.mean(peack_width(indexes)))
        width=img.shape[1]
        height=img.shape[0]
        rois = []
        indexes = np.insert(indexes, 0, 0)
        indexes = np.insert(indexes, len(indexes), width-1)
        #print("svee: ", indexes)
        first = 0
        second = 1
        while (first < len(indexes)) and (second < len(indexes)):
            width = indexes[second] - indexes[first]
            if width <9: # paramètre de réglage
                second += 1
                continue
            roi = img[0:height, indexes[first]:indexes[first]+width]
            rois.append(roi)
            first = second
            second += 1
        rois.reverse()
        for i in range(len(rois)):
            if rois[i].shape[1]:
                rois[i]=imutils.rotate(rois[i],180)
                #ax = plt.subplot(1, len(rois), i+1)
                rois[i]=cv2.resize(rois[i],(28,28))
#                 plt.imshow(rois[i], cmap='binary')
#                 plt.title(i, fontdict={'fontsize': 15, 'color': 'black'})
#                 plt.axis('off')
#         plt.tight_layout()
#         plt.show()
        return rois 

def preprocessing(img):
    img_copy=img.copy()
    img = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    thresh = 255-histogram_norm(img) 
    thresh = cropp(thresh)
    return thresh

def seg_number(path):
    img=cv2.imread(path)
    im=img[30:64,525:] # montant en chiffre
    cv2.imwrite("c.png",im)
    thresh=preprocessing(im)
    rotated = imutils.rotate(thresh, 180)
    h=thresh.shape[0]
    img=rotated[10:,:].copy()
    #img=thresh[7:int(h/1.7),:]
    b=binary_graph(img)
    f=binary_index(b)
    if len(f)==0:
        return 0
    else:
        s=segment1(rotated,f)
        return s
    


# In[ ]:




