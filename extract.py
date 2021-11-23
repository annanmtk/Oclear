#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[13]:


def check_extraction(im):
    zone=[]
    ######## montant en chiffre ###########
    mtc=im[5:45,545:] 
    #mtc=mtc.reshape(mtc.shape[0],mtc.shape[1]*mtc.shape[2])
    #fig, ax = plt.subplots(1, figsize=(10, 10))
    #ax.imshow(mtc, cmap='Greys_r')
    
    ######## montant en lettre écrit sur la première ligne #####
    mtl1=im[40:70,160:]
  
    ######## montant en lettre écrit sur la seconde ligne ######
    mtl2=im[70:85,25:] 
    
    ######## Date de la transaction ###########
    date=im[125:152,525:] 
    
    ######## nom du bénéficiaire #############
    nom=im[102:125,80:] 
    
    ######## nom du bénéficiaire #############
    sign=im[160:242,475:680] 
    
    ######## Lieu de transaction #############
    lieu=im[125:145,375:520] 
    
    zone=[mtc,mtl1,mtl2,date,nom,sign,lieu]
    return zone


# In[14]:


def delete_line(img):
    
    image = img.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    #cv2.imshow('thresh', thresh)
    ##cv2.imshow('detected_lines', detected_lines)
    #cv2.imshow('image', image)
    #cv2.imshow('result', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    plt.imshow(result)
    return result


# In[ ]:




