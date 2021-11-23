#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import Levenshtein as lev
def levCalclulate(str1, str2):
    Distance = lev.distance(str1, str2)
    Ratio = lev.ratio(str1, str2)
    #print("Levenshtein entre {0} et {1}".format(str1, str2))
    #print("> Distance: {0}\n> Ratio: {1}\n".format(Distance, Ratio))
    return Distance


# In[3]:


def answers(word):
    Price_list=["UN","DEUX","TROIS","QUATRE","CINQ","SIX","SEPT","HUIT","NEUF","DIX","ONZE","DOUZE","TREIZE","QUATORZE","QUINZE",
          "SEIZE","VINGT","TRENTE","QUARANTE","CINQUANTE","SOIXANTE","CENT","MILLE","MILLES","MILLION","MILLIONS","MILLIARD","MILLIARDS",'DE','FRANC','FRANCS']
    dic={}
    for k,ch in enumerate(Price_list):
        dic[k]=ch
    l=[]
    for ch in Price_list:
        l.append(levCalclulate(word,ch))
    return dic[np.argmin(l)]


# In[ ]:





# In[ ]:





# In[ ]:




