
# coding: utf-8

# In[1]:


import sys
import pdb
from os.path import isfile,join
import os
import numpy as np
import nibabel as nib
import scipy.io as sio


# In[2]:


def loadImageAsNifti(imageToRead):
    image_proxy = nib.load(imageToRead)
    imageData = image_proxy.get_fdata() # check get_data or get_fdata, dtype = uint8
    print("**** Image successfully loaded ****")
    
    return (imageData, image_proxy)


# In[4]:


def main():
    if len(argv) < 2 :   # number of arguments
        print("**ERROR!**: Few parameters used")
        sys.exit()
    
    imagefolderRead = "/home/caelyn/Desktop/dataset/OASIS_QuickNAT/training-labels-remap-256"
    imagenameRead = []
    bg_ratio = []
    wm_ratio =[]
    gm_ratio = []
    csf_ratio = []
    vdc_ratio = []
    
    # list image file
    if os.path.exists(imagefolderRead):
        imagenameRead = [file for file in os.listdir(imagefolderRead) if isfile(join(imagefolderRead,file))]
    imagenameRead.sort()
    
    for i in range(0,len(imagenameRead)):
        imageToRead = imagefolderRead + '/' + imagenameRead[i]
        print(imageToRead)
        [imageData, image_proxy]= loadImageAsNifti(imageToRead)
                    
        # relabel to GM, WM, CSF
        total_count = np.size(imageData)
        
        bg_count = imageData.count(0)
        bg_ratio[i] = np.log(total_count/bg_count)
        
        wm_count = imageData.count(128)
        wm_ratio[i] = np.log(total_count/wm_count)
        
        gm_count = imageData.count(64)
        gm_ratio[i] = np.log(total_count/gm_count)
        
        csf_count = imageData.count(255)
        csf_ratio[i] = np.log(total_count/csf_count)
        
        vdc_count = imageData.count(192)        
        vdc_ratio[i] = np.log(total_count/vdc_count)
        
        
    print("******** voxel ratio for each categories ********")
    print("background: ", bg_ratio)
    print("whitematter :", wm_ratio)
    print("greymatter: ", gm_ratio)
    print("csf: ", csf_ratio)
    print("vdc: ", vdc_ratio)
    
if __name__ == '__main__': main
    
    

