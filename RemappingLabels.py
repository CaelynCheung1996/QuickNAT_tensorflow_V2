#!/usr/bin/env python
# coding: utf-8

# In[21]:


import sys
import pdb
from os.path import isfile,join
import os
import numpy as np
import nibabel as nib
import scipy.io as sio


# In[42]:


# ID remapping list
origidGM = [31,32,36,37,38,39,47,48,55,56,57,58,59,60,210,211] # mapped to 1
origidWM =[35,40,41,44,45] # mapped to 2
origidCSF = [4,11,51,52] # mapped to 3
origidVDC = [61,62] # VDC+background
origidBG = [0]
newid = [1,2,3] # 1:GM 2:WM 3:CSF


# In[23]:


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]
    imageNames.sort()


# In[44]:


def loadImageAsNifti(imageToRead):
    image_proxy = nib.load(imageToRead)
    imageData = image_proxy.get_fdata() # check get_data or get_fdata, dtype = uint8
    print("**** Image successfully loaded ****")
    
    return (imageData, image_proxy)


# In[36]:


def saveImageAsNifti(imageToSave,imageName,imageToRead):

    [imageData,image_proxy] = loadImageAsNifti(imageToRead)
    
    # generate new nii file
    niiToSave = nib.Nifti1Image(imageToSave, image_proxy.affine)
    # niiToSave.set_data_dtype(imageType)
    dim = len(imageToSave.shape)
    zooms = list(image_proxy.header.get_zooms()[:dim])
    if len(zooms) < dim :
        zooms = zooms + [1.0]*(dim-len(zooms)) # check this part 
    
    niiToSave.header.set_zooms(zooms)
    nib.save(niiToSave, imageName)
    
    print ("**** Image succesfully saved ****")


# In[52]:


def labelsRemap(argv):
    if len(argv) < 2 :   # number of arguments
        print("**ERROR!**: Few parameters used")
        sys.exit()
    
    imagefolderRead = argv[0]
    imagefolderSave = argv[1]
    imagenameRead = []
    
    # list image file
    if os.path.exists(imagefolderRead):
        imagenameRead = [file for file in os.listdir(imagefolderRead) if isfile(join(imagefolderRead,file))]
    imagenameRead.sort()
    
    for i in range(0,len(imagenameRead)):
        imageToRead = imagefolderRead + '/' + imagenameRead[i]
        print(imageToRead)
        [imageData, image_proxy]= loadImageAsNifti(imageToRead)
        origIDindex = np.unique(imageData)
        # print(np.unique(imageData))
        # print(len(origIDindex))
        
        # correct labels
        newIDimage = np.zeros(imageData.shape,dtype=np.int8) # check dtype of original
        newimageDtype = np.dtype(np.int8)
        
        # relabel the index > 100
        for i_d in origIDindex:
            if i_d > 100:
                if i_d%2 == 0:
                    idx_even = np.where(imageData == i_d)
                    imageData[idx_even] = 210 
                elif i_d%2 != 0:
                    idx_odd = np.where(imageData == i_d)
                    imageData[idx_odd] = 211  
        
        
                    
        # print(np.unique(imageData))    
        
        # relabel to GM, WM, CSF        
        for GMid in origidGM:
            idx = np.where(imageData == GMid)
            newIDimage[idx] = 1
        
        for WMid in origidWM:
            idx = np.where(imageData == WMid)
            newIDimage[idx] = 2
        
        for CSFid in origidCSF:
            idx = np.where(imageData == CSFid)
            newIDimage[idx] = 3
            
        for VDCid in origidVDC:
            idx = np.where(imageData == VDCid)
            newIDimage[idx] = 4 
            
        for BGid in origidBG:
            idx = np.where(imageData == BGid)
            newIDimage[idx] = 0            
            
        # save nii
        print("**** Saving Labels ****")
        imagenameSave = imagefolderSave + '/' + imagenameRead[i]
        
        # new nii file properties setting
        saveImageAsNifti(newIDimage, imagenameSave, imageToRead)
        
        
    print("******** Finished Label Remapping ********")
    
if __name__ == '__main__':
    labelsRemap(sys.argv[1:]) 
    
    


# In[ ]:




