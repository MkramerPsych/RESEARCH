################################################################################
#                     Inference Module for Trained Models                      #
#                        M. Kramer, TarrLab Summer 2019                        #
################################################################################

################################################################################
#REQUIRES: TORCH,TORCHVISION,TABULATE,MATPLOTLIB, SKIMAGE, TQDM, TENSORBOARDX  #
################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import skimage.color as color
from tqdm import tqdm
from tabulate import tabulate
from PIL import ImageFilter
from customLoader import ImageFolderWithPaths
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import csv
import copy

if __name__ == '__main__': #PREVENT BROKEN PIPE ERROR, ALLOW FOR MULTIPROCESSING

    def colorspace_convert(img):
        img = img.permute(1,2,0)
        img = color.rgb2lab(img)
        #img = color.lab2rgb(img)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)
        img = img.to(dtype=torch.float32)
        return img

    def gaussian_blur(img):
        img = img.filter(ImageFilter.GaussianBlur(radius=2.5))
        return img

    data_transforms = transforms.Compose([ #VALIDATION TRANSFORMS
            transforms.Resize((224,224)),
            transforms.CenterCrop((100,100)),
           # transforms.Grayscale(3),
            #transforms.Lambda(lambda img: gaussian_blur(img)), #GAUSSIAN BLUR TRANSFORM (ON PIL IMAGE)
            transforms.ToTensor(),
           # transforms.Lambda(lambda img: colorspace_convert(img)), #LAB COLORSPACE CONVERSION (ON TORCH TENSOR)
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    data_dir = 'data/VGG_Subset_600000/val' #PATH TO DATA FROM CURRENT DIRECTORY

    image_dataset = ImageFolderWithPaths(data_dir,transform=data_transforms)  #CREATE DATASETS FROM FOLDERS

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1,shuffle=True, num_workers=4) #CREATE LOADERS FROM DATASETS

    dataset_size = len(image_dataset) #GET NUM OF IMAGES IN EACH DATASET

    class_names = image_dataset.classes #PULL CLASS NAMES FROM DATA

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #MOVE DATA TO APPROPRIATE PROCESSING
    print(torch.cuda.device_count(), 'available GPUs') #PRINT NUMBER OF AVAILABLE GPUS
    print('Model Running on: {}'.format(device)) #RETURN CUDA:0 OR CPU

    ##RG RATIO CALCULATION##

    def RGratio(): #SET BATCH SIZE TO 1 BEFORE EXECUTING
        ratiodict = {}
        for inputs, classes, paths in tqdm(dataloader,unit_scale=dataloader.batch_size,desc='Calculating RG Ratios',dynamic_ncols=True):
            inputs = np.squeeze(inputs) #COLLAPSE BATCH DIMENSION
            red = inputs[0,:,:]
            green = inputs[1,:,:]
            blue = inputs[2,:,:]
            R_sum = torch.sum(red)
            G_sum = torch.sum(green)
            R_G_ratio = R_sum/G_sum
            ratiodict[paths] = (R_G_ratio.item(),classes.item())

        print('ratio dict loaded')
        print(len(ratiodict))
        males = []
        females = []
        for inputs, classes, paths in dataloader:
            temp = ratiodict[paths]
            if temp[1] == 1:
                males.append(temp[0])
            else:
                females.append(temp[0])
        
        print(len(males))
        print(len(females))

        with open('males.csv','w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(map(lambda x: [x], males))
        csvFile.close()
        with open('females.csv','w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(map(lambda x: [x], females))
        csvFile.close()

        plt.hist(list(ratiodict.values()), color = 'blue', edgecolor = 'black', bins = int(15/0.1))
        plt.title('Distribution of R:G Ratios in Dataset')
        plt.xlabel('R:G Ratio')
        plt.ylabel('Num Images')
        plt.show()

RGratio()
