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
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
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

    ##INFERENCE FUNCTION: LOADS PRETRAINED MODEL AND VALIDATES##

    def inference(PATH_to_model):
        model_arch = models.resnet18(num_classes=2).to(device) #model architecture
        dict_model = torch.load(PATH_to_model,map_location='cpu') #load params from file
        model_arch.load_state_dict(dict_model['model_state_dict'])
        model_arch.eval() #set to validate mode
        summary(model_arch,input_size=(3,224,224))

        #CONFUSION MATRIX AND CLASS ACCURACY CALCULATIONS#
        nb_classes = 2

        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        id_corr = []
        id_incorr = []

        with torch.no_grad():
            for inputs, classes, paths in tqdm(dataloader,unit_scale=dataloader.batch_size,desc='Evaluating',dynamic_ncols=True):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model_arch(inputs)
                _, preds = torch.max(outputs, 1)
                path_to_pred = dict(zip(paths,preds))
                path_to_class = dict(zip(paths,classes))

                for path in paths:
                    if path_to_pred[path] == path_to_class[path]:
                        id_corr.append(path)
                    else:
                        id_incorr.append(path)

                for t, p in zip(classes.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

        print('Correct classifications:',len(id_corr))
        print('Incorrect classifications:',len(id_incorr))

        #with open('correct_id_imgs.txt', 'w') as f:
         #   for item in id_corr:
          #      f.write("%s\n" % item)
        #with open('incorrect_id_imgs.txt', 'w') as f:
         #   for item in id_incorr:
          #      f.write("%s\n" % item)
                
        #PRINT FORMATTED CONFUSION TABLE#
        #TODO: IF TABULATE IS NOT PRESENT, SIMPLY print(confusion_matrix)
        print()
        print('Confusion Matrix for Nval={}'.format(dataset_size))
        print(tabulate([['TN', confusion_matrix[0,0]], ['FP', confusion_matrix[0,1]], ['FN', confusion_matrix[1,0]],
                                 ['TP', confusion_matrix[1,1]]], headers=['Condition', '#images'], tablefmt='orgtbl'))
        print()

        #PRECISION AND F1 SCORES#
        precision = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
        recall = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
        F1 = 2 * (precision * recall) / (precision + recall)
        print('best precision: ', precision.item())
        print('best recall: ', recall.item())
        print('best F1 score: ', F1.item())

        print()

        #PRINT CLASS ACCURACY#
        class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
        for i in range(nb_classes):
            print('accuracy for {}: {:4f}'.format(class_names[i], class_acc[i].item()))

inference('./states/trial_2/VGG_Fullcolor_Norm.pt')
