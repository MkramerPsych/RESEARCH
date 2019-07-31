################################################################################
#                      RESNET18 FOR BINARY CLASSIFICATION                      #
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
from tensorboardX import SummaryWriter
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import copy
if __name__ == '__main__': #PREVENT BROKEN PIPE ERROR, ALLOW FOR MULTIPROCESSING

################################################################################
#                              DATA PREPROCESSING                              #
################################################################################
    def colorspace_convert(img):
        img = img.permute(1,2,0)
        img = color.rgb2lab(img)
        #img = color.lab2rgb(img)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)
        img = img.to(dtype=torch.float32)
        return img

    def gaussian_blur(img):
        img = img.filter(ImageFilter.GaussianBlur(radius=4.5))
        return img

    data_transforms = { #LIST OF TRANSFORMS SPLIT BY DATASET

        'train': transforms.Compose([ #TRAINING TRANSFORMS
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
           # transforms.Lambda(lambda img: gaussian_blur(img)), #GAUSSIAN BLUR TRANSFORM (ON PIL IMAGE)
            transforms.ToTensor(),
           # transforms.Lambda(lambda img: colorspace_convert(img)), #LAB COLORSPACE CONVERSION (ON TORCH TENSOR)
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #COMMENT OUT WHEN USING LAB COLORSPACE
        ]),

        'val': transforms.Compose([ #VALIDATION TRANSFORMS
            transforms.Resize((224,224)),
           # transforms.Lambda(lambda img: gaussian_blur(img)), #GAUSSIAN BLUR TRANSFORM (ON PIL IMAGE)
            transforms.ToTensor(),
           # transforms.Lambda(lambda img: colorspace_convert(img)), #LAB COLORSPACE CONVERSION (ON TORCH TENSOR)
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    data_dir = 'data/lfsex' #PATH TO DATA FROM CURRENT DIRECTORY

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']} #CREATE DATASETS FROM FOLDERS

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,shuffle=True, num_workers=4) for x in ['train', 'val']} #CREATE LOADERS FROM DATASETS

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} #GET NUM OF IMAGES IN EACH DATASET

    class_names = image_datasets['train'].classes #PULL CLASS NAMES FROM DATA

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #MOVE DATA TO APPROPRIATE PROCESSING
    print(torch.cuda.device_count(), 'available GPUs')
    print('Model Running on: {}'.format(device))

    # #DATA VISUALIZATION#
   # img, label = image_datasets['train'][9000]
   # a = img.squeeze(0)
   # print(a.shape)
   # a = a.numpy().transpose((1,2,0))
   # print(a.shape)
   # a = np.clip(a,0,2)
   # plt.imshow(a)
   # plt.title(class_names[label])
   # plt.show()
    #END DATA VISUALZATION#

################################################################################
#                              TRAINING FUNCTION                               #
################################################################################

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25): #25 EPOCH DEFAULT
        since = time.time() #TRACK TIME

        best_model_wts = copy.deepcopy(model.state_dict()) #SAVE BEST PARAMETERS
        best_acc = 0.0 #SAVE BEST VALIDATION ACCURACY
       # writer = SummaryWriter()

        for epoch in range(num_epochs):

            #FORMAT READOUT#
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            #CONTROL PHASE#
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step() #DECAY LEARNING RATE
                    model.train() #TRAIN
                else:
                    model.eval() #VALIDATION

                running_loss = 0.0
                running_corrects = 0

                #TRAINING LOOP#
                for inputs, labels in tqdm(dataloaders[phase], unit_scale=dataloaders[phase].batch_size, desc='Epoch {}, {} phase'.format(epoch + 1, phase)):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad() #CLEAR GRADIENT BUFFERS

                    #FORWARD PROP#
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        #BACKPROP#
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    #LOSS & ACCURACY TRACKING#
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                #NORMALIZE LOSS & ACCURACY#
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f}, {} Accuracy: {:.4f}'.format(phase, epoch_loss, phase, epoch_acc))

                #TENSORBOARDX LOGGING#
               # if phase == 'train':
                #    writer.add_scalar('loss/training_loss',epoch_loss,epoch)
                 #   writer.add_scalar('acc/training_acc',epoch_acc,epoch)
               # if phase == 'val':
                #    writer.add_scalar('loss/validation_loss',epoch_loss,epoch)
                 #   writer.add_scalar('acc/validation_acc',epoch_acc,epoch)

                #SAVE PARAMETERS & ACCURACY#
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                #SAVE STATE_DICT TO DISK#
                if epoch % 5 == 0: #EVERY 5 EPOCHS
                    torch.save({
                        'epoch' : epoch,
                        'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'loss' : epoch_loss,
                        },'tester.pt')

            print()

        #RETURN SUMMARY OF TRAINING#
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        #LOAD TRAINED PARAMETERS AND RETURN MODEL#
        model.load_state_dict(best_model_wts)
        return model

################################################################################
#                                MAIN FUNCTION                                 #
################################################################################
    def main(): #RUN MODEL WITH DEFINED LOSS, OPTIMIZER, AND LR SCHEDULER

        #NETWORK ARCHITECTURE#
        model_ft = models.resnet18(num_classes=2).to(device)
        if torch.cuda.device_count() > 1:
            print('utilizing', torch.cuda.device_count(), 'GPUs')
            model_ft = nn.DataParallel(model_ft)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model_ft.parameters())
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # #TRAIN MODEL WITH GIVEN ARCHITECTURE#
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=20)

        #CONFUSION MATRIX AND CLASS ACCURACY CALCULATIONS#
       # nb_classes = 2

       # confusion_matrix = torch.zeros(nb_classes, nb_classes)

       # with torch.no_grad():
           # for i, (inputs, classes) in enumerate(dataloaders['val']):
           #     inputs = inputs.to(device)
            #    classes = classes.to(device)
           #     outputs = model_ft(inputs)
           #     _, preds = torch.max(outputs, 1)
             #   for t, p in zip(classes.view(-1), preds.view(-1)):
            #            confusion_matrix[t.long(), p.long()] += 1

        #PRINT FORMATTED CONFUSION TABLE#
        #TODO: IF TABULATE IS NOT PRESENT, SIMPLY print(confusion_matrix)
     #   print()
     #   print('Confusion Matrix for Nval={}'.format(dataset_sizes['val']))
     #  print(tabulate([['TN', confusion_matrix[0,0]], ['FP', confusion_matrix[0,1]], ['FN', confusion_matrix[1,0]],
                           #      ['TP', confusion_matrix[1,1]]], headers=['Condition', '#images'], tablefmt='orgtbl'))
      #  print()

        #PRECISION AND F1 SCORES#
      #  precision = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
      #  recall = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
      #  F1 = 2 * (precision * recall) / (precision + recall)
      #  print('best precision: ', precision.item())
      #  print('best recall: ', recall.item())
      #  print('best F1 score: ', F1.item())

        #PRINT CLASS ACCURACY#
       # class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
       # for i in range(nb_classes):
        #    print('accuracy for {}: {:4f}'.format(class_names[i], class_acc[i].item()))

    main()
