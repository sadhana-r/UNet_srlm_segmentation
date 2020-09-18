#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:05:15 2020

@author: sadhana-ravikumar
"""
import sys
sys.path.append('./utilities')

from unet_model import UNet
import numpy as np
import config_srlm as config
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch
import preprocess_data as p
import loss as l
import os.path as osp
import os
import nibabel as nib
import torch.nn as nn

def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Colorize:

    def __init__(self, n=4):
        
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def computeGeneralizedDSC_patch(probability, seg):
    
     seg = seg.cpu().numpy()
     probability = probability.cpu().numpy()
     preds = np.argmax(probability, 1)
     
     gt = seg.ravel()#[seg > 0]
     myseg = preds.ravel()#[seg > 0]
     
     gdsc = sum(gt == myseg)/ len(gt)
     
     return gdsc
 
def computeGeneralizedDSC(gt, seg):
    
     gt_seg = gt[gt > 0]
     myseg = seg[gt > 0]
     
     gdsc = 100*(sum(gt_seg == myseg)/ len(gt_seg))
     
     return gdsc
 
def generate_prediction(output):    
    """
    Generates predictions based on the output of the network
    """    
    #convert output to probabilities
    probability = F.softmax(output, dim = 1)
    _, preds_tensor = torch.max(probability, 1)
    
    return preds_tensor, probability
    
def plot_images_to_tfboard(img, seg, output, step, is_training = True):
    
    preds, probability = generate_prediction(output)
    
    if is_training:
        for i in range(1):
            writer.add_image('Training/Intensity images/'+str(i), img[i,:,:,:,24], global_step = step)
            writer.add_image('Training/Ground Truth seg/'+ str(i), color_transform(seg[i,None,:,:,24]), global_step = step)
            writer.add_image('Training/Predicted seg/'+ str(i), color_transform(preds[i,None,:,:,24]), global_step = step)
    else:
        for i in range(1):
            writer.add_image('Validation/Intensity images/'+str(i), img[i,:,:,:,24], global_step = step)
            writer.add_image('Validation/Ground Truth seg/'+ str(i), color_transform(seg[i,None,:,:,24]), global_step = step)
            writer.add_image('Validation/Predicted seg/'+ str(i), color_transform(preds[i,None,:,:,24]), global_step = step)
        
       
c = config.Config_Unet()
dir_names = config.Config()

# Set up GPU if available    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model or not. If load model, the modelDir and tfboardDir should have existed. Otherwise they
# will be created forcefully, wiping off the old one.
load_model = True

#Set up directories
root_dir = dir_names.root_dir
experiment_name = 'Experiment_21072020_dsc_patch96'
tfboard_dir = dir_names.tfboard_dir + '/' + experiment_name
model_dir = dir_names.model_dir + '/' + experiment_name + '/'
output_dir = dir_names.valout_dir + '/' + experiment_name + '/'
model_file = model_dir + 'model_20.pth'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
if not load_model:
    c.force_create(model_dir)
    c.force_create(tfboard_dir)

num_class = 3

# load unet model
if not load_model:
    net = UNet(num_class = num_class)
    net = net.to(device)
else:
    net = UNet(num_class = num_class)
    net.load_state_dict(torch.load(model_file, map_location = torch.device(device)))
    net = net.to(device)
    net.eval()

#Initialize class to convert labels to color images
color_transform = Colorize()
    
#Set up data
#Define image dataset (reads in full images and segmentations)
image_dataset = p.ImageDataset(csv_file = c.train_val_csv)

# Split dataset into train and validation
#train_ratio = 0.8
#num_train = math.ceil(train_ratio*len(image_dataset))
#num_val = len(image_dataset) - num_train
#train_dataset, val_dataset = torch.utils.data.random_split(image_dataset, [num_train, num_val])

# Set up tensor board
writer = SummaryWriter(tfboard_dir)
  
# Define a loss function and optimizer
weights = torch.ones(num_class)
#Give a larger weight to SRLM for SRLM
#weights[-1] = 5
weights = weights.to(device)
#ignore_index?
#criterion = nn.CrossEntropyLoss(weights)
criterion = l.GeneralizedDiceLoss(num_classes=num_class, weight = weights)
optimizer = optim.Adam(net.parameters(), lr = c.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = c.step_size, gamma = 0.1)

## Save and genearte patches for datasets
tp_dir = dir_names.patch_dir + "/training_data"
vp_dir = dir_names.patch_dir + "/validation_data"

#c.force_create(tp_dir)
#c.force_create(vp_dir)
#if os.path.exists(dir_names.train_patch_csv):
#    os.remove(dir_names.train_patch_csv)
#if os.path.exists(dir_names.val_patch_csv):
#    os.remove(dir_names.val_patch_csv)
#  
##Train/val split determined in the split_data.csv
#for i in range(len(image_dataset)):
#    sample = image_dataset[i]
#    print(sample['id'])
#    if(sample['type'] == 'train'):
#        print(sample['type'])
#        patches = p.GeneratePatches(sample, is_training = True, transform = True, multires =  False)
#    else:
#        print(sample['type'])
#        patches = p.GeneratePatches(sample, is_training = True, transform = False, multires =  False)
#        
###Now create a dataset with the patches
train_dataset = p.PatchDataset(csv_file = dir_names.train_patch_csv)
val_dataset = p.PatchDataset(csv_file = dir_names.val_patch_csv)

## DON't USE
## Concatenate all datasets
#print("Training Datastes...")
#print("Generating patches for dataset 1")
#
#sample = train_dataset[0]
#train_patches_all = p.PatchDataset(sample, is_training = True, transform = True)
#for i in range(1,len(train_dataset)):
#    
#    print("Generating patches for dataset ",i+1)
#    sample = train_dataset[i]
#    patches = p.PatchDataset(sample, is_training = True, transform = True)
#    train_patches_all = torch.utils.data.ConcatDataset([train_patches_all, patches])
#    
### Validation - test accuracy on subset of patches
#print("Validation datsets...")      
#print("Generating patches for dataset 1")
#sample = val_dataset[0]
#val_patches_all = p.PatchDataset(sample, is_training = True, transform = False)
#
#for i in range(1,len(val_dataset)):
#    
#    print("Generating patches for dataset ", i+1)
#    sample = val_dataset[i]
#    patches = p.PatchDataset(sample, is_training = True, transform = False)
#    val_patches_all = torch.utils.data.ConcatDataset([val_patches_all, patches])
## USE FROM HERE    

## Training loop
#training_loss = 0.0
#
#for epoch in range(c.num_epochs):    
#    
#    trainloader = DataLoader(train_dataset, batch_size = c.batch_size, shuffle = True, num_workers = c.num_thread)
#    net.train()
#     #Zero the parameter gradients
#    optimizer.zero_grad()
#    for j, patch_batched in enumerate(trainloader,0):
#        
#        img = patch_batched['image'][:,None,...].to(device)
#        seg = patch_batched['seg'].to(device)
#        
#        # forward + backward + optimize
#        output = net(img)        
#        
#        #Compute prediction
#        _, probability = generate_prediction(output)
#
#        #Compute dice loss        
#        loss = criterion(output, seg.long())
#        loss.backward()
#        
#        training_loss += loss.item()
#        
#        if (j+1)%c.batch_step == 0:
#            
#            # every "batch_step" iteration sof batches - accumulate gradients
#            optimizer.step()
#            optimizer.zero_grad()
#
#        if j % 5 == 4: #print every 5 batches
#            #Plot images
#            plot_images_to_tfboard(img, seg, output, epoch*len(trainloader) + j, is_training = True)            
#            print('Training loss: [epoch %d,  iter %5d] loss: %.3f lr: %.5f' %(epoch +1, j+1, training_loss/5, scheduler.get_lr()[0]))
#            writer.add_scalar('training_loss', training_loss/5, epoch*len(trainloader) + j)
#            training_loss = 0.0
#            
#    
#    ## Validation on randomly sampled patches   
#    validation_loss = 0
#    validation_dsc = 0
#    count = 0
#    with torch.no_grad():
#        
#        valloader = DataLoader(val_dataset, batch_size = c.batch_size, shuffle = True, num_workers = c.num_thread)
#        net.eval()        
#        
#        for j, patch_batched in enumerate(valloader):
#                           
#            img = patch_batched['image'][:,None,...].to(device)
#            #img = patch_batched['image'].permute(0,4,1,2,3).to(device) - with generatedeepmedic patches
#            seg = patch_batched['seg'].to(device)
#            #print(img.shape)
#
#            output = net(img)
#            
#            loss = criterion(output, seg.long())
#            
#            pred, probability = generate_prediction(output)
#            
#            gdsc = computeGeneralizedDSC_patch(probability, seg)
#            
#            validation_loss += loss.item()
#            validation_dsc += gdsc
#            
#            count += 1
#            
#            if j % 5 == 4: #print every 5 batches
#            #Plot images
#                plot_images_to_tfboard(img, seg, output, epoch*len(valloader) + j, is_training = False)            
#
#     
#        print('Validation loss: epoch %d loss: %.3f' %(epoch +1, validation_loss/count))
#        writer.add_scalar('validation_loss', validation_loss/count, epoch + 1)
#        writer.add_scalar('validation_accuracy', validation_dsc/count, epoch + 1)
#        
#        scheduler.step()
#        
#        #Save the model at the end of every epoch
#        model_file = model_dir + 'model_' + str(epoch + 1) + '.pth'
#        torch.save(net.state_dict(), model_file)
#        
## when predicting, I need to do softmax and argmax                
#print('Finished Training')
#
##Save the model
#torch.save(net.state_dict(), model_file)
#
#writer.close()

# Run network on validation set and save outputs
pad_size = c.half_patch[0]
gdsc_val = []

print(next(net.parameters()).is_cuda)
with torch.no_grad():
    for i in range(len(image_dataset)):
        print(i)
        sample = image_dataset[i]
        if(sample['type'] == 'test'):
        
            image_id = sample['id']
            print("Generating test patches for ", image_id )
            test_patches = p.GeneratePatches(sample, is_training = False, transform =False)
            
            testloader = DataLoader(test_patches, batch_size = c.batch_size, shuffle = False, num_workers = c.num_thread)    
    
            image_shape = sample['image'].shape
            affine = sample['affine']
            
            ## For assembling image
            im_shape_pad = [x + pad_size*2 for x in image_shape]
            prob = np.zeros([num_class] + list(im_shape_pad))
            rep = np.zeros([num_class] + list(im_shape_pad))
            
            pred_list = []
            for j, patch_batched in enumerate(testloader):
                
                    print("batch", j)                
                    img = patch_batched['image'][:,None,...].to(device)
                    seg = patch_batched['seg'].to(device)
                    cpts = patch_batched['cpt']
                    output = net(img)
                    probability = F.softmax(output, dim = 1).cpu().numpy()
                    
                    #Crop the patch to only use the center part
                    probability = probability[:,:,c.patch_crop_size:-c.patch_crop_size,
                                              c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size]
                                    
                    ## Assemble image in loop!
                    n, C, hp, wp, dp = probability.shape
                    half_shape = torch.tensor([hp, wp,dp])/2
    #                half_shape = half_shape.astype(int)
                    hs, ws, ds = half_shape
                    
                    for cpt, pred in zip(list(cpts), list(probability)):
                        #if np.sum(pred)/hs/ws/ds < 0.1:
                        prob[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += pred
                        rep[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += 1
                                    
    #                pred_list.append((probability, cpts))
                    
             #Crop the image since we added padding when generating patches
            prob = prob[:,pad_size:-pad_size, pad_size:-pad_size,pad_size:-pad_size]
            rep = rep[:,pad_size:-pad_size,pad_size:-pad_size,pad_size:-pad_size]
            rep[rep==0] = 1e-6
        
            # Normalized by repetition
            prob = prob/rep
        
            seg_pred = np.argmax(prob, axis = 0).astype('float')
            prob = np.moveaxis(prob,0,-1)
            
            gdsc = computeGeneralizedDSC(sample['seg'], seg_pred)
            print("Prediction accuracy", gdsc)
            gdsc_val.append(gdsc)
            
            nib.save(nib.Nifti1Image(prob, affine), osp.join(output_dir, "prob_" + str(image_id) + ".nii.gz"))
            nib.save(nib.Nifti1Image(seg_pred, affine), osp.join(output_dir, "seg_" + str(image_id) + ".nii.gz" ))
            
            print("Done!")
        
    print("Average validation accuracy is ", sum(gdsc_val)/len(gdsc_val))