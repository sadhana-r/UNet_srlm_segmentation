# This scripts houses some usual config and constants used in the network
import os
import shutil
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="2"

class Config(object):
    def __init__(self):
        ## Directories and training image generation
        # Root directory that house the image
        self.root_dir = "/home/sadhana-ravikumar/Documents/Sadhana/unet3d_srlm"
        
        # Directory that contain CNN related files
        self.code = self.root_dir + "/scripts"
                
        # csv file that store train/test split
        self.train_val_csv = self.root_dir + "/data_csv/phg_split.csv"
        self.final_test_csv = self.root_dir + "/data_csv/phg_split_test.csv"
        #self.final_test_csv = "/home/sadhana-ravikumar/Documents/Sadhana/exvivo_MTL_histo/test_srlm.csv"
        
        
        self.patch_dir = self.root_dir + "/patch_data"
        self.train_patch_csv = self.root_dir + "/data_csv/train_patch.csv"
        self.val_patch_csv = self.root_dir + "/data_csv/val_patch.csv"
        
        # Directories that contain the model
        self.model_dir = self.root_dir + "/model"
        
        # Directories that contain the tensorboard output
        self.tfboard_dir = self.root_dir + "/tfboard"
        
        # Directories that store the validation output
        self.valout_dir = self.root_dir + "/validation_output"
        
        # Directories that store the validation output
        self.test_dir = self.root_dir + "/test_output"
                
        # disease to code
        self.diseaseCode = {"Control": 0, "aMCI": 1}
                
    def force_create(self, folder):
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)    

        
class Config_Unet(Config):
    def __init__(self):
        # Set base init
        super().__init__()
        
        # Number image per tfrecord in train and test set
#        self.nTrainPerTfrecord = 10
#        self.nTestPerTfrecord = 10
        
        # Training patch params
        self.num_pos = 150 #300 # Multires will generate 200 at each resolution
        self.num_neg = 125 #150 # try switching number of pos and neg
        self.aug = 5
        self.num_thread = 8
        
        # Multi resolution patch size and spacing setting
        self.patchsize_multi_res = [(1, (48, 48, 48)),(0.7,(48,48,48))]
        self.segsize = (96,96,96)
        self.half_patch = np.ceil((np.array(self.segsize) - 1) / 2).astype(np.int32)
        self.test_patch_spacing = (24, 24, 24) #
        self.patch_crop_size = 4
        
        ## Learning parameters
        self.batch_size = 3
        self.batch_step = 5 # Gradient accumulation every batch_step
        self.shuffle_buffer = 100
        self.learning_rate = 1e-3
        self.step_size = 10
        
        ## Training parameters 
        self.num_epochs = 30 #60 #100
        
        # Visualization params
        self.num_image_to_show = 3
        
        # num batch to save model
        self.batch_save_model = 8
        
        # num batch for validation
        self.batch_validation = 1
        
