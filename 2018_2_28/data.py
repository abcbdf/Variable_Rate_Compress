import torch
import h5py
from torch import nn
from scipy.misc import imresize
import torch
import cv2
import os
import logging

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file, phase):
        self.file = get_image_list(file)
        
    def __getitem__(self, id):
        real_id = int(id / 256)
        logging.debug(self.file[real_id])
        try:
            temp = imresize(cv2.imread(self.file[real_id]),(512,512)).transpose([2,0,1]).astype('float32')/255. *2-1
        except:
            logging.error("reading picture error:" + self.file[real_id])
        x = int((id - real_id * 256) / 16)
        y = id - real_id * 256 - x * 16
        return temp[:, x : x + 32, y : y + 32]
    
    def __len__(self):
        return len(self.file) * 256

def get_image_list(train_dir):
    image_list = []
    index = 0
    for dir in os.listdir(train_dir):
        index += 1
        image_list.append(os.path.abspath(train_dir + dir))
    return image_list
        
