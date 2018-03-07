import torch
import h5py
from torch import nn
from scipy.misc import imresize
import torch
import cv2
from PIL import Image
import os
import logging
from multiprocessing import Pool
import numpy as np
import time
import random
import time
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file, phase):
        self.file = get_image_list(file, phase)
        pool = Pool()
        self.datas = pool.map(self.getitem, range(len(self.file)))
        pool.close()
        pool.join()
        self.phase = phase
        
    def __getitem__(self, id):
        t = time.time()
        random.seed(int(str(t%1)[2:7]))
        _, height, width = self.datas[id].shape
        if(self.phase == 'train'):
            h = random.randint(0, height - 40)
            w = random.randint(0, width - 40)
            return self.datas[id][:, h : h + 32, w : w + 32]
        elif(self.phase == 'val'):
            return self.datas[id]
        


        # real_id = int(id / 256)
        # logging.debug(self.file[real_id])
        # temp = self.datas[real_id]
        # x = int((id - real_id * 256) / 16)
        # y = id - real_id * 256 - x * 16
        # return temp[:, x : x + 32, y : y + 32]
    
    def __len__(self):
        return len(self.file)
        

    def getitem(self, id):
        print(self.file[id])   
        img = mpimg.imread(self.file[id])
        if(len(img.shape) == 2):
            timg = np.zeros([img.shape[0], img.shape[1], 3])
            timg[:, :, 0] = img  
            img = timg             
        return img.transpose([2,0,1]).astype('float32')*2-1

def get_image_list(train_dir, phase):
    image_list = []
    index = 0
    for dir in os.listdir(train_dir):
        if(phase == 'train' or (phase == 'val' and dir == 'schicka-307.png')):
            index += 1
            image_list.append(os.path.abspath(train_dir + dir))
        if(index > 1000):
            break
    return image_list
        
