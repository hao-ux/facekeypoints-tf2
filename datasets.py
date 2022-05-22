import pandas as pd
import math
import tensorflow as tf
import matplotlib.image as mpimg
import os
import numpy as np


class FaceKeyPointsDatasets(tf.keras.utils.Sequence):
    def __init__(self, csv_file, datasets_path, batch_size, mean, std):
        self.df = pd.read_csv(csv_file, encoding='utf-8')
        self.datasets_pth = datasets_path
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        
    def __getitem__(self, index):
        image_list = []
        kpt_list = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % len(self.df)
            img_name = os.path.join(self.datasets_pth, self.df.iloc[i, 0])
            img = mpimg.imread(img_name)
            if img.shape[2] == 4:
                img =img[:,:,0:3]
            kpt = self.df.iloc[i, 1:].values
            kpt = kpt.astype('float').reshape(-1)
            img, kpt = self.Resize(img, kpt, 256)
            img, kpt = self.RandomCrop(img, kpt, 224)
            img, kpt = self.GrayNormalize(img, kpt, self.mean, self.std)
            img = np.array(img, dtype='float32')
            kpt = np.array(kpt, dtype='float32')
            image_list.append(img)
            kpt_list.append(kpt)
        
        return np.array(image_list), np.array(kpt_list)
            
            
            
    def Resize(self, img, kpt, output_size):
        assert isinstance(output_size, (int, tuple))
        
        img_copy = np.array(img)
        kpt_copy = np.array(kpt)
        
        h, w = img_copy.shape[:2]
        if isinstance(output_size, int):
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size
        new_h, new_w = int(new_h), int(new_w)
        img = tf.image.resize(img_copy, (new_h, new_w))
        # img = np.array(cv2.resize(img_copy, (new_h, new_w)), np.float32)
        kpt_copy[0::2] = kpt_copy[0::2] * new_w / w
        kpt_copy[1::2] = kpt_copy[1::2] * new_h / h
        return img, kpt_copy
    
    def RandomCrop(self, img, kpt, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            output_size = output_size
        
        image_copy = np.array(img)
        key_pts_copy = np.array(kpt)

        h, w = image_copy.shape[:2]
        new_h, new_w = output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image_copy = image_copy[top: top + new_h,
                      left: left + new_w]

        key_pts_copy[::2] = key_pts_copy[::2] - left
        key_pts_copy[1::2] = key_pts_copy[1::2] - top

        return image_copy, key_pts_copy
    
    
    def GrayNormalize(self, img, kpt, mean, std):
        img_copy = np.array(img)
        kpt_copy = np.array(kpt)
        
        # 灰度化
        img_copy = tf.image.rgb_to_grayscale(img_copy)

        img_copy = img_copy / 255.0
        
        # 坐标点缩放到-1，1
        kpt_copy = (kpt_copy - mean) / std
        
        
        return img_copy, kpt_copy
        
            
            
            
    def __len__(self):
        return math.ceil(len(self.df) / float(self.batch_size))