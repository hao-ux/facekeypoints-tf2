
import tensorflow as tf
import numpy as np
from net import nets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import pandas as pd
from utils.utils import decode_show
import time

class FaceKeyPointsNet(object):
    _defaults = {
        "model_path": "./model_data/mobilenetv2.h5", # 权重路径
        "input_shape": [224, 224],  # 输入图片大小
        "backbone": 'mobilenetv2', # 网络结构
    }
    
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
        
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.df = pd.read_csv('./datasets/training_frames_keypoints.csv')
        key_pts_values = self.df.values[:,1:]
        self.data_mean = key_pts_values.mean() # 计算均值
        self.data_std = key_pts_values.std()   # 计算标准差
        
        # 使用gpu
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.generate()
        print("导入模型成功！！！")
        
    def generate(self):
        inputs = Input(shape=(self.input_shape[0], self.input_shape[1], 1))
        self.model = Model(inputs=inputs, outputs=nets[self.backbone](inputs=inputs, classes=68*2))
        self.model.load_weights(self.model_path)
        
        
    def detect_image(self, img):
        img = np.array(img)
        if img.shape[2] == 4:
            img =img[:,:,:3]
        rgb_img = self.Resize(img, 256)
        rgb_img = self.RandomCrop(rgb_img, 224)
        img = self.GrayNormalize(rgb_img)
        img = np.array([img], dtype='float32')
        out =self.model.predict(img)
        out = np.expand_dims(np.array(out),2)
        decode_show(rgb_img, out, self.data_mean, self.data_std)
        
    def eval(self, img):
        img = np.array(img)
        if img.shape[2] == 4:
            img =img[:,:,:3]
        rgb_img = self.Resize(img, 256)
        rgb_img = self.RandomCrop(rgb_img, 224)
        img = self.GrayNormalize(rgb_img)
        img = np.array([img], dtype='float32')
        out =self.model.predict(img)
        out = np.expand_dims(np.array(out),2)
        return out
    
    def fps(self, img, n=100):
        start = time.time()
        img = np.array(img)
        if img.shape[2] == 4:
            img =img[:,:,:3]
        rgb_img = self.Resize(img, 256)
        rgb_img = self.RandomCrop(rgb_img, 224)
        img = self.GrayNormalize(rgb_img)
        img = np.array([img], dtype='float32')
        for _ in range(n):
            out =self.model.predict(img)
        end = time.time()
        avg_time = (end - start)/n
        return avg_time
        
        
    def Resize(self, img, output_size):
        assert isinstance(output_size, (int, tuple))
        
        img_copy = np.array(img)
        
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
        
        return img
    
    def RandomCrop(self, img, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            output_size = output_size
        
        image_copy = np.array(img)

        h, w = image_copy.shape[:2]
        new_h, new_w = output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image_copy = image_copy[top: top + new_h,
                      left: left + new_w]


        return image_copy
    
    def GrayNormalize(self, img):
        img_copy = np.array(img)
        # 灰度化
        img_copy = tf.image.rgb_to_grayscale(img_copy)

        img_copy = img_copy / 255.0
        
        
        return img_copy
        