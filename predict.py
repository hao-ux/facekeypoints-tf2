from facekeypointsnet import FaceKeyPointsNet
import matplotlib.image as mpimg
import cv2
import time
import numpy as np
# 预测图片

facekeypointnet = FaceKeyPointsNet()
# ----------------------- #
# mode=1 测试单张图片预测时间
# mode=0 测试单张图片效果展示

mode = 0
# 测试100次，仅在mode=1有效
n = 100

if __name__ == '__main__':

    if mode == 0:
        while True:
            img = input('Input image filename:')
            try:
                image = mpimg.imread(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                facekeypointnet.detect_image(image)
    elif mode == 1:
        image = mpimg.imread('./img/face.png')
        t = facekeypointnet.fps(image, n=100)
        print("FPS:{}, n: {}, time: {}".format(1/t, n, t))
            
    