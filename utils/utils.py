import cv2
import numpy as np
import matplotlib.pylab as plt

def show_all_keypoints(image, predicted_key_pts):

    plt.imshow(image)
    # # 展示关键点
    for i in range(0, len(predicted_key_pts), 2):
        plt.scatter(predicted_key_pts[i], predicted_key_pts[i+1], s=20, marker='.', c='m')
    plt.show()


def decode_show(test_imgs,test_outputs,maen,std, batch_size=1, h=10,w=10):
    if len(test_imgs.shape) == 3:
        test_images = np.array([test_imgs])
        
    for i in range(batch_size):
        plt.figure(figsize=(h, w))
        ax = plt.subplot(1, batch_size, i+1)

        # 随机裁剪后的图像
        image = test_images[i]

        
        # 模型的输出，未还原的预测关键点坐标值
        predicted_key_pts = test_outputs[i]

        # 还原后的真实的关键点坐标值


        # 展示图像和关键点
        return show_all_keypoints(image, predicted_key_pts)
            


