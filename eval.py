import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from facekeypointsnet import FaceKeyPointsNet
from tqdm import tqdm

# --------------------------------------------- #

# 对测试集进行评估

def nme(y_true, y_pred):
    
    y_true = np.array(y_true, np.float32).reshape((-1, 2))
    y_pred = np.array(y_pred, np.float32).reshape((-1, 2))

    interocular = np.linalg.norm(y_true[36, ] - y_true[45, ])
    rmse = np.sum(np.linalg.norm(y_pred - y_true, axis=1)) / (interocular * y_pred.shape[0])
    return rmse
        

if __name__ == '__main__':
    face_kpt = FaceKeyPointsNet()
    data_file = './datasets/test_frames_keypoints.csv'
    data_img = './datasets/test'
    
    df = pd.read_csv(data_file, encoding='utf-8')
    s = 0

    for i in tqdm(range(len(df)), desc='进行中', ncols=100):
        img_name = os.path.join(data_img, df.iloc[i, 0])
        kpt = np.array(df.iloc[i, 1:].values)
        img = mpimg.imread(img_name)
        out = face_kpt.eval(img)
        out = np.squeeze(out)
        s += nme(kpt, out)
        
    print("NME:{:.3f}".format(s/len(df)))

        
    
    
