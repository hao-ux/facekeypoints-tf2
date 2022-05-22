import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from utils.callback import LossHistory
from utils.loss import SmoothL1Loss
from tensorflow.keras.optimizers import Adam
from datasets import FaceKeyPointsDatasets
from net import nets
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# 使用gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ------------------------------------ #
# -----------参数设置------------------ # 
# batch_size -> 批次
# 主干网络 ->  'mobilenetv2',
# epochs -> 轮次
# model_path
# input_shape -> 输入图片大小
# lr -> 学习率

batch_size = 8
backbone = 'mobilenetv2'
epochs = 100
model_path = ''
input_shape = [224,224]
lr=1e-3

def adjust_lr(epoch, lr=lr):
    print("Seting to %s" % (lr))
    if epoch < 3:
        return lr
    else:
        return lr * 0.93


if __name__ == '__main__':
    df = pd.read_csv('./datasets/training_frames_keypoints.csv', encoding='utf-8')
    
    labels = df.values[:,1:]
    data_mean = labels.mean()
    data_std = labels.std()
    
    traindatasets = FaceKeyPointsDatasets('./datasets/training_frames_keypoints.csv', './datasets/training', batch_size=batch_size, mean=data_mean, std=data_std)

    inputs = Input(shape=(input_shape[0], input_shape[1], 1))
    model = Model(inputs=inputs, outputs=nets[backbone](inputs=inputs, classes=68*2))
    
    callback = [
            EarlyStopping(monitor='loss', patience=15, verbose=1),
            ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}.h5',monitor='loss',
                            save_weights_only=True, save_best_only=False, period=1),
            TensorBoard(log_dir='./logs1'),
            LossHistory('./logs1'),
            LearningRateScheduler(adjust_lr)
        ]
    
    if model_path != '':
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=SmoothL1Loss())

    history = model.fit(
            x                      = traindatasets,
            workers                = 1,
            epochs                 = epochs,
            callbacks              = callback,
            steps_per_epoch        = len(traindatasets),
            verbose=1
        )
    

