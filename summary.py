from net.mobilenetv2 import MobilenetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from net import nets

# 查看网络结构

if __name__ == '__main__':
    inputs = Input((224, 244 ,1))
    backbone = 'mobilenetv2'
    classes = 68
    model = Model(inputs=inputs, outputs=nets[backbone](inputs=inputs, classes=classes*2))
    model.summary()