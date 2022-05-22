
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2D, DepthwiseConv2D, Dropout,ZeroPadding2D, Add, Dense,
    GlobalAveragePooling2D, Input, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def correct_pad(inputs, kernel_size):
    img_dim = 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)] # 224, 224
    

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size) # 1,1

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2) # 1, 1

    correct = (kernel_size[0] // 2, kernel_size[1] // 2) # 0,0

    return ((correct[0] - adjust[0], correct[0]), 
            (correct[1] - adjust[1], correct[1]))

# 保证特征层为8得倍数
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def relu6(x):
    return K.relu(x, max_value=6)

def Conv2D_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(
        filters=filters, kernel_size=kernel_size, padding='valid',
        use_bias=False, strides=strides
    )(inputs)
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.999)(x)
    x = Activation(relu6)(x)

    return x

def bottleneck(inputs, expansion, stride, alpha, filters):
    in_channels = K.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    # 数据扩充
    x = Conv2D(expansion * in_channels,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None)(x)
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.999)(x)
    x = Activation(relu6)(x)

    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3))(x)

    # 深度卷积
    x = DepthwiseConv2D(kernel_size=3,
                        strides=stride,
                        activation=None,
                        use_bias=False,
                        padding='same' if stride == 1 else 'valid')(x)
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.999)(x)

    x = Activation(relu6)(x)

    # 1x1卷积用于改变通道数
    x = Conv2D(pointwise_filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None)(x)

    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.999)(x)

    if (in_channels == pointwise_filters) and stride == 1:
        return Add()([inputs, x])
    return x


def MobilenetV2(inputs, alpha=1.0, dropout=1e-3, classes=17):

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D(padding=correct_pad(inputs, 3))(inputs)
    x = Conv2D_block(x, filters=first_block_filters, kernel_size=3, strides=(2, 2))

    x = bottleneck(x, filters=16, alpha=alpha, stride=1, expansion=1)

    x = bottleneck(x, filters=24, alpha=alpha, stride=2, expansion=6)
    x = bottleneck(x, filters=24, alpha=alpha, stride=1, expansion=6)

    x = bottleneck(x, filters=32, alpha=alpha, stride=2, expansion=6)
    x = bottleneck(x, filters=32, alpha=alpha, stride=1, expansion=6)
    x = bottleneck(x, filters=32, alpha=alpha, stride=1, expansion=6)
    x = bottleneck(x, filters=64, alpha=alpha, stride=2, expansion=6)
    x = bottleneck(x, filters=64, alpha=alpha, stride=1, expansion=6)
    x = bottleneck(x, filters=64, alpha=alpha, stride=1, expansion=6)
    x = bottleneck(x, filters=64, alpha=alpha, stride=1, expansion=6)

    x = bottleneck(x, filters=96, alpha=alpha, stride=1, expansion=6)
    x = bottleneck(x, filters=96, alpha=alpha, stride=1, expansion=6)
    x = bottleneck(x, filters=96, alpha=alpha, stride=1, expansion=6)
    x = bottleneck(x, filters=160, alpha=alpha, stride=2, expansion=6)
    x = bottleneck(x, filters=160, alpha=alpha, stride=1, expansion=6)
    x = bottleneck(x, filters=160, alpha=alpha, stride=1, expansion=6)
    
    x = bottleneck(x, filters=320, alpha=alpha, stride=1, expansion=6)

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2D_block(x, filters=last_block_filters, kernel_size=1, strides=(1, 1))

    x = GlobalAveragePooling2D()(x)    
    x = Dropout(dropout, name='dropout')(x)
    x = Dense(1000)(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dense(classes)(x)
    return x


if __name__ == '__main__':
    inputs = Input(shape=(224,224,1))
    classes = 136
    model = Model(inputs=inputs, outputs=MobilenetV2(inputs=inputs, classes=classes))

    model.summary()
    
    
