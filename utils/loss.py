

import tensorflow as tf

def SmoothL1Loss(delta=1.0):
    def _SmoothL1Loss(y_true, y_pred):
        loss = tf.keras.losses.huber(y_true, y_pred, delta=delta)
        loss = tf.reduce_mean(loss)
        return loss
    return _SmoothL1Loss