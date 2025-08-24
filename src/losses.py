import tensorflow as tf

def class_balanced_focal_loss(alpha, gamma=2.0, eps=1e-8):
    alpha_tf = tf.constant(alpha, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        ce = -tf.math.log(pt)
        a = tf.reduce_sum(y_true * alpha_tf, axis=-1)
        fl = a * tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(fl)
    return loss
