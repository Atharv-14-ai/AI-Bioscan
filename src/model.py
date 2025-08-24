import tensorflow as tf
from tensorflow.keras import layers, models

def build_strong_cnn(input_shape, n_classes, l2=1e-4):
    inp = layers.Input(shape=input_shape)
    x = inp
    for f in (32, 64, 128, 128):
        x = layers.Conv2D(f, (3,3), padding='same', use_bias=False,
                          kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.35)(x)
    x = layers.Conv2D(256, (3,3), padding='same', use_bias=False, name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return models.Model(inp, out)
