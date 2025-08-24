import numpy as np, tensorflow as tf

def gradcam_spectrogram(model, x, class_index=None, layer_name="last_conv"):
    # x: (1, F, T, 1)
    if class_index is None:
        preds = model.predict(x, verbose=0)[0]
        class_index = int(np.argmax(preds))
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]  # (H', W', C)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap
