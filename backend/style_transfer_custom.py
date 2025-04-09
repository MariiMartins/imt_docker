import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model

content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1', 'block2_conv1',
    'block3_conv1', 'block4_conv1',
    'block5_conv1'
]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def load_vgg19_model(weights_path):
    print("Carregando VGG19 com pesos locais...")
    vgg = vgg19.VGG19(include_top=False, weights=None)
    vgg.load_weights(weights_path)
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return Model(inputs=vgg.input, outputs=model_outputs)

def gram_matrix(x):
    result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    input_shape = tf.shape(x)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def get_feature_representations(model, content_image, style_image):
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    style_features = [gram_matrix(layer) for layer in style_outputs[:num_style_layers]]
    content_features = [layer for layer in content_outputs[num_style_layers:]]
    return style_features, content_features

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    for target, comb in zip(gram_style_features, style_output_features):
        style_score += tf.reduce_mean(tf.square(gram_matrix(comb) - target))
    for target, comb in zip(content_features, content_output_features):
        content_score += tf.reduce_mean(tf.square(comb - target))

    style_score *= style_weight
    content_score *= content_weight
    return style_score + content_score

@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        loss = compute_loss(**cfg)
    return tape.gradient(loss, cfg['init_image']), loss

def apply_style_transfer(model, content_image, style_image, alpha=1.0, iterations=50):
    print(f"🎨 Aplicando transferência customizada com alpha={alpha:.2f}...")
    content_tensor = tf.convert_to_tensor(content_image)
    style_tensor = tf.convert_to_tensor(style_image)

    content_tensor = vgg19.preprocess_input(content_tensor)
    style_tensor = vgg19.preprocess_input(style_tensor)

    style_features, content_features = get_feature_representations(model, content_tensor, style_tensor)
    init_image = tf.Variable(content_tensor, dtype=tf.float32)

    opt = tf.optimizers.Adam(learning_rate=5.0)

    content_weight = 1e3 * (1 - alpha)
    style_weight = 1e-2 * alpha

    cfg = {
        'model': model,
        'loss_weights': (style_weight, content_weight),
        'init_image': init_image,
        'gram_style_features': style_features,
        'content_features': content_features
    }

    best_loss, best_img = float('inf'), None

    for i in range(iterations):
        grads, loss = compute_grads(cfg)
        opt.apply_gradients([(grads, init_image)])
        init_image.assign(tf.clip_by_value(init_image, 0.0, 255.0))

        if loss < best_loss:
            best_loss = loss
            best_img = init_image.numpy()

    # Interpolação direta com alpha para melhor controle de intensidade
    best_img = (alpha * best_img + (1 - alpha) * content_image).astype(np.uint8)

    return best_img
