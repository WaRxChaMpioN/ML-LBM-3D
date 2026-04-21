"""
CNNModels.py - TF2/Keras compatible version
"""

import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np

xavier = tf.initializers.GlorotUniform


# ─────────────────────────────────────────────────────────────────────────────
# Custom activations as Keras Layers
# ─────────────────────────────────────────────────────────────────────────────

class ConcatELU(kl.Layer):
    def call(self, x):
        return tf.nn.elu(tf.concat([x, -x], axis=-1))

class ConcatReLU(kl.Layer):
    def call(self, x):
        return tf.nn.crelu(x, axis=-1)

class LReLU(kl.Layer):
    def call(self, x):
        return tf.maximum(x, 0.2 * x)


def set_nonlinearity(name):
    if name == 'concat_elu':
        return ConcatELU()
    elif name == 'elu':
        return kl.Activation('elu')
    elif name == 'concat_relu':
        return ConcatReLU()
    elif name == 'relu':
        return kl.Activation('relu')
    else:
        raise ValueError(f'nonlinearity {name} is not supported')


def apply_nonlinearity(x, nonlinearity):
    """Apply a nonlinearity layer to tensor x, creating a new instance if needed."""
    # Each call needs a fresh layer instance to avoid weight-sharing issues
    return nonlinearity.__class__()(x)


# ─────────────────────────────────────────────────────────────────────────────
# Core layer builders
# ─────────────────────────────────────────────────────────────────────────────

def conv_layer(inputs, kernel_size, stride, num_features, idx,
               nonlinearity=None, nDims=2):
    if nDims == 2:
        x = kl.Conv2D(num_features, kernel_size, strides=stride,
                      padding='same', use_bias=False,
                      kernel_initializer=xavier(),
                      name=f'{idx}_conv2d')(inputs)
    else:
        x = kl.Conv3D(num_features, kernel_size, strides=stride,
                      padding='same', use_bias=False,
                      kernel_initializer=xavier(),
                      name=f'{idx}_conv3d')(inputs)
    if nonlinearity is not None:
        x = apply_nonlinearity(x, nonlinearity)
    return x


def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx,
                         nonlinearity=None, nDims=2):
    if nDims == 2:
        x = kl.Conv2DTranspose(num_features, kernel_size, strides=stride,
                               padding='same', use_bias=False,
                               kernel_initializer=xavier(),
                               name=f'{idx}_convT2d')(inputs)
    else:
        x = kl.Conv3DTranspose(num_features, kernel_size, strides=stride,
                               padding='same', use_bias=False,
                               kernel_initializer=xavier(),
                               name=f'{idx}_convT3d')(inputs)
    if nonlinearity is not None:
        x = apply_nonlinearity(x, nonlinearity)
    return x


def nin(x, num_units, idx):
    """Network-in-network: 1x1 convolution implemented as Dense on last axis."""
    return kl.Dense(num_units, use_bias=False,
                    kernel_initializer=xavier(),
                    name=f'{idx}_nin')(x)


def denselayer(inputs, output_size, name=None):
    return kl.Dense(output_size, use_bias=False,
                    kernel_initializer=xavier(),
                    name=name)(inputs)


# ─────────────────────────────────────────────────────────────────────────────
# Gated residual block
# ─────────────────────────────────────────────────────────────────────────────

def res_block(x, a=None, filter_size=16, kernel_size=3,
              nonlinearity=None, keep_p=1.0,
              stride=1, gated=False, name="resnet", nDims=2):

    orig_x        = x
    in_filters    = x.shape[-1]

    # First conv: skip activation on first layer (input channels == 1)
    if in_filters == 1:
        x_1 = conv_layer(x, kernel_size, stride, filter_size,
                         name + '_conv_1', nDims=nDims)
    else:
        x_1 = apply_nonlinearity(x, nonlinearity)
        x_1 = conv_layer(x_1, kernel_size, stride, filter_size,
                         name + '_conv_1', nDims=nDims)

    # Skip connection from encoder
    if a is not None:
        a_act = apply_nonlinearity(a, nonlinearity)
        a_nin = nin(a_act, filter_size, name + '_nin')
        # pad spatial dims if needed
        x1_shape = x_1.shape[1:nDims+1]
        a_shape  = a_nin.shape[1:nDims+1]
        pad = [[0, 0]]
        for d in range(nDims):
            pad.append([0, int(x1_shape[d]) - int(a_shape[d])])
        pad.append([0, 0])
        if any(p[1] > 0 for p in pad):
            a_nin = kl.ZeroPadding2D(
                padding=((pad[1][1], 0), (pad[2][1], 0)),
                name=name + '_skip_pad'
            )(a_nin) if nDims == 2 else kl.ZeroPadding3D(
                padding=((pad[1][1], 0), (pad[2][1], 0), (pad[3][1], 0)),
                name=name + '_skip_pad'
            )(a_nin)
        x_1 = kl.Add(name=name + '_skip_add')([x_1, a_nin])

    x_1 = apply_nonlinearity(x_1, nonlinearity)

    if keep_p < 1.0:
        x_1 = kl.Dropout(rate=1.0 - keep_p)(x_1)

    # Second conv
    if not gated:
        x_2 = conv_layer(x_1, kernel_size, 1, filter_size,
                         name + '_conv_2', nDims=nDims)
    else:
        x_2 = conv_layer(x_1, kernel_size, 1, filter_size * 2,
                         name + '_conv_2', nDims=nDims)
        # gating: split channels, multiply sigmoid gate
        x_2 = GatedActivation(name=name + '_gate')(x_2)

    # Downsample skip if stride > 1
    if stride > 1:
        if nDims == 2:
            orig_x = kl.AveragePooling2D(pool_size=stride, strides=stride,
                                          padding='same',
                                          name=name + '_avgpool')(orig_x)
        else:
            orig_x = kl.AveragePooling3D(pool_size=stride, strides=stride,
                                          padding='same',
                                          name=name + '_avgpool')(orig_x)

    # Channel padding to match filter sizes
    out_f = filter_size
    in_f  = int(orig_x.shape[-1])

    if out_f > in_f:
        orig_x = ChannelPad(out_f - in_f, name=name + '_chanpad_skip')(orig_x)
    elif out_f < in_f:
        x_2 = ChannelPad(in_f - out_f, name=name + '_chanpad_x2')(x_2)

    return kl.Add(name=name + '_residual')([orig_x, x_2])


# ─────────────────────────────────────────────────────────────────────────────
# Custom Keras layers for operations that can't be plain tf.* calls
# ─────────────────────────────────────────────────────────────────────────────

class GatedActivation(kl.Layer):
    """Split channels in half: output = first_half * sigmoid(second_half)"""
    def call(self, x):
        half = x.shape[-1] // 2
        return x[..., :half] * tf.nn.sigmoid(x[..., half:])


class ChannelPad(kl.Layer):
    """Pad the channel dimension on the left with zeros."""
    def __init__(self, pad_size, **kwargs):
        super().__init__(**kwargs)
        self.pad_size = pad_size

    def call(self, x):
        ndim = len(x.shape) - 2   # spatial dims
        pad  = [[0, 0]] * (ndim + 1) + [[self.pad_size, 0]]
        return tf.pad(x, pad)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'pad_size': self.pad_size})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Generator (Gated U-Net)
# ─────────────────────────────────────────────────────────────────────────────

def gatedResnetGenerator(inputs, nr_res_blocks=1, keep_prob=1.0,
                         nonlinearity_name='concat_elu', gated=True,
                         filter_size=8, kernel_size=3,
                         nDims=2, outputType='vel'):

    nonlinearity = set_nonlinearity(nonlinearity_name)
    a = []
    x = inputs

    # ── Encoder ──────────────────────────────────────────────────────────────
    for i in range(nr_res_blocks):
        x = res_block(x, filter_size=filter_size, kernel_size=kernel_size,
                      nonlinearity=nonlinearity, keep_p=keep_prob,
                      gated=gated, name=f'resnet_1_{i}', nDims=nDims)
    a.append(x)

    filter_size *= 2
    x = res_block(x, filter_size=filter_size, kernel_size=kernel_size,
                  nonlinearity=nonlinearity, keep_p=keep_prob, stride=2,
                  gated=gated, name='resnet_2_downsample', nDims=nDims)
    for i in range(nr_res_blocks):
        x = res_block(x, filter_size=filter_size, kernel_size=kernel_size,
                      nonlinearity=nonlinearity, keep_p=keep_prob,
                      gated=gated, name=f'resnet_2_{i}', nDims=nDims)
    a.append(x)

    filter_size *= 2
    x = res_block(x, filter_size=filter_size, kernel_size=kernel_size,
                  nonlinearity=nonlinearity, keep_p=keep_prob, stride=2,
                  gated=gated, name='resnet_3_downsample', nDims=nDims)
    for i in range(nr_res_blocks):
        x = res_block(x, filter_size=filter_size, kernel_size=kernel_size,
                      nonlinearity=nonlinearity, keep_p=keep_prob,
                      gated=gated, name=f'resnet_3_{i}', nDims=nDims)
    a.append(x)

    filter_size *= 2
    x = res_block(x, filter_size=filter_size, kernel_size=kernel_size,
                  nonlinearity=nonlinearity, keep_p=keep_prob, stride=2,
                  gated=gated, name='resnet_4_downsample', nDims=nDims)
    for i in range(nr_res_blocks):
        x = res_block(x, filter_size=filter_size, kernel_size=kernel_size,
                      nonlinearity=nonlinearity, keep_p=keep_prob,
                      gated=gated, name=f'resnet_4_{i}', nDims=nDims)
    a.append(x)

    filter_size *= 2
    x = res_block(x, filter_size=filter_size, kernel_size=kernel_size,
                  nonlinearity=nonlinearity, keep_p=keep_prob, stride=2,
                  gated=gated, name='resnet_5_downsample', nDims=nDims)
    for i in range(nr_res_blocks):
        x = res_block(x, filter_size=filter_size, kernel_size=kernel_size,
                      nonlinearity=nonlinearity, keep_p=keep_prob,
                      gated=gated, name=f'resnet_5_{i}', nDims=nDims)

    # ── Decoder ───────────────────────────────────────────────────────────────
    filter_size //= 2
    x = transpose_conv_layer(x, kernel_size, 2, filter_size, 'up_conv_1', nDims=nDims)
    for i in range(nr_res_blocks):
        x = res_block(x, a=a[-1] if i == 0 else None,
                      filter_size=filter_size, kernel_size=kernel_size,
                      nonlinearity=nonlinearity, keep_p=keep_prob,
                      gated=gated, name=f'resnet_up_1_{i}', nDims=nDims)

    filter_size //= 2
    x = transpose_conv_layer(x, kernel_size, 2, filter_size, 'up_conv_2', nDims=nDims)
    for i in range(nr_res_blocks):
        x = res_block(x, a=a[-2] if i == 0 else None,
                      filter_size=filter_size, kernel_size=kernel_size,
                      nonlinearity=nonlinearity, keep_p=keep_prob,
                      gated=gated, name=f'resnet_up_2_{i}', nDims=nDims)

    filter_size //= 2
    x = transpose_conv_layer(x, kernel_size, 2, filter_size, 'up_conv_3', nDims=nDims)
    for i in range(nr_res_blocks):
        x = res_block(x, a=a[-3] if i == 0 else None,
                      filter_size=filter_size, kernel_size=kernel_size,
                      nonlinearity=nonlinearity, keep_p=keep_prob,
                      gated=gated, name=f'resnet_up_3_{i}', nDims=nDims)

    filter_size //= 2
    x = transpose_conv_layer(x, kernel_size, 2, filter_size, 'up_conv_4', nDims=nDims)
    for i in range(nr_res_blocks):
        x = res_block(x, a=a[-4] if i == 0 else None,
                      filter_size=filter_size, kernel_size=kernel_size,
                      nonlinearity=nonlinearity, keep_p=keep_prob,
                      gated=gated, name=f'resnet_up_4_{i}', nDims=nDims)

    # ── Output head ───────────────────────────────────────────────────────────
    if outputType == 'vel':
        x = conv_layer(x, kernel_size, 1, nDims,     'last_conv', nDims=nDims)
    elif outputType == 'fq':
        x = conv_layer(x, kernel_size, 1, 19,        'last_conv', nDims=nDims)
    elif outputType == 'velP':
        x = conv_layer(x, kernel_size, 1, nDims + 1, 'last_conv', nDims=nDims)
    elif outputType == 'P':
        x = conv_layer(x, kernel_size, 1, 1,         'last_conv', nDims=nDims)
    elif outputType == 'k':
        x = kl.Flatten(name='flatten_k')(x)
        x = denselayer(x, 16, name='dense_k_1')
        x = LReLU(name='lrelu_k')(x)
        x = denselayer(x, 1,  name='dense_k_2')

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────────────────────

def discriminatorTF(input_disc, kernel, filters, is_train=True,
                    reuse=False, nDims=2):

    def disc_block(x, n_filters, k_size, stride, tag):
        x = conv_layer(x, k_size, stride, n_filters, tag + '_conv', nDims=nDims)
        x = kl.BatchNormalization(momentum=0.9, epsilon=1e-3,
                                   scale=False, name=tag + '_bn')(x, training=is_train)
        x = LReLU(name=tag + '_lrelu')(x)
        return x

    x = input_disc
    x = conv_layer(x, kernel, 1, filters, 'disc_conv_1', nDims=nDims)
    x = LReLU(name='disc_lrelu_1')(x)

    x = disc_block(x, filters,     kernel, 2, 'disblock_1')
    x = disc_block(x, filters * 2, kernel, 1, 'disblock_2')
    x = disc_block(x, filters * 2, kernel, 2, 'disblock_3')
    x = disc_block(x, filters * 4, kernel, 1, 'disblock_4')
    x = disc_block(x, filters * 4, kernel, 2, 'disblock_5')
    x = disc_block(x, filters * 8, kernel, 1, 'disblock_6')
    x = disc_block(x, filters * 8, kernel, 2, 'disblock_7')

    x = kl.Flatten(name='disc_flatten')(x)
    x = denselayer(x, filters * 16, name='disc_dense_1')
    x = LReLU(name='disc_lrelu_final')(x)
    logits = denselayer(x, 1, name='disc_logits')
    out = kl.Activation('sigmoid', name='disc_sigmoid')(logits)

    return out, logits