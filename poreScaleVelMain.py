'''
ML for predicting steady state flow fields either in vel form or fq form, in 2D and 3D
Ported to TensorFlow 2.x / Keras by removing all TF1 session-based APIs.
'''

from sys import stdout
import argparse
import numpy as np
import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from scipy import io
from CNNModels import *
from timeit import default_timer as timer
from skimage.metrics import peak_signal_noise_ratio as psnr
from tifffile import imwrite
import h5py

# ─────────────────────────────────────────────────────────────────────────────
# GPU setup
# ─────────────────────────────────────────────────────────────────────────────
def setup_gpu(gpuIDs):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuIDs
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ─────────────────────────────────────────────────────────────────────────────
# Model parameter summary
# ─────────────────────────────────────────────────────────────────────────────
def summarise_model(model_vars):
    gParams = 0
    for variable in model_vars:
        shape = variable.shape
        variable_parameters = np.prod(shape)
        print(f'{variable.name}  numParams: {variable_parameters}  shape: {shape}')
        gParams += variable_parameters
    print(f'Network Parameters: {gParams}')
    return gParams


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders  (unchanged logic, TF-agnostic numpy operations)
# ─────────────────────────────────────────────────────────────────────────────
def _hr_image_path(path, subset, IO, id):
    return os.path.join(path, f'{subset}_{IO}', f'{id:04}-vels.npy')

def _lr_image_path(path, subset, IO, id):
    return os.path.join(path, f'{subset}_{IO}', f'{id:04}-geom.npy')


def load3DDatasetToRAM(path, trainIDs, valIDs):
    print('Loading Datasets into RAM for 3D training')
    trainA = np.sort(glob.glob(path + '/train_inputs/*'))[0:np.max(trainIDs)]
    trainB = np.sort(glob.glob(path + '/train_outputs/*'))[0:np.max(trainIDs)]
    testA  = np.sort(glob.glob(path + '/validation_inputs/*'))[0:np.max(valIDs)]
    testB  = np.sort(glob.glob(path + '/validation_outputs/*'))[0:np.max(valIDs)]

    def _load_pairs(filesA, filesB):
        imgA, imgB = [], []
        for fA, fB in zip(filesA, filesB):
            stdout.write(f'\rLoading {fB}'); stdout.flush()
            a = np.load(fA).astype('float32')
            b = np.load(fB).astype('float32')
            if len(a.shape) == 3:
                a = np.expand_dims(a, 3)
            else:
                a = a  # geometry already (Nx,Ny,Nz,1), no transpose needed
            b = b  # velocity already (Nx,Ny,Nz,3), no transpose needed
            imgA.append(a[:, :, :, 0:1])
            imgB.append(b[:, :, :, 0:3])
        stdout.write('\n')
        return (np.vstack(np.expand_dims(imgA, 0)),
                np.vstack(np.expand_dims(imgB, 0)))

    tA, tB = _load_pairs(trainA, trainB)
    vA, vB = _load_pairs(testA,  testB)
    return tA, tB, vA, vB


def loadDataset(iterNum, batch_size, trainImageIDs, img_width, img_height,
                path, subset, numOutputs, numInputs, inputType, outputType):
    numBatches = np.floor(np.max(trainImageIDs) / batch_size)
    index      = int(np.mod(iterNum, numBatches))
    beg, end   = index * batch_size, (index + 1) * batch_size
    IDs        = trainImageIDs[int(beg):int(end)]

    input_batch  = np.zeros((len(IDs), img_width, img_height, numInputs),  dtype='float32')
    output_batch = np.zeros((len(IDs), img_width, img_height, numOutputs), dtype='float32')

    for i, id in enumerate(IDs):
        inputs  = np.load(_lr_image_path(path, subset, 'inputs',  id))
        outputs = np.load(_hr_image_path(path, subset, 'outputs', id))

        if inputs.shape[0] < img_width:
            inputs = np.transpose(inputs, (1, 2, 0))
        if inputType == 'P':
            inputs = inputs[:, :, -1]
        if inputType == 'bin':
            inputs = np.expand_dims(inputs, 2)
        inputs = inputs[:, :, 0:numInputs]
        input_batch[i] = inputs

        if outputs.shape[0] < img_width:
            outputs = np.transpose(outputs, (1, 2, 0))
        if outputType == 'P':
            outputs = outputs[:, :, -1]
        if numOutputs == 1:
            outputs = np.expand_dims(outputs, 2)
        outputs = outputs[:, :, 0:numOutputs]
        output_batch[i] = outputs

    return input_batch, output_batch


def loadDatasetReg(iterNum, batch_size, trainImageIDs, img_width, img_height,
                   path, subset, numOutputs, numInputs, inputType, outputVec):
    numBatches = np.floor(np.max(trainImageIDs) / batch_size)
    index      = int(np.mod(iterNum, numBatches))
    beg, end   = index * batch_size, (index + 1) * batch_size
    IDs        = trainImageIDs[int(beg):int(end)]

    input_batch  = np.zeros((len(IDs), img_width, img_height, numInputs), dtype='float32')
    output_batch = np.zeros((len(IDs),), dtype='float32')

    for i, id in enumerate(IDs):
        inputs = np.load(_lr_image_path(path, subset, 'inputs', id))
        if inputs.shape[0] < img_width:
            inputs = np.transpose(inputs, (1, 2, 0))
        if inputType == 'P':
            inputs = inputs[:, :, -1]
        if inputType == 'bin':
            inputs = np.expand_dims(inputs, 2)
        inputs = inputs[:, :, 0:numInputs]
        input_batch[i]  = inputs
        output_batch[i] = outputVec[id - 1]

    return input_batch, output_batch


# ─────────────────────────────────────────────────────────────────────────────
# LBM distribution-function momentum extraction
# ─────────────────────────────────────────────────────────────────────────────
def VoxelMomentum3D(labels):
    vx = (labels[...,1]-labels[...,2]+labels[...,7]-labels[...,8]
         +labels[...,9]-labels[...,10]+labels[...,11]-labels[...,12]
         +labels[...,13]-labels[...,14])
    vy = (labels[...,3]-labels[...,4]+labels[...,7]-labels[...,8]
         -labels[...,9]+labels[...,10]+labels[...,15]-labels[...,16]
         +labels[...,17]-labels[...,18])
    vz = (labels[...,5]-labels[...,6]+labels[...,11]-labels[...,12]
         -labels[...,13]+labels[...,14]+labels[...,15]-labels[...,16]
         -labels[...,17]+labels[...,18])
    return tf.stack([vx, vy, vz], axis=-1)

def VoxelMomentum2D(labels):
    vx = (labels[...,1]-labels[...,2]+labels[...,7]-labels[...,8]
         +labels[...,9]-labels[...,10]+labels[...,11]-labels[...,12]
         +labels[...,13]-labels[...,14])
    vy = (labels[...,3]-labels[...,4]+labels[...,7]-labels[...,8]
         -labels[...,9]+labels[...,10]+labels[...,15]-labels[...,16]
         +labels[...,17]-labels[...,18])
    vz = (labels[...,5]-labels[...,6]+labels[...,11]-labels[...,12]
         -labels[...,13]+labels[...,14]+labels[...,15]-labels[...,16]
         -labels[...,17]+labels[...,18])
    return tf.stack([vx, vy, vz], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────
def compute_mse_loss(predVelField, realVelField, gLoss, inputGeom, nDims, numOutputs):
    if gLoss == 'L1':
        return tf.reduce_sum(tf.abs(predVelField - realVelField))
    elif gLoss == 'L2':
        return tf.reduce_sum(tf.square(predVelField - realVelField))
    elif gLoss == 'L0.5':
        return tf.reduce_sum(tf.sqrt(tf.abs(predVelField - realVelField)))
    elif gLoss == 'L2Scaled':
        weights = inputGeom + 1.0
        return tf.reduce_sum(tf.square((predVelField - realVelField) * weights))
    elif gLoss == 'L1Scaled':
        weights = inputGeom
        return tf.reduce_sum(tf.abs((predVelField - realVelField) * weights))
    else:
        raise ValueError(f'Unknown loss type: {gLoss}')


def compute_conservation_loss(predVel, realVel, alpha, nDims):
    if alpha <= 0:
        return tf.constant(0.0)

    if nDims == 2:
        qXr  = tf.reduce_sum(realVel[:,:,:,0], axis=1)
        qXp  = tf.reduce_sum(predVel[:,:,:,0], axis=1)
        qYr  = tf.reduce_sum(realVel[:,:,:,1], axis=2)
        qYp  = tf.reduce_sum(predVel[:,:,:,1], axis=2)
        qXXr = tf.reduce_sum(realVel[:,:,:,0], axis=2)
        qXXp = tf.reduce_sum(predVel[:,:,:,0], axis=2)
        qYYr = tf.reduce_sum(realVel[:,:,:,1], axis=1)
        qYYp = tf.reduce_sum(predVel[:,:,:,1], axis=1)
        return alpha * (tf.reduce_sum(tf.square(qXr-qXp))
                      + tf.reduce_sum(tf.square(qYr-qYp))
                      + tf.reduce_sum(tf.square(qXXr-qXXp))
                      + tf.reduce_sum(tf.square(qYYr-qYYp)))
    elif nDims == 3:
        qXr   = tf.reduce_sum(realVel[:,:,:,:,0], axis=[2,3])
        qXp   = tf.reduce_sum(predVel[:,:,:,:,0], axis=[2,3])
        qYr   = tf.reduce_sum(realVel[:,:,:,:,1], axis=[1,3])
        qYp   = tf.reduce_sum(predVel[:,:,:,:,1], axis=[1,3])
        qZr   = tf.reduce_sum(realVel[:,:,:,:,2], axis=[1,2])
        qZp   = tf.reduce_sum(predVel[:,:,:,:,2], axis=[1,2])
        qXXr  = tf.reduce_sum(realVel[:,:,:,:,0], axis=[1,3])
        qXXp  = tf.reduce_sum(predVel[:,:,:,:,0], axis=[1,3])
        qYYr  = tf.reduce_sum(realVel[:,:,:,:,1], axis=[2,3])
        qYYp  = tf.reduce_sum(predVel[:,:,:,:,1], axis=[2,3])
        qZZr  = tf.reduce_sum(realVel[:,:,:,:,2], axis=[2,3])
        qZZp  = tf.reduce_sum(predVel[:,:,:,:,2], axis=[2,3])
        qXXXr = tf.reduce_sum(realVel[:,:,:,:,0], axis=[1,2])
        qXXXp = tf.reduce_sum(predVel[:,:,:,:,0], axis=[1,2])
        qYYYr = tf.reduce_sum(realVel[:,:,:,:,1], axis=[1,2])
        qYYYp = tf.reduce_sum(predVel[:,:,:,:,1], axis=[1,2])
        qZZZr = tf.reduce_sum(realVel[:,:,:,:,2], axis=[1,3])
        qZZZp = tf.reduce_sum(predVel[:,:,:,:,2], axis=[1,3])
        return alpha * (tf.reduce_sum(tf.square(qXr-qXp))
                      + tf.reduce_sum(tf.square(qYr-qYp))
                      + tf.reduce_sum(tf.square(qZr-qZp))
                      + tf.reduce_sum(tf.square(qXXr-qXXp))
                      + tf.reduce_sum(tf.square(qYYr-qYYp))
                      + tf.reduce_sum(tf.square(qZZr-qZZp))
                      + tf.reduce_sum(tf.square(qXXXr-qXXXp))
                      + tf.reduce_sum(tf.square(qYYYr-qYYYp))
                      + tf.reduce_sum(tf.square(qZZZr-qZZZp)))


# ─────────────────────────────────────────────────────────────────────────────
# Single train step  (GradientTape)
# ─────────────────────────────────────────────────────────────────────────────
@tf.function
def train_step_G(inputGeom, realVelField,
                 generator, g_optimizer,
                 gLoss, alpha, nDims, numOutputs, outputType,
                 delta, beta, gamma):

    with tf.GradientTape() as tape:
        predVelField = generator(inputGeom, training=True)

        mse_loss = compute_mse_loss(predVelField, realVelField,
                                     gLoss, inputGeom, nDims, numOutputs)

        # velocity extraction for fq output type
        if outputType in ('vel', 'velP', 'P', 'k'):
            realVel = realVelField
            predVel = predVelField
        elif outputType == 'fq':
            if nDims == 2:
                realVel = VoxelMomentum2D(realVelField)
                predVel = VoxelMomentum2D(predVelField)
            else:
                realVel = VoxelMomentum3D(realVelField)
                predVel = VoxelMomentum3D(predVelField)
            mse_loss = mse_loss + tf.reduce_sum(tf.square(predVel - realVel))
        else:
            realVel = realVelField
            predVel = predVelField

        cons_loss = compute_conservation_loss(predVel, realVel, alpha, nDims)

        g_loss = mse_loss + cons_loss

        if outputType == 'P':
            g_loss = g_loss + 1e-4 * tf.reduce_sum(
                tf.image.total_variation(predVelField))

    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return g_loss, mse_loss, cons_loss


@tf.function
def train_step_GAN(inputGeom, realVelField,
                   generator, discriminator,
                   g_optimizer, d_optimizer,
                   gLoss, alpha, nDims, numOutputs, outputType,
                   advRatio, delta, beta, gamma):

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        predVelField = generator(inputGeom, training=True)

        mse_loss = compute_mse_loss(predVelField, realVelField,
                                     gLoss, inputGeom, nDims, numOutputs)

        if outputType in ('vel', 'velP', 'P', 'k'):
            realVel = realVelField
            predVel = predVelField
        elif outputType == 'fq':
            if nDims == 2:
                realVel = VoxelMomentum2D(realVelField)
                predVel = VoxelMomentum2D(predVelField)
            else:
                realVel = VoxelMomentum3D(realVelField)
                predVel = VoxelMomentum3D(predVelField)
            mse_loss = mse_loss + tf.reduce_sum(tf.square(predVel - realVel))
        else:
            realVel = realVelField
            predVel = predVelField

        cons_loss = compute_conservation_loss(predVel, realVel, alpha, nDims)

        # Discriminator forward passes
        _, logits_real = discriminator(realVelField, training=True)
        _, logits_fake = discriminator(predVelField, training=True)

        # Discriminator loss
        d_loss1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_real, labels=tf.ones_like(logits_real)))
        d_loss2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_fake, labels=tf.zeros_like(logits_fake)))
        d_loss = d_loss1 + d_loss2

        # Generator adversarial loss
        adv_loss = advRatio * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_fake, labels=tf.ones_like(logits_fake)))

        g_loss = mse_loss + cons_loss + adv_loss

        if outputType == 'P':
            g_loss = g_loss + 1e-4 * tf.reduce_sum(
                tf.image.total_variation(predVelField))

    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    return g_loss, mse_loss, cons_loss, adv_loss, d_loss


@tf.function
def val_step(inputGeom, realVelField,
             generator, gLoss, alpha, nDims, numOutputs, outputType,
             delta, beta, gamma):
    predVelField = generator(inputGeom, training=False)
    mse_loss = compute_mse_loss(predVelField, realVelField,
                                 gLoss, inputGeom, nDims, numOutputs)
    if outputType in ('vel', 'velP', 'P', 'k'):
        realVel = realVelField
        predVel = predVelField
    else:
        realVel = realVelField
        predVel = predVelField
    cons_loss = compute_conservation_loss(predVel, realVel, alpha, nDims)
    g_loss = mse_loss + cons_loss
    return predVelField, g_loss


# ─────────────────────────────────────────────────────────────────────────────
# Keras model wrappers
# (wraps the functional graph builders into a tf.keras.Model)
# ─────────────────────────────────────────────────────────────────────────────
def build_generator(inputShape, nr_res_blocks, keep_prob, reluType,
                    gatedResFlag, numFilters, baseKernelSize, nDims, outputType):
    inputs = tf.keras.Input(shape=inputShape[1:], name='geom_input')
    outputs = gatedResnetGenerator(
        inputs,
        nr_res_blocks     = nr_res_blocks,
        keep_prob         = keep_prob,
        nonlinearity_name = reluType,
        gated             = gatedResFlag,
        filter_size       = numFilters,
        kernel_size       = baseKernelSize,
        nDims             = nDims,
        outputType        = outputType
    )
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='Generator')


def build_discriminator(inputShape, kernel, filters, nDims):
    inputs = tf.keras.Input(shape=inputShape[1:], name='disc_input')
    out, logits = discriminatorTF(
        input_disc = inputs,
        kernel     = kernel,
        filters    = filters,
        is_train   = True,
        nDims      = nDims
    )
    return tf.keras.Model(inputs=inputs, outputs=[out, logits],
                          name='Discriminator')


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing  (identical to original)
# ─────────────────────────────────────────────────────────────────────────────
def int_range(s):
    try:
        fr, to = s.split('-')
        return range(int(fr), int(to) + 1)
    except Exception:
        raise argparse.ArgumentTypeError(f'invalid integer range: {s}')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):  return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def str2int(v):
    if v == 'M': return v
    try: return int(v)
    except: raise argparse.ArgumentTypeError('int value expected.')

def str2float(v):
    if v == 'M': return v
    try: return float(v)
    except: raise argparse.ArgumentTypeError('float value expected.')


parser = argparse.ArgumentParser(description='PSMLS')
parser.add_argument('--train',           type=str2bool,  default=False)
parser.add_argument('--test',            type=str2bool,  default=False)
parser.add_argument('--gpuIDs',          type=str,       default='0')
parser.add_argument('--nDims',           type=str2int,   default=2)
parser.add_argument('--residual-blocks', default=1)
parser.add_argument('--numFilters',      default=64)
parser.add_argument('--baseKernelSize',  default=3)
parser.add_argument('--gan',             type=str2bool,  default=False)
parser.add_argument('--advRatio',        type=str2float, default=1e-3)
parser.add_argument('--gLoss',           type=str,       default='L1')
parser.add_argument('--alpha',           type=str2float, default=0)
parser.add_argument('--beta',            type=str2float, default=1)
parser.add_argument('--gamma',           type=str2float, default=0)
parser.add_argument('--delta',           type=str2float, default=1)
parser.add_argument('--batch-size',      type=str2int,   default=16)
parser.add_argument('--keep-prob',       type=str2int,   default=0.7)
parser.add_argument('--reluType',        type=str,       default='concat_relu')
parser.add_argument('--gatedResFlag',    type=str2bool,  default=True)
parser.add_argument('--width',           type=str2int,   default=256)
parser.add_argument('--height',          type=str2int,   default=256)
parser.add_argument('--depth',           type=str2int,   default=1)
parser.add_argument('--num-epochs',      type=str2int,   default=500)
parser.add_argument('--epoch-step',      type=str2int,   default=50)
parser.add_argument('--learnRate',       type=str2float, default=1e-4)
parser.add_argument('--outputType',      type=str,       default='vel')
parser.add_argument('--inputType',       type=str,       default='bin')
parser.add_argument('--dataset',         type=str,       default='./velMLdistDataset_BIN')
parser.add_argument('--trainIDs',        type=int_range, default='1-8000')
parser.add_argument('--valIDs',          type=int_range, default='8001-9000')
parser.add_argument('--restore',         default=None)
parser.add_argument('--restoreD',        default=None)
parser.add_argument('--contEpoch',       default=0)
parser.add_argument('--valPlot',         type=str2bool,  default=True)
parser.add_argument('--testInputs',      type=str,       default='./test')
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Unpack args
# ─────────────────────────────────────────────────────────────────────────────
trainFlag       = args.train
testFlag        = args.test
gpuIDs          = args.gpuIDs
gLoss           = args.gLoss
alpha           = args.alpha
beta            = args.beta
gamma           = args.gamma
delta           = args.delta
valPlotFlag     = args.valPlot
numFilters      = int(args.numFilters)
residual_blocks = int(args.residual_blocks)
baseKernelSize  = int(args.baseKernelSize)
keep_prob       = args.keep_prob
reluType        = args.reluType
gatedResFlag    = args.gatedResFlag
outputType      = args.outputType
inputType       = args.inputType
nDims           = args.nDims
restore         = args.restore
restoreD        = args.restoreD
path            = args.dataset
trainImageIDs   = args.trainIDs
valImageIDs     = args.valIDs
img_width       = args.width
img_height      = args.height
img_depth       = args.depth
batch_size      = args.batch_size
epochs          = args.num_epochs
learnRate       = args.learnRate
ganFlag         = args.gan
advRatio        = args.advRatio

iterations_train  = int(np.ceil(np.max(trainImageIDs) / batch_size))
numValIterations  = int(np.ceil((np.max(valImageIDs) - np.min(valImageIDs)) / batch_size))

setup_gpu(gpuIDs)

# ─────────────────────────────────────────────────────────────────────────────
# I/O shapes
# ─────────────────────────────────────────────────────────────────────────────
numOutputs = {'vel': nDims, 'fq': 19, 'velP': nDims+1, 'P': 1, 'k': 0}.get(outputType, nDims)
numInputs  = {'bin': 1, 'velP': nDims+1, 'vel': nDims, 'P': 1}.get(inputType, 1)

if nDims == 2:
    outputShape = [batch_size, img_width, img_height, numOutputs]
    inputShape  = [batch_size, img_width, img_height, numInputs]
elif nDims == 3:
    outputShape = [batch_size, img_width, img_height, img_depth, numOutputs]
    inputShape  = [batch_size, img_width, img_height, img_depth, numInputs]

if outputType == 'k':
    outputShape = [batch_size]
    outputVec   = np.load('./regKTrains.npy')

# ─────────────────────────────────────────────────────────────────────────────
# Build models
# ─────────────────────────────────────────────────────────────────────────────
generator = build_generator(
    inputShape, residual_blocks, keep_prob, reluType,
    gatedResFlag, numFilters, baseKernelSize, nDims, outputType)

summarise_model(generator.trainable_variables)

# Learning-rate schedule: halves every epoch_step epochs (matches original)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = learnRate,
    decay_steps           = args.epoch_step * iterations_train,
    decay_rate            = 0.5,
    staircase             = True
)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

if ganFlag and trainFlag:
    discriminator = build_discriminator(outputShape, kernel=3, filters=32, nDims=nDims)
    d_optimizer   = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
else:
    discriminator = None
    d_optimizer   = None

# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────
ckpt = tf.train.Checkpoint(generator=generator, g_optimizer=g_optimizer)
if restore is not None:
    ckpt.restore(restore).expect_partial()
    print(f'Generator restored from {restore}')
    # Force new LR after restore — checkpoint restores old optimizer+schedule
    # Rebuild schedule fresh with the --learnRate arg value
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = learnRate,
        decay_steps           = args.epoch_step * iterations_train,
        decay_rate            = 0.5,
        staircase             = True)
    g_optimizer.learning_rate = lr_schedule
    print(f'LR schedule reset: initial_lr={learnRate} (overrides checkpoint LR)')
    # Force LR after restore — checkpoint saves old optimizer+schedule state
    # Rebuild lr_schedule with new learnRate and assign fresh
    try:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learnRate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True)
        g_optimizer.learning_rate.assign(learnRate)
        print(f'LR forced to {learnRate} after restore')
    except Exception as e:
        try:
            g_optimizer.learning_rate.assign(learnRate)
            print(f'LR assigned to {learnRate} (schedule rebuild failed: {e})')
        except Exception as e2:
            print(f'LR override failed: {e2}')
    # Force LR after restore — checkpoint saves old LR, override with arg
    try:
        g_optimizer.learning_rate.assign(learnRate)
        print(f'LR forced to {learnRate} (overriding checkpoint LR)')
    except Exception as e:
        print(f'LR override failed: {e}')

if ganFlag and discriminator is not None:
    ckpt_d = tf.train.Checkpoint(discriminator=discriminator, d_optimizer=d_optimizer)
    if restoreD is not None:
        ckpt_d.restore(restoreD).expect_partial()
        print(f'Discriminator restored from {restoreD}')

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
ganStr = '-gan' if (ganFlag and trainFlag) else ''
name   = path.split('_')[0].split('/')[-1]

if trainFlag:
    if nDims == 3:
        (trainInputsDataset, trainOutputsDataset,
         valInputsDataset,   valOutputsDataset) = load3DDatasetToRAM(
             path, trainImageIDs, valImageIDs)

    rightNow     = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folderName   = (f'{rightNow}-velCNN{ganStr}-{name}-{residual_blocks}-'
                    f'{baseKernelSize}-{numFilters}-{gLoss}-'
                    f'{alpha}-{beta}-{gamma}-{delta}')
    trainingDir  = os.path.join('./outputs', folderName)
    trainOutDir  = os.path.join('./trainingOutputs', folderName)
    os.makedirs(trainingDir, exist_ok=True)
    os.makedirs(trainOutDir, exist_ok=True)

    ckpt_manager  = tf.train.CheckpointManager(ckpt, trainingDir, max_to_keep=10000)
    if ganFlag:
        ckpt_manager_d = tf.train.CheckpointManager(ckpt_d, trainingDir, max_to_keep=10000)

    oldMeanValMSE = 1e10
    epochsMSE     = np.zeros(epochs)

    for epochNum in range(epochs):
        start       = timer()
        trainingMSE = np.zeros(iterations_train)

        for iterNum in range(iterations_train):
            # ── load batch ───────────────────────────────────────────────────
            if nDims == 2:
                if outputType == 'k':
                    inputsTrain, outputsTrain = loadDatasetReg(
                        iterNum, batch_size, trainImageIDs,
                        img_width, img_height, path, 'train',
                        numOutputs, numInputs, inputType, outputVec)
                else:
                    inputsTrain, outputsTrain = loadDataset(
                        iterNum, batch_size, trainImageIDs,
                        img_width, img_height, path, 'train',
                        numOutputs, numInputs, inputType, outputType)
            elif nDims == 3:
                nb  = int(np.floor(np.max(trainImageIDs) / batch_size))
                idx = int(np.mod(iterNum, nb))
                inputsTrain  = trainInputsDataset [idx*batch_size:(idx+1)*batch_size]
                outputsTrain = trainOutputsDataset[idx*batch_size:(idx+1)*batch_size]

            # ── scale outputs ────────────────────────────────────────────────
            if outputType == 'velP':
                outputsTrain[..., -1] = (outputsTrain[..., -1] + gamma) / beta
            elif outputType == 'vel':
                outputsTrain[..., 0:nDims] *= delta

            # ── gradient step ─────────────────────────────────────────────────
            if not ganFlag:
                errg, errMSE, errCons = train_step_G(
                    tf.constant(inputsTrain), tf.constant(outputsTrain),
                    generator, g_optimizer,
                    gLoss, alpha, nDims, numOutputs, outputType,
                    delta, beta, gamma)
                errAdv = errd = errdR = errdF = 0.0
            else:
                errg, errMSE, errCons, errAdv, errd = train_step_GAN(
                    tf.constant(inputsTrain), tf.constant(outputsTrain),
                    generator, discriminator,
                    g_optimizer, d_optimizer,
                    gLoss, alpha, nDims, numOutputs, outputType,
                    advRatio, delta, beta, gamma)
                errdR = errdF = 0.0

            lr_now = g_optimizer.learning_rate(g_optimizer.iterations).numpy() \
                     if hasattr(g_optimizer.learning_rate, '__call__') \
                     else float(g_optimizer.learning_rate)

            stdout.write(
                f'\rLR: {lr_now:.4e} '
                f'Epoch [{epochNum+1:4d}/{epochs:4d}] '
                f'[{iterNum+1:4d}/{iterations_train:4d}]: '
                f'g_loss: {float(errg):.4f} '
                f'(mse: {float(errMSE):.4f} '
                f'cons: {float(errCons):.4f} '
                f'adv: {float(errAdv):.4f}) '
                f'd_loss: {float(errd):.4f}')
            stdout.flush()
            trainingMSE[iterNum] = float(errg)

        stdout.write('\n')
        print(f'Mean GLoss: {np.mean(trainingMSE):.4f}')
        epochsMSE[epochNum] = np.mean(trainingMSE)

        # ── Validation every 10 epochs ────────────────────────────────────────
        if (epochNum + 1) % 10 == 0 or epochNum == 0:
            epoch_dir = os.path.join(trainOutDir, f'epoch-{epochNum+1}')
            os.makedirs(epoch_dir, exist_ok=True)
            valMSE = np.zeros(numValIterations)

            for n in range(numValIterations):
                if nDims == 2:
                    if outputType == 'k':
                        inputsVal, outputsVal = loadDatasetReg(
                            n, batch_size, valImageIDs,
                            img_width, img_height, path, 'validation',
                            numOutputs, numInputs, inputType, outputVec)
                    else:
                        inputsVal, outputsVal = loadDataset(
                            n, batch_size, valImageIDs,
                            img_width, img_height, path, 'validation',
                            numOutputs, numInputs, inputType, outputType)
                elif nDims == 3:
                    nb  = int(np.floor(np.max(valImageIDs) / batch_size))
                    idx = int(np.mod(n, nb))
                    inputsVal  = valInputsDataset [idx*batch_size:(idx+1)*batch_size]
                    outputsVal = valOutputsDataset[idx*batch_size:(idx+1)*batch_size]

                if outputType == 'velP':
                    outputsVal[..., -1] = (outputsVal[..., -1] + gamma) / beta
                elif outputType == 'vel':
                    outputsVal[..., 0:nDims] *= delta

                predictsVal, errvg = val_step(
                    tf.constant(inputsVal), tf.constant(outputsVal),
                    generator, gLoss, alpha, nDims, numOutputs, outputType,
                    delta, beta, gamma)
                predictsVal = predictsVal.numpy()
                valMSE[n]   = float(errvg)

                # unscale for saving
                if outputType == 'velP':
                    predictsVal[..., -1] = predictsVal[..., -1] * beta - gamma
                    outputsVal[..., -1]  = outputsVal[..., -1]  * beta - gamma
                elif outputType == 'vel':
                    predictsVal[..., 0:nDims] /= delta
                    outputsVal[..., 0:nDims]  /= delta

                # save .mat files
                if valPlotFlag and ((epochNum + 1) % 100 == 0 or epochNum == 0):
                    for i in range(batch_size):
                        if outputType == 'vel' and nDims == 2:
                            dPred = {'solid': inputsVal[i,:,:,0],
                                     'velX':  predictsVal[i,:,:,0],
                                     'velY':  predictsVal[i,:,:,1]}
                            dReal = {'solid': inputsVal[i,:,:,0],
                                     'velX':  outputsVal[i,:,:,0],
                                     'velY':  outputsVal[i,:,:,1]}
                        elif outputType == 'vel' and nDims == 3:
                            dPred = {'solid': inputsVal[i,:,:,:,0],
                                     'velX':  predictsVal[i,:,:,:,0],
                                     'velY':  predictsVal[i,:,:,:,1],
                                     'velZ':  predictsVal[i,:,:,:,2]}
                            dReal = {'solid': inputsVal[i,:,:,:,0],
                                     'velX':  outputsVal[i,:,:,:,0],
                                     'velY':  outputsVal[i,:,:,:,1],
                                     'velZ':  outputsVal[i,:,:,:,2]}
                        elif outputType == 'velP' and nDims == 2:
                            dPred = {'solid':   inputsVal[i,:,:,0],
                                     'velX':    predictsVal[i,:,:,0],
                                     'velY':    predictsVal[i,:,:,1],
                                     'density': predictsVal[i,:,:,2]}
                            dReal = {'solid':   inputsVal[i,:,:,0],
                                     'velX':    outputsVal[i,:,:,0],
                                     'velY':    outputsVal[i,:,:,1],
                                     'density': outputsVal[i,:,:,2]}
                        elif outputType == 'P' and nDims == 2:
                            dPred = {'solid': inputsVal[i,:,:,0],
                                     'P':     predictsVal[i,:,:,0]}
                            dReal = {'solid': inputsVal[i,:,:,0],
                                     'P':     outputsVal[i,:,:,0]}
                        elif outputType == 'k':
                            dPred = {'perm': predictsVal[i]}
                            dReal = {'perm': outputsVal[i]}
                        else:
                            dPred = {'pred': predictsVal[i]}
                            dReal = {'real': outputsVal[i]}

                        io.savemat(f'{epoch_dir}/{n+1:04}-{i}-pred.mat', dPred)
                        io.savemat(f'{epoch_dir}/{n+1:04}-{i}-real.mat', dReal)

                stdout.write(f'\rValidation [{n+1:4d}/{numValIterations:4d}] '
                             f'MSE: {float(errvg):.4f}')
                stdout.flush()

            stdout.write('\n')
            end = timer()
            meanValMSE = np.mean(valMSE)
            print(f'Mean Val Loss: {meanValMSE:.4f}  Epoch Time: {end-start:.2f}s')

            if meanValMSE < oldMeanValMSE:
                save_path = ckpt_manager.save()
                print(f'  Checkpoint saved → {save_path}')
                if ganFlag:
                    ckpt_manager_d.save()
                oldMeanValMSE = meanValMSE


# ─────────────────────────────────────────────────────────────────────────────
# TESTING
# ─────────────────────────────────────────────────────────────────────────────
if testFlag:
    sample_files  = glob.glob(args.testInputs + '/*.mat')
    sample_files += glob.glob(args.testInputs + '/*.npy')
    sample_files  = sorted(sample_files)
    restore_tag  = restore.split('/')[-1].split('.')[0] if restore else 'notRestored'
    outdir       = os.path.join(args.testInputs, f'CNNOutputs-{restore_tag}-{name}')
    os.makedirs(outdir, exist_ok=True)

    for sampleFile in sample_files:
        fileName = os.path.splitext(os.path.basename(sampleFile))[0]
        print(f'Estimating velocity field for: {sampleFile}')

        if sampleFile.endswith('.npy'):
            img = np.load(sampleFile).astype('float32')
            # Shape: (Nx,Ny,Nz,1) or (Nx,Ny,Nz) — ensure channel dim exists
            if img.ndim == nDims:
                img = np.expand_dims(img, -1)
            # img is now (Nx,Ny,Nz,1) — add batch dim
            img = np.expand_dims(img, 0)
            img = np.repeat(img, batch_size, 0)
        else:
            arrays = {}
            try:
                with h5py.File(sampleFile, 'r') as f:
                    for k, v in f.items():
                        arrays[k] = np.array(v)
                img = np.array(arrays['temp'], dtype='float32')
            except Exception:
                mat = scipy.io.loadmat(sampleFile)
                key = [k for k in mat.keys() if not k.startswith('_')][0]
                img = mat[key].astype('float32')
            img = np.expand_dims(img, nDims)
            img = np.expand_dims(img, 0)
            img = np.repeat(img, batch_size, 0)

        pred = generator(tf.constant(img), training=False).numpy()
        pred = np.squeeze(pred[0])            # take first item, remove batch dim

        if outputType == 'velP':
            pred[..., nDims] = pred[..., nDims] * beta - gamma
        elif outputType == 'vel':
            pred[..., 0:nDims] /= delta

        if nDims == 2:
            if outputType == 'vel':
                dPred = {'solid': img[0,:,:,0], 'velX': pred[:,:,0], 'velY': pred[:,:,1]}
            elif outputType == 'velP':
                dPred = {'solid': img[0,:,:,0], 'velX': pred[:,:,0],
                         'velY': pred[:,:,1], 'density': pred[:,:,2]}
            elif outputType == 'P':
                dPred = {'solid': img[0,:,:,0], 'density': pred}
        elif nDims == 3:
            if outputType == 'vel':
                dPred = {'solid': img[0,:,:,:,0], 'velX': pred[:,:,:,0],
                         'velY': pred[:,:,:,1], 'velZ': pred[:,:,:,2]}
            elif outputType == 'velP':
                dPred = {'solid': img[0,:,:,:,0], 'velX': pred[:,:,:,0],
                         'velY': pred[:,:,:,1], 'velZ': pred[:,:,:,2],
                         'density': pred[:,:,:,3]}
            elif outputType == 'P':
                dPred = {'solid': img[0,:,:,:,0], 'density': pred}

        io.savemat(f'{outdir}/{fileName}-pred.mat', dPred)
        print(f'  Saved → {outdir}/{fileName}-pred.mat')