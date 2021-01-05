import config
import utils.free_memory as free_memory
from utils.batches import generator_batch
from utils.callbacks import SaveImageCallback
import glob
import os.path as P
import segmentation_models as sm
from segmentation_models.utils import set_trainable
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback

# --- liberando memoria (precisa fazer isso nas RTX, CUDA 10+)
free_memory.liberando_memoria_gpu()

# -- getting dataset
train_fns = sorted(glob.glob(P.join(config.TRAIN_FOLDER, config.TYPE_OF_IMAGES)))
valid_fns = sorted(glob.glob(P.join(config.VALIDATION_FOLDER, config.TYPE_OF_IMAGES)))

train_gen = generator_batch(train_fns, bs=config.BATCH_SIZE)
valid_gen = generator_batch(valid_fns, bs=config.BATCH_SIZE, validation=True)

# --- define model
model = sm.Unet(config.BACKBONE, classes=1, input_shape=(config.WIDTH, config.HEIGHT, config.INPUT_CHANNEL),
                encoder_weights='imagenet', encoder_freeze=True)

if(config.LOAD_MODEL != ''):
    print('Loading pre-trained model...')
    model.load_weights(config.LOAD_MODEL) # load pre-trained weights

model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# --- Callbacks
checkpoint_model_best = ModelCheckpoint(config.SAVE_MODEL_PATH + '/%s.hdf5' % 'epoch_{epoch:05d}_best_model_val_loss_{val_loss:05f}',
                                            monitor='val_loss', save_best_only=True, verbose=1, mode='min')

check_before_epochs = ModelCheckpoint(config.SAVE_MODEL_PATH + '/model_epoch_{epoch:05d}_val_loss_{val_loss:05f}.hdf5',
                                          monitor='val_loss', period=10, verbose=1, mode='min')


save_net = SaveImageCallback(valid_gen)

# --- training
model.fit_generator(
    generator=train_gen, steps_per_epoch=config.TRAIN_STEPS,
    epochs=config.EPOCHS,
    verbose=1,
    validation_data=valid_gen, validation_steps=config.VALIDATION_STEPS,
    callbacks=[save_net, checkpoint_model_best, check_before_epochs],
    use_multiprocessing=True,
    workers=-1,
    initial_epoch=config.INITIAL_EPOCH
)