import numpy as np
import cv2
import config
import segmentation_models as sm

def generator_batch(fns, bs, validation=False):
    preprocess_input = sm.get_preprocessing(config.BACKBONE)
    batches = []
    for i in range(0, len(fns), bs):
        batches.append(fns[i: i + bs])

    print("Batching {} batches of size {} each for {} total files".format(len(batches), bs, len(fns)))
    while True:
        for fns in batches:
            imgs_batch = []
            masks_batch = []
            bounding_batch = []
            for fn in fns:
                if config.INPUT_CHANNEL == 1:
                    _img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
                else:
                    _img = cv2.imread(fn)
                if _img is None:
                    print(fn)
                    continue

                _img = cv2.resize(_img, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_CUBIC)
                #_img = _img.astype('float32')


                if config.INPUT_CHANNEL == 1:
                    if validation:
                        mask = cv2.imread(fn.replace(config.VALIDATION_FOLDER, config.VALIDATION_MASK_FOLDER), cv2.IMREAD_GRAYSCALE)
                    else:
                        mask = cv2.imread(fn.replace(config.TRAIN_FOLDER, config.TRAIN_MASK_FOLDER), cv2.IMREAD_GRAYSCALE)
                else:
                    if validation:
                        mask = cv2.imread(fn.replace(config.VALIDATION_FOLDER, config.VALIDATION_MASK_FOLDER))
                    else:
                        mask = cv2.imread(fn.replace(config.TRAIN_FOLDER, config.TRAIN_MASK_FOLDER))
                if mask is None:
                    print(fn)
                    continue

                mask = cv2.resize(mask, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_CUBIC)
                _img = 255 - _img
                #_img = 1 - (_img.reshape((config.WIDTH, config.HEIGHT, config.INPUT_CHANNEL)) / 255)
                #mask = mask.reshape((config.WIDTH, config.HEIGHT, config.INPUT_CHANNEL)) / 255
                mask = mask / 255
                mask = mask > 0.3

                mask = mask.astype('float32')
                imgs_batch.append(_img)
                masks_batch.append(mask)

            imgs_batch = np.asarray(imgs_batch).reshape((bs, config.WIDTH, config.HEIGHT, config.INPUT_CHANNEL)).astype('float32')
            masks_batch = np.asarray(masks_batch).reshape((bs, config.WIDTH, config.HEIGHT, config.INPUT_CHANNEL)).astype('float32')

            # preprocess input
            imgs_batch = preprocess_input(imgs_batch)
            #masks_batch = preprocess_input(masks_batch)

            yield imgs_batch, masks_batch