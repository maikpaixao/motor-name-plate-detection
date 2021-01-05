import config
import utils.free_memory as free_memory
import segmentation_models as sm
import glob
import os.path as P
from time import time
import utils.parser_img as parser_img
from utils.preprocessing import preprocessing_img
import numpy as np
import cv2
from tqdm import tqdm


def load_img_mask(path):
    '''
    Load the image and its label mask

    :param path:    path to img
    :return:        img, mask
    '''

    if config.INPUT_CHANNEL == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)

    mask = cv2.imread(path.replace('/image/', '/label/'), cv2.IMREAD_GRAYSCALE)
    mask = mask/255

    return img, mask




def test_model(model, path_imgs, theshold=0.9):
    '''
    generate statistics about an model

    :param model:           keras model
    :param path_imgs:       path to images. [path_1, path_2, ...]
    :param theshold:        Minimum iou value to consider the plate as found

    :return:                accuracy, iou
    '''

    iou = []
    accuracy = []
    inference_time = []

    # for each batch
    for i in tqdm(range(len(path_imgs))):
        img, gt = load_img_mask(path_imgs[i])

        # perform inference
        time_begin = time()
        mask = model.predict(preprocessing_img(img))
        final_time = time() - time_begin
        inference_time.append(final_time)

        mask = cv2.resize(mask[0], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        # threshold
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # calculating iou
        iou_ = float(sm.metrics.iou_score(gt.reshape(gt.shape[0], gt.shape[1], 1), mask.reshape(mask.shape[0], mask.shape[1], 1)))

        # metrics
        iou.append(iou_)
        if iou_ >= theshold:
            accuracy.append(1)
        else:
            accuracy.append(0)


        # finding corners
        pts = parser_img.find_points(mask*255)

        # projecting mask
        img_mask = img.copy().astype('float32')
        img_mask[:,:,1] += mask*255*0.3
        img_mask[img_mask > 255] = 255

        # projecting corners
        parser_img.draw_points(img_mask, pts)
        #'''
        # saving output img
        img_name = path_imgs[i].split('/')[-1].split('.')[0]
        parser_img.img_write(P.join(config.OUTPUT_IMAGES, img_name + '_iou_' + str(iou_) + '.png'), img_mask)

        # bad segmentation
        if iou_ < theshold:
            parser_img.img_write(P.join(config.OUTPUT_IMAGES, 'bad', img_name + '_iou_' + str(iou_) + '.png'), img_mask)
        #'''
    iou = np.asarray(iou)
    accuracy = np.asarray(accuracy)
    inference_time = np.mean(inference_time)

    print('Mean Accuracy: ' +str(accuracy.mean()))
    print('Max IoU: ' + str(iou.max()))
    print('Mean IoU: ' + str(iou.mean()))
    print('Min IoU: ' + str(iou.min()))
    print('Mean Time: ' + str(inference_time.mean()))


    return


# --- liberando memoria (precisa fazer isso nas RTX, CUDA 10+)
free_memory.liberando_memoria_gpu()

# -- getting dataset
test_fns = sorted(glob.glob(P.join(config.TEST_FOLDER, config.TYPE_OF_IMAGES)))

# --- load model
model = sm.Unet(config.BACKBONE, classes=1, input_shape=(config.WIDTH, config.HEIGHT, config.INPUT_CHANNEL), encoder_weights='imagenet', encoder_freeze=False)
model.load_weights(config.PLATE_MODEL) # load pre-trained weights


# --- test on train data
test_model(model, test_fns)
