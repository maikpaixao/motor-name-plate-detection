import segmentation_models as sm
import cv2
import config

def preprocessing_img(img):
    '''
    Pre-processing img to run on keras model.

    :param img:     opencv img
    :return:        img after preprocessing
    '''

    preprocess_input = sm.get_preprocessing(config.BACKBONE)

    _img = cv2.resize(img, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_CUBIC)
    _img = 255 - _img
    _img = _img.reshape((1, config.WIDTH, config.HEIGHT, config.INPUT_CHANNEL)).astype('float32')

    return preprocess_input(_img)