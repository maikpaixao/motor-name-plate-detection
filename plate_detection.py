import config
import utils.free_memory as free_memory
import segmentation_models as sm
import utils.parser_img as parser_img
from utils.preprocessing import preprocessing_img

class PlateDetection():
    '''
    Class to perform plate detection.
    '''

    def __init__(self):

        # --- liberando memoria (precisa fazer isso nas RTX, CUDA 10+)
        free_memory.liberando_memoria_gpu()

        # --- load model
        self.model = sm.Unet(config.BACKBONE, classes=1, input_shape=(config.WIDTH, config.HEIGHT, config.INPUT_CHANNEL),
                        encoder_weights='imagenet', encoder_freeze=False)

        # load pre-trained weights
        self.model.load_weights(config.PLATE_MODEL)

    def detect_plate(self, img):
        '''
        Detect plate.
        :param img:     image
        :return:        plate image
        '''

        # calculate plate mask
        mask = self.model.predict(preprocessing_img(img))
        mask = mask[0]

        # re-scale img mask
        mask = parser_img.img_resize(mask, dim=(img.shape[1], img.shape[0]))

        # threshold
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask * 255

        # finding corners
        pts = parser_img.find_points(mask)

        # correct perspective
        plate = parser_img.perspective(img, pts)

        return plate