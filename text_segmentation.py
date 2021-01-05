import config
import utils.free_memory as free_memory
import segmentation_models as sm
import utils.parser_img as parser_img
import utils.parser_ocr as parser_ocr
from utils.preprocessing import preprocessing_img
import cv2

class TextSegmentation():
    '''
    Class to perform text segmentation.
    '''

    def __init__(self):

        # --- liberando memoria (precisa fazer isso nas RTX, CUDA 10+)
        free_memory.liberando_memoria_gpu()

        # --- load model
        self.model = sm.Unet(config.BACKBONE, classes=1, input_shape=(config.WIDTH, config.HEIGHT, config.INPUT_CHANNEL),
                        encoder_weights='imagenet', encoder_freeze=False)

        # load pre-trained weights
        self.model.load_weights(config.TEXT_MODEL)

    def segment_text(self, img):
        '''
        Segment text.
        :param img:     image
        :return:        text mask
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

        return mask

    def extract_text(self, img, img_mask=[], path_name = ''):
        '''
        Extract text from image

        :param img:         image
        :param img_mask:    text segmentation mask image
        :return:            extracted text
        '''

        # if there is no mask image, create one
        if img_mask == []:
            img_mask = self.segment_text(img)

        #'''
        image = img.copy().astype('float32')
        image[:, :, 1] += img_mask.copy().astype('float32') * 0.3
        image[image > 255] = 255
        image = image.astype('uint8')
        #'''

        img_mask = img_mask.astype('uint8')
        # find contours in the mask
        countours = parser_img.find_countours(img_mask)

        # for each contour
        for c in countours:

            # crop image on the countour region
            cropped, top_left_corner = parser_img.crop(c, img)

            #cropped = parser_img.add_margin(cropped)

            # if img is vertical, rotate it
            if (cropped.shape[0] > cropped.shape[1] * 1.5):
                cropped = parser_img.img_rotate90(cropped)

            text = parser_ocr.extract_text(cropped)

            #print(text)
            #parser_img.img_show(cropped)
            parser_img.write_on_image(image, text=text, position=top_left_corner, font_size=60)
        parser_img.img_write(path_name, image)