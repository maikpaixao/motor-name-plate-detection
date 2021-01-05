from keras.callbacks import Callback
import threading
import config
import cv2

class SaveImageCallback(Callback):
    def __init__(self, data_gen,stroke=False):
        super(SaveImageCallback, self).__init__()
        self.lock = threading.Lock()
        self.data_gen = data_gen

    def on_epoch_end(self, epoch, logs={}):
        self.lock = threading.Lock()
        with self.lock:
            data, gt = next(self.data_gen)
            mask = self.model.predict_on_batch(data)
            for i in range(mask.shape[0]):
                cv2.imwrite(config.OUTPUT_IMAGES + '/%d-%d-2_mask.png' % (epoch, i), mask[i, :, :, 0] * 255)
                cv2.imwrite(config.OUTPUT_IMAGES + '/%d-%d-1_gt.png' % (epoch, i), gt[i, :, :, 0] * 255)
                cv2.imwrite(config.OUTPUT_IMAGES + '/%d-%d-0_data.png' % (epoch, i), data[i, :, :, 0] * 255)
