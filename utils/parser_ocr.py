import pytesseract
from pytesseract import Output
import numpy as np

def extract_text_psm(img, psm=7):
    '''
    Extract text from image with specific psm

    :param img:     image
    :param psm:     psm mode
    :return:        text, mean conf
    '''
    try:
        lang = 'eng+por'
        config = '--psm '+str(psm)+' --oem 1'

        data = pytesseract.image_to_data(img, config=config, lang=lang, output_type=Output.DICT)
        conf = np.asarray(data['conf']).astype('int8')

        # confiance
        mean_conf = 0
        if len(conf[conf > 0]) > 1:
            mean_conf = np.mean(conf[conf > 0])
        elif len(conf[conf > 0]) == 1:
            mean_conf = conf[conf > 0][0]

        # text
        text_ = np.asarray(data['text'])[conf > 0]
        text = ''
        for t in text_:
            text += t + ' '

        text = text.replace('\x0c', '')

        return text, mean_conf
    except Exception as e:
        return '', 0

def extract_text(img):
    '''
    Extract text from a image. calculate best psm.

    :param img:     image
    :return:        text
    '''
    try:
        psm = [7, 11, 4]
        text = ''
        conf = -1
        for i in range(len(psm)):
            text_, conf_ = extract_text_psm(img, psm=psm[i])
            #print('conf:',conf_, 'psm:',psm[i], 'text:', text_)
            if conf_ > conf:
                conf = conf_
                text = text_

        return text
    except Exception as e:
        return ''