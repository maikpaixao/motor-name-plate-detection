from plate_detection import PlateDetection
from text_segmentation import TextSegmentation
import utils.parser_img as parser_img
import natsort
from tqdm import tqdm
from pathlib import Path

def read_images(path):
    directories = list(path.iterdir())
    directories_sorted = []
    for fn in sorted([str(p) for p in directories]):
        directories_sorted.append(fn)
    directories_sorted = natsort.natsorted(directories, key=str)
    return directories_sorted


if __name__ == '__main__':
    image_dir = Path('sample/')
    image_dir = list(image_dir.iterdir())

    model_plate = PlateDetection()
    model_text = TextSegmentation()

    for count, img_path in enumerate(image_dir):
        print('In Progress', img_path)
        img = parser_img.img_read(str(img_path))
        plate = model_plate.detect_plate(img)
        text_mask = model_text.segment_text(plate)

        path_name = 'saved/' + str(img_path)[8:-4] + '.jpg'
        model_text.extract_text(plate, text_mask, path_name)

