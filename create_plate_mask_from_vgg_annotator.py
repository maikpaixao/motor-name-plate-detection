import json
import utils.parser_img as parser_img
import numpy as np

def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
        return data

project_file = 'dataset/test_corrigidas/test_corrigidas.json'
imgs_dir = 'dataset/test_corrigidas/image/'
imgs_mask_dir = 'dataset/test_corrigidas/label/'

project_vgg_annotator = read_json(project_file)
metadata = project_vgg_annotator['_via_img_metadata']

for i in metadata:
    img_file = metadata[i]
    regions = img_file['regions']
    if len(regions) == 4:
        img = parser_img.img_read(imgs_dir + img_file['filename'])
        img_mask = np.zeros(img.shape)

        coordinates = []
        for j in range(len(regions)):
            coordinates.append([regions[j]['shape_attributes']['cx'], regions[j]['shape_attributes']['cy']])

        parser_img.draw_polylines(img_mask, coordinates)
        parser_img.img_write(imgs_mask_dir + img_file['filename'],img_mask)