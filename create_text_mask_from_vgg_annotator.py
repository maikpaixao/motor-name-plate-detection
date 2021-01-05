import json
import utils.parser_img as parser_img
import numpy as np

def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
        return data

project_file = 'dataset/text/test/text_test.json'
imgs_dir = 'dataset/text/test/image/'
imgs_mask_dir = 'dataset/text/test/label/'

project_vgg_annotator = read_json(project_file)
metadata = project_vgg_annotator['_via_img_metadata']

for i in metadata:
    img_file = metadata[i]
    regions = img_file['regions']

    img = parser_img.img_read(imgs_dir + img_file['filename'])
    img_mask = np.zeros(img.shape)


    for j in range(len(regions)):
        coordinates = []
        for k in range(len(regions[j]['shape_attributes']['all_points_y'])):
            coordinates.append([regions[j]['shape_attributes']['all_points_x'][k], regions[j]['shape_attributes']['all_points_y'][k]])

        parser_img.draw_polylines(img_mask, coordinates)
    parser_img.img_write(imgs_mask_dir + img_file['filename'],img_mask)