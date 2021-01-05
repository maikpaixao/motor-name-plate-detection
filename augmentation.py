from pathlib import Path
import imageio
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import natsort
from tqdm import tqdm


def combine_functions(function, l, name=None):
    f_list = l.copy()
    f_list = [f[0] for f in f_list]
    f_list.append(function[0])
    combination = iaa.Sequential(f_list)
    return combination


def write_augmentated_points(file, points):
    file.write('4' + ',' +
               str(int(points[0].x)) + ',' +
               str(int(points[0].y)) + ',' +
               str(int(points[1].x)) + ',' +
               str(int(points[1].y)) + ',' +
               str(int(points[2].x)) + ',' +
               str(int(points[2].y)) + ',' +
               str(int(points[3].x)) + ',' +
               str(int(points[3].y)) + ',' + ',' + '\n')
    file.close()


def get_keypoints(filename, image):
    file = open(str(filename), "r")

    points = file.readline()
    str1 = ''.join(points)
    points = str1[1:-1].split(',')

    kps = KeypointsOnImage([
        Keypoint(x=int(points[1]), y=int(points[2])),
        Keypoint(x=int(points[3]), y=int(points[4])),
        Keypoint(x=int(points[5]), y=int(points[6])),
        Keypoint(x=int(points[7]), y=int(points[8]))
    ], shape=image.shape)
    return kps


def read_images(path):
    directories = list(path.iterdir())
    directories_sorted = []
    for fn in sorted([str(p) for p in directories]):
        directories_sorted.append(fn)
    directories_sorted = natsort.natsorted(directories, key=str)
    return directories_sorted


def main():
    # Define augmentation functions to be applied
    # main functions
    main_functions = [
        (iaa.Rotate((-3.0, 3.0)), 'rotate'),
        (iaa.TranslateX(px=(-10, 10)), 'translatex'),
        (iaa.TranslateY(px=(-10, 10)), 'translatey'),
        (iaa.ScaleX((0.5, 1.1)), 'scalex'),
        (iaa.ScaleY((0.5, 1.1)), 'scaley')
    ]

    # complementary functions
    comp_functions = [
        (iaa.Add((-10, 10), per_channel=0.5), 'add'),
        (iaa.AdditiveGaussianNoise(scale=0.2 * 255), 'add_gaussian'),
        (iaa.AdditiveLaplaceNoise(scale=(0, 0.2 * 255)), 'add_laplace'),
        (iaa.AddElementwise((-40, 40)), 'add_wise'),
        (iaa.Multiply((0.5, 1.5), per_channel=0.5), 'mult'),
        (iaa.MultiplyElementwise((0.5, 1.5)), 'mult_wise'),
        (iaa.ReplaceElementwise(0.1, [0, 255]), 'replace'),
        (iaa.Dropout(p=(0, 0.2)), 'drop'),
        (iaa.CoarseDropout(0.02, size_percent=0.5), 'drop_coarse'),
        (iaa.ImpulseNoise(0.1), 'impulse'),
        (iaa.SaltAndPepper(0.1), 'salt'),
        (iaa.JpegCompression(compression=(70, 99)), 'compression'),
        (iaa.GaussianBlur(sigma=(0.0, 3.0)), 'blur_gaussian'),
        (iaa.AverageBlur(k=(2, 11)), 'blur_average'),
        (iaa.MedianBlur(k=(3, 11)), 'blur_median'),
        (iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)), 'blur_bilateral'),
        (iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((50, 100))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
        ]), 'color_space'),
        (iaa.ChangeColorTemperature((1100, 10000)), 'color_temp'),
        (iaa.LogContrast(gain=(0.6, 1.4)), 'contrast_log'),
        (iaa.LinearContrast((0.4, 1.6)), 'contrast_linear'),
        (iaa.Sequential([
            iaa.AllChannelsHistogramEqualization(),
            iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization())
        ]), 'histo_equalize'),
        (iaa.Fliplr(1.0), 'flipr'),
        (iaa.ShearX((-3.0, 3.0)), 'shearx'),
        (iaa.ShearY((-3.0, 3.0)), 'sheary')
    ]

    # Combined functions
    combined_functions = [(combine_functions(main_functions[0], comp_functions), 'combined_rotate'),
                          (combine_functions(main_functions[1], comp_functions), 'combined_translatex'),
                          (combine_functions(main_functions[2], comp_functions), 'combined_translatey'),
                          (combine_functions(main_functions[3], comp_functions), 'combined_scalex'),
                          (combine_functions(main_functions[4], comp_functions), 'combined_scaley')
                          ]

    functions = main_functions + comp_functions  # + combined_functions

    print("Augmentation functions loaded...")

    # Read images and coordinates from .jpg's and .txt's
    output_image = 'dataset/text/train_augmented/image/'
    output_mask = 'dataset/text/train_augmented/label/'
    path_name = 'dataset/text/train/image/'
    mask_path_name = 'dataset/text/train/label/'

    # imager per operation
    imager_per_operation = 1

    path = Path(path_name)
    mask_path = Path(mask_path_name)

    imagedir = read_images(path)
    maskdir = read_images(mask_path)

    image_files = []
    for image in imagedir:
        if str(image)[-4:] == '.jpg':
            image_files.append(image)

    mask_files = []
    for mask in maskdir:
        if str(mask)[-4:] == '.jpg':
            mask_files.append(mask)

    print("Executing functions...")

    # Run agumentation for isolated functions
    for i, image_path in tqdm(enumerate(image_files)):
        image = imageio.imread(str(image_path))
        mask = imageio.imread(str(mask_files[i]))
        mask = SegmentationMapsOnImage(mask, shape=image.shape)

        for function in functions:
            seq = iaa.Sequential([function[0]])
            for k in range(imager_per_operation):
                image_aug, mask_aug = seq(image=image, segmentation_maps=mask)

                imageio.imwrite(output_image + str(image_path).split("/")[-1][:-4] + '_' +
                                function[1] + str(k) + '.jpg', image_aug)

                imageio.imwrite(output_mask + str(mask_files[i]).split("/")[-1][:-4] + '_' +
                                function[1] + str(k) + '.jpg', mask_aug.get_arr())

    print("Finished")


if __name__ == '__main__':
    main()