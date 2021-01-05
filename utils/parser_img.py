import cv2
import numpy as np
import os
import imutils
import config
from PIL import Image, ImageFont, ImageDraw

def img_read(path):
    """
    Method to read an image.
    :param path: Path to the image you want to read.
    :return: Returns a image in a np.array format.
    """
    img = cv2.imread(str(path))
    return img

def img_write(path, img):
    """
    Method to write an image.
    :param path: Path where you want to write the image.
    :param img: The image in a np.array format.
    """

    cv2.imwrite(path, img)

def img_show(img):
    """
    Display an image on screen. Press any keyboard key to end the display.
    :param img: The image in a np.array format.
    """
    if (len(img) > 0):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def img_resize(img, scale=1, dim=[]):

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    if dim==[]:
        dim = (width, height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

def img_color2gray(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def img_gray2color(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def img_rotate90(img):
    '''
    Rotate image in 90 degrees

    :param img: image
    :return:    rotated image
    '''
    return cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

def find_points(image):
    '''

    :param image: binary image shape = [x, y]
    :return:    pts
    '''
    # gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #gray = image
    #_, threshold = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(image.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    if (image.shape[0] > image.shape[1]):
        size = image.shape[0]
    else:
        size = image.shape[1]

    # screenCnt = []
    pts = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        n = approx.ravel()
        if len(approx) == 4:
            if (cv2.contourArea(c) > 0.7 * size + 300):
                # screenCnt.append(approx)
                i = 0
                for j in n:
                    if (i % 2 == 0):
                        x = n[i]
                        y = n[i + 1]
                        pts.append([x, y])
                    i = i + 1
    return np.asarray(pts)

def draw_points(img, points, size=1, color=(255, 0, 255), invert=False, thickness=5):
    """
    Draw points on an image.
    :param img:         The image in a np.array format.
    :param points:      The 'x' and 'y' coordinates of the points you want to draw. Eg: [[x1, y1], ...].
    :param size:        The size you want to draw the points, default value is 5.
    :param color:       The color you want to draw the points, default value is color=(255, 0, 255).
    :param invert:      If True, the 'x' and 'y' coordinates will be inverted.
    """

    for i in range(len(points)):
        if (invert):
            cv2.circle(img, (int(points[i][1]), int(points[i][0])), size, color, thickness=thickness)
        else:
            cv2.circle(img, (int(points[i][0]), int(points[i][1])), size, color, thickness=thickness)


def draw_polylines(img, points, color=(255, 255, 255), isClosed=True, isFilled=True, thickness=1):
    """
    Draw a losangle on the image.
    :param img:         The image in a np.array format.
    :param points:      The 'x' and 'y' coordinates of the points you want to draw. Eg: points[0] = p_min; points[1] = p_max.
    :param color:       The color you want to draw the points, default value is color=(255, 0, 255).
    :param invert:      If True, the 'x' and 'y' coordinates will be inverted.
    :param thickness:   How large is the rectangle lines in pixels. Default = 3.
    """

    if not isFilled:
        cv2.polylines(img, points,
                      isClosed, color,
                      thickness)
    else:
        cv2.fillPoly(img, [np.asarray(points)], color=color)

def perspective(img, pts):
    """
    Transform perceptive, leave the image plan.
    :param img: 		Image.
    :param points: 		[[tof_left], [top_right], [bottom_right], [bottom_left]]
    :return:			The new image.
    """

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def add_margin(img):
    '''
    Add a white margin on the img
    :param img:     image
    :return:        image with the white margin
    '''

    img_aux = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
    img_aux[img_aux<255] = 255

    begin_x = int(img_aux.shape[0] / 4)
    final_x = int(img_aux.shape[0] / 4) * 3

    diff_x = img.shape[0] - (final_x - begin_x)
    final_x += diff_x

    begin_y = int(img_aux.shape[1] / 4)
    final_y = int(img_aux.shape[1] / 4) * 3

    diff_y = img.shape[1] - (final_y - begin_y)
    final_y += diff_y

    img_aux[begin_x:final_x, begin_y:final_y] = img
    return img_aux

def crop(cnt, img, margin=5):
    '''
    Use contours to crop image

    :param cnt:     contours
    :param img:     image
    :return:        cropped image
    '''

    x,y,w,h = cv2.boundingRect(cnt)
    y1 = y - margin
    x1 = x - margin
    x2 = x + w + margin
    y2 = y + h + margin

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > img.shape[1]:
        x2 = img.shape[1]
    if y2 > img.shape[0]:
        y2 = img.shape[0]

    cropped = img[y1 : y2, x1 : x2]
    return cropped, [x, y]

def write_on_image(img, text='', position=(0,0), font_size=12):
    '''
    Write text on image

    :param img:         image
    :param text:        text to be written
    :param position:    position to write the text
    :param font_size:   font size

    :return:            There is no return, the text will be written on the image.
    '''

    image = Image.fromarray(img)
    font_type = ImageFont.truetype(config.FONT_PATH, font_size)
    draw = ImageDraw.Draw(image)
    draw.text(xy=position, text=text, fill=(155,0,255), font=font_type)

    img_ = np.array(image)
    for i in range(256):
        img[img_ == i] = i


def find_countours(img_mask):
    #kernel = np.ones((2, 2), np.uint8)
    #mask_dilated = cv2.dilate(img_mask, kernel, iterations=1)

    cnts = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    screenCnt = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        screenCnt.append(approx)
    return screenCnt