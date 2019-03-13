import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

images_dir = 'image/frames'


def crop_image_by_rect(image, rect):
    return image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]


def crop_image_by_center_wh(image, center, w, h):
    return image[center[1] - h // 2:center[1] + h // 2, center[0] - w // 2:center[0] + w // 2]


def crop_image_by_2_corners(image, top_left, bottom_right):
    return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


def crop_image_by_4_corners(image, corners):
    top_left = tuple(corners[0][0])
    bottom_right = tuple(corners[2][0])

    return crop_image_by_2_corners(image, top_left, bottom_right)


def prepare_image(image, threshold_level, inverse=False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold binary
    if inverse:
        ret, thresh_bin = cv2.threshold(img_gray, threshold_level, 255, cv2.THRESH_BINARY_INV)
    else:
        ret, thresh_bin = cv2.threshold(img_gray, threshold_level, 255, cv2.THRESH_BINARY)

    return thresh_bin


def find_digit_frames(img):
    img_thresh = prepare_image(img, 200)

    m2, finded_contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    digit_frame_list = []
    for cnt in finded_contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.13 * peri, True)
        area_threshold = 10
        area = cv2.contourArea(cnt)

        if len(approx) == 4 and area > area_threshold:
            img_thresh_hard = prepare_image(img, 127)
            digit_frame = crop_image_by_4_corners(img_thresh_hard, approx)

            digit_frame_list.append(digit_frame)

    fig = plt.figure()
    for i, digit_frame in enumerate(reversed(digit_frame_list)):
        fig.add_subplot(1, len(digit_frame_list), i + 1)
        plt.axis('off')
        plt.imshow(digit_frame, cmap='gray')
    plt.show()

    return reversed(digit_frame_list)


def find_digit_inside_frame(img):
    img_thresh = prepare_image(img, 90, inverse=True)
    img_shape = img_thresh.shape

    # Get contours
    m2, finded_contours, hierarchy = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in finded_contours]

    digit_frame_list = []
    for rect in rects:
        digit_frame = crop_image_by_rect(img_thresh, rect)

        # Horizon pads count = height * 1 / 5. If equal 0, then min 1
        horizon_pads_count = rect[3] * 2 // 5
        if horizon_pads_count == 0:
            horizon_pads_count = 1
        # Vertical pads count = ((horizon_pads_count * 2 + img_height) - img_width) // 2
        vertical_pads_count = ((horizon_pads_count * 2 + rect[3]) * 5 // 6 - rect[2]) // 2
        if vertical_pads_count <= 0:
            vertical_pads_count = 1

        # Add padding
        digit_frame = np.pad(digit_frame, ((horizon_pads_count, horizon_pads_count), (vertical_pads_count, vertical_pads_count)),
               'constant', constant_values=(0, 0))

        digit_frame_list.append(cv2.bitwise_not(digit_frame))

    return reversed(digit_frame_list)


def save_images(folder, parent_name, images_list):
    for i, image in enumerate(images_list):
        print('{0}/{1}_{2}.jpg'.format(folder, parent_name, i + 1))
        cv2.imwrite('{0}/{1}_{2}.jpg'.format(folder, parent_name, i + 1), image)


images_file_names_list = [file for file in os.listdir(images_dir) if os.path.isfile(images_dir + '/' + file)]
images_count = len(images_file_names_list)

print(images_file_names_list)

fig = plt.figure()
for i, image_file_name in enumerate(images_file_names_list):
    image_rgb = cv2.imread(images_dir + '/' + image_file_name)

    digit_frame_list = find_digit_inside_frame(image_rgb)

    save_images(images_dir + '/digits', os.path.splitext(image_file_name)[0], digit_frame_list)
