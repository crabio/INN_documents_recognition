import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf


def crop_image_by_rect(image, rect):
    return image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]


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


def find_inn_text(image):
    template = cv2.imread('templates/inn.png')
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    weight, height = template_gray.shape[::-1]

    res = cv2.matchTemplate(image, template_gray, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = max_loc
    bottom_right = (top_left[0] + weight, top_left[1] + height)
    probability = max_val

    return top_left, bottom_right, probability


def get_inn_number_frame(image, inn_top_left, inn_bottom_right):
    inn_frame_shape = (1100, 110)

    frame_center_left = (inn_bottom_right[0], int(inn_bottom_right[1] / 2 + inn_top_left[1] / 2))

    frame_top_left = (frame_center_left[0], int(frame_center_left[1] - inn_frame_shape[1] / 2))
    frame_bottom_right = (frame_top_left[0] + inn_frame_shape[0], frame_top_left[1] + inn_frame_shape[1])

    return crop_image_by_2_corners(image, frame_top_left, frame_bottom_right)


def load_network_config():
    input_shape = (28, 28, 1)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = "model/cp.ckpt"

    model.load_weights(checkpoint_path)

    return model


def network_recognize_digit(model, image):
    image_gray_resized = cv2.bitwise_not(cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA))
    image_gray_delated = cv2.dilate(image_gray_resized, (3, 3))

    data_float = image_gray_delated.astype('float32')
    data_float /= 255

    pred = model.predict(data_float.reshape(1, 28, 28, 1))

    # print(pred.round(decimals=4)[0][pred.argmax()], pred.argmax())

    return pred.round(decimals=4)[0][pred.argmax()], pred.argmax()


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
            digit_frame = crop_image_by_4_corners(img, approx)

            digit_frame_list.append(digit_frame)

    # fig = plt.figure()
    # for i, digit_frame in enumerate(reversed(digit_frame_list)):
    #     fig.add_subplot(1, len(digit_frame_list), i + 1)
    #     plt.axis('off')
    #     plt.imshow(digit_frame, cmap='gray')
    # plt.show()

    return reversed(digit_frame_list)


def find_digit_inside_frame(img):
    img_thresh = prepare_image(img, 127, inverse=True)

    # Get contours
    m2, finded_contours, hierarchy = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in finded_contours]
    rect = rects[0]

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
    digit_frame = np.pad(digit_frame,
                         ((horizon_pads_count, horizon_pads_count), (vertical_pads_count, vertical_pads_count)),
                         'constant', constant_values=(0, 0))

    digit_frame_inv = cv2.bitwise_not(digit_frame)

    return digit_frame_inv


classifier = load_network_config()

img_rgb = cv2.imread('image/inn-1.jpg')

img_thresh = prepare_image(img_rgb, 200)

inn_top_left, inn_bottom_right, inn_probability = find_inn_text(img_thresh)

inn_robability = inn_probability

inn_number_frame = get_inn_number_frame(img_rgb, inn_top_left, inn_bottom_right)

plt.imshow(inn_number_frame, cmap='gray')
plt.show()

digit_frame_list = find_digit_frames(inn_number_frame)

inn_numbers_list = []
for digit_frame in digit_frame_list:
    digit_img = find_digit_inside_frame(digit_frame)

    digit_probability, digit_number = network_recognize_digit(classifier, digit_img)

    inn_probability *= digit_probability

    inn_numbers_list.append(digit_number)

    # plt.imshow(digit_img, cmap='gray')
    # plt.show()

# 1 - error inn len, 2 - error checksum
def check_inn(inn_numbers_list):
    normal_inn_len = 12
    inn_muxes_1_list = np.array((7, 2, 4, 10, 3, 5, 9, 4, 6, 8))
    inn_muxes_2_list = np.array((3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8 ))
    if len(inn_numbers_list) != normal_inn_len:
        return 1
    else:
        # Get first 10 digits and convert to numpy array
        # Mux each digit to specific number
        mux_1_result = np.sum(np.multiply(np.array(inn_numbers_list[:10]), inn_muxes_1_list))
        # Calc first checksum number
        inn_1_checksum = mux_1_result - (mux_1_result // 11 * 11)
        if inn_1_checksum == 10:
            inn_1_checksum = 0
        # Get first 11 digits and convert to numpy array
        # Mux each digit to specific number
        mux_2_result = np.sum(np.multiply(np.array(inn_numbers_list[:11]), inn_muxes_2_list))
        # Calc second checksum number
        inn_2_checksum = mux_2_result - (mux_2_result // 11 * 11)
        if inn_2_checksum == 10:
            inn_2_checksum = 0

        if inn_numbers_list[10] == inn_1_checksum and inn_numbers_list[11] == inn_2_checksum:
            return 0
        else:
            return 2

check_status = check_inn(inn_numbers_list)

if check_status == 0:
    print(inn_probability, ''.join(str(e) for e in inn_numbers_list))
elif check_status == 1:
    print('Digits count < 12. Not like in INN')
elif check_status == 2:
    print('Checksum is not corect.')
else:
    print('Unknown error')