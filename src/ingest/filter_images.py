import pandas as pd
import cv2
from pyzbar.pyzbar import decode

from src import IMAGES_DIR_PATH, REJECTED_IMAGES_DIR_PATH, IMAGES_DF_PATH

def is_qrcode(gray_image):
    detected_objects = decode(gray_image)
    if detected_objects:
        return True

    inverted_image = cv2.bitwise_not(gray_image)
    detected_objects = decode(inverted_image)

    return len(detected_objects) > 0

def is_gradient(gray_image, threshold=100):

    h, w = gray_image.shape

    if h < 100 and w < 100:
        return True

    margin_h = int(h * 0.1)
    margin_w = int(w * 0.1)

    if h > 2 * margin_h and w > 2 * margin_w:
        cropped_image = gray_image[margin_h:h - margin_h, margin_w:w - margin_w]
    else:
        return True

    laplacian_var = cv2.Laplacian(cropped_image, cv2.CV_64F).var()
    return laplacian_var < threshold

def filter_images():
    df_images = pd.read_pickle(IMAGES_DF_PATH)

    filtered = []

    for index, image_row in df_images.iterrows():
        image_path = IMAGES_DIR_PATH / image_row['image_path']
        image = cv2.imread(image_path)

        if image is None:
            filtered.append(False)
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            filtered.append(not (is_qrcode(gray_image) or is_gradient(gray_image)))

        if not filtered[-1]:
            image_path.rename(REJECTED_IMAGES_DIR_PATH/ image_row['image_path'])

    df_images['filtered'] = filtered
    df_images.to_pickle(IMAGES_DF_PATH)