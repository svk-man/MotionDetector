import os
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import albumentations as A
import cv2
from pascal_voc_writer import Writer
import matplotlib.pyplot as plt


def read_bbox_from_xml(xml_path):
    bbox = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    member = root.find('object')
    bbox.append(int(member[4][0].text))
    bbox.append(int(member[4][1].text))
    bbox.append(int(member[4][2].text))
    bbox.append(int(member[4][3].text))
    return bbox


def read_class_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    member = root.find('object')
    return member.find('name').text


def xml_to_csv(path, type_directory):
    xml_list = []
    path_core = path[path.find('data/{}'.format(type_directory)):None] + "/"
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (path_core.replace('\\', '/') + root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


# https://stackoverflow.com/questions/17530542/how-to-add-pandas-data-to-an-existing-csv-file
def save_sml_to_csv(df, csv_path, sep=","):
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csv_path, nrows=1, sep=sep).columns):
        raise Exception(
            "Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(
                len(pd.read_csv(csv_path, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csv_path, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csv_path, mode='a', index=False, sep=sep, header=False)


def visualize_bbox(img, bbox, color=(255, 0, 0), thickness=2):
    x_min, y_min, x_max, y_max = map(lambda v: int(v), bbox)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


def show_image(image, bbox):
    image = visualize_bbox(image.copy(), bbox)
    f = plt.figure(figsize=(18, 12))
    plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        interpolation='nearest'
    )
    plt.axis('off')
    f.tight_layout()
    plt.show()


def main():
    bbox_params = A.BboxParams(
        format='pascal_voc',
        min_area=1,
        min_visibility=0.5,
        label_fields=['field_id']
    )

    # Test image from folder: OPF_rat
    folder_path = '../augmentate/images/OPF_rat'

    image_name = 'image'
    image_ext = '.jpg'
    image_width = 1920
    image_height = 1080
    image_path = folder_path + '/' + image_name + image_ext
    xml_path = folder_path + '/' + image_name + '.xml'

    augmented_path = folder_path + '/augmented'
    os.makedirs(augmented_path, exist_ok=True)

    image = cv2.imread(image_path)
    animal_id_bbox = read_bbox_from_xml(xml_path)
    # show_image(image, bbox=animal_id_bbox)

    augmentation_list = {
        # 'Blur': A.Blur(blur_limit=(8, 12), always_apply=True, p=1.0),
        # 'VerticalFlip': A.VerticalFlip(always_apply=True, p=1.0),
        # 'HorizontalFlip': A.HorizontalFlip(always_apply=True, p=1.0),
        # 'Flip': A.Flip(always_apply=True, p=1.0), # если оба, и horizontal, и vertical - то нормально, иначе - повтор
        # 'Transpose': A.Transpose(always_apply=True, p=1.0),
        # 'RandomGamma': A.RandomGamma(gamma_limit=(40, 200), eps=None, always_apply=True, p=1.0),
        # 'RandomRotate90': A.RandomRotate90(always_apply=True, p=1.0),
        # 'Rotate': A.Rotate(limit=(-360, 360), interpolation=1, border_mode=2, value=None, mask_value=None,
        #                   always_apply=True, p=1.0),
        # 'HueSaturationValue': A.HueSaturationValue(hue_shift_limit=(-15, 15), sat_shift_limit=(-15, 15),
        #                                           val_shift_limit=(-15, 15), always_apply=True, p=1.0),
        # 'RGBShift': A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, always_apply=True, p=1.0),
        # 'RandomBrightness': A.RandomBrightness(limit=0.2, always_apply=True, p=1.0),
        # 'RandomContrast': A.RandomContrast(limit=0.2, always_apply=True, p=1.0),
        # 'MotionBlur': A.MotionBlur(blur_limit=(8, 12), always_apply=True, p=1.0),
        # 'MedianBlur': A.MedianBlur(blur_limit=7, always_apply=True, p=1.0),
        # 'GaussianBlur': A.GaussianBlur(blur_limit=7, always_apply=True, p=1.0),
        # 'GaussNoise': A.GaussNoise(var_limit=(10.0, 250.0), mean=0, always_apply=True, p=1.0),
        # 'CLAHE': A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
        # 'ToGray': A.ToGray(always_apply=True, p=1.0),
        # 'RandomBrightnessContrast': A.RandomBrightnessContrast(brightness_limit=0.2,
        #                                                        contrast_limit=0.2, brightness_by_max=True,
        #                                                        always_apply=True, p=1.0),
        # 'ISONoise': A.ISONoise(color_shift=(0.0, 0.5), intensity=(0.1, 0.5), always_apply=True, p=1.0),
        # 'MultiplicativeNoise': A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True,
        #                                             elementwise=True, always_apply=True, p=1.0)
    }

    for k, v in augmentation_list.items():
        augmentation = A.Compose([
            v
        ],
            bbox_params=bbox_params
        )

        augmented = augmentation(image=image, bboxes=[animal_id_bbox], field_id=['1'])
        augmented_image_path = augmented_path + "/" + image_name + '_' + k + image_ext

        cv2.imwrite(augmented_image_path, augmented['image'])

        writer = Writer(augmented_path, image_width, image_height)
        bbox_coords = augmented['bboxes'][0]
        bbox_class = read_class_from_xml(xml_path)
        writer.addObject(bbox_class, bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3])
        writer.save(augmented_image_path.replace(image_ext, '.xml'))

    for i in range(100):
        augmentation = A.Compose([
            A.RandomSizedBBoxSafeCrop(300, 300, erosion_rate=0.0, interpolation=1, always_apply=True, p=1.0),
            A.VerticalFlip(always_apply=False, p=0.5),
            A.HorizontalFlip(always_apply=False, p=0.5),
            A.Flip(always_apply=False, p=0.5),
            A.Transpose(always_apply=False, p=0.5),
            A.RandomGamma(gamma_limit=(40, 150), eps=None, always_apply=False, p=0.25),
            A.RandomRotate90(always_apply=False, p=0.5),
            A.Rotate(limit=(-360, 360), interpolation=1, border_mode=1, value=None, mask_value=None,
                     always_apply=False, p=0.5),
            A.GaussNoise(var_limit=(10.0, 150.0), mean=0, always_apply=False, p=0.25),
            A.ISONoise(color_shift=(0.0, 0.5), intensity=(0.1, 0.5), always_apply=False, p=0.25),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True,
                                  elementwise=True, always_apply=False, p=0.25),
            A.Blur(blur_limit=3, always_apply=False, p=0.15),
            A.ToGray(always_apply=False, p=0.15),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(-15, 15), sat_shift_limit=(-15, 15),
                                     val_shift_limit=(-15, 15), always_apply=False, p=0.5),
                A.RGBShift(r_shift_limit=3, g_shift_limit=5, b_shift_limit=5, always_apply=False, p=0.5)
            ], p=0.25),
            A.OneOf([
                A.RandomBrightness(limit=0.2, always_apply=False, p=0.5),
                A.RandomContrast(limit=0.2, always_apply=False, p=0.5)
            ], p=0.5)
        ],
            bbox_params=bbox_params
        )

        augmented = augmentation(image=image, bboxes=[animal_id_bbox], field_id=['1'])
        augmented_image_path = augmented_path + "/" + image_name + '_' + str(i) + image_ext

        cv2.imwrite(augmented_image_path, augmented['image'])

        writer = Writer(augmented_path, image_width, image_height)
        bbox_coords = augmented['bboxes'][0]
        bbox_class = read_class_from_xml(xml_path)
        writer.addObject(bbox_class, bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3])
        writer.save(augmented_image_path.replace(image_ext, '.xml'))

    print('\nSuccessfully augmentated images.')


main()
