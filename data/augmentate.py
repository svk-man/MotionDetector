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
    bbox.append(int(float(member[4][0].text)))
    bbox.append(int(float(member[4][1].text)))
    bbox.append(int(float(member[4][2].text)))
    bbox.append(int(float(member[4][3].text)))
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
                     int(float(member[4][0].text)),
                     int(float(member[4][1].text)),
                     int(float(member[4][2].text)),
                     int(float(member[4][3].text))
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
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csv_path, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csv_path, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csv_path, mode='a', index=False, sep=sep, header=False)


def main(directory_list):
    bbox_params = A.BboxParams(
        format='pascal_voc',
        min_area=1,
        min_visibility=0.5,
        label_fields=['field_id']
    )

    dataset_dir = '../train/tensorflow-object-detection-api/data/augmented/data'
    augmented_dir = 'data/augmented'
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(augmented_dir, exist_ok=True)

    for images_dir in directory_list:
        print("\nMain folder: " + images_dir)
        images_path = os.path.join(os.getcwd(), 'data/{}'.format(images_dir))

        # Пройти по всем папкам и найти images и их xmls
        images_list = []
        for i in os.walk(images_path):
            if i[2]:
                #print("Folder: ")
                #print(i[0])
                #print("with files:")
                #print(i[2])
                #print()
                path = i[0]
                path_core = path[path.find(images_path):None] + "/"
                for image_name in i[2]:
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images_list.append(path_core.replace('\\', '/') + image_name)

        images_count = len(images_list)
        i = 0
        is_cyrillic_path = False
        for image_path in images_list:
            print(image_path)
            image = cv2.imread(image_path)
            if image is None:
                f = open(image_path, "rb")
                chunk = f.read()
                chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
                image = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
                is_cyrillic_path = True
            image_name = os.path.basename(image_path)
            image_ext = os.path.splitext(image_path)[1]
            xml_path = image_path.replace(image_ext, '.xml')
            animal_id_bbox = read_bbox_from_xml(xml_path)
            # show_image(image, bbox=animal_id_bbox)

            # Аугментировать изображение
            augmentation_orig = A.Compose(
                [
                    A.RandomSizedBBoxSafeCrop(640, 640, erosion_rate=0.0, interpolation=1, always_apply=True, p=1.0),
                ],
                bbox_params=bbox_params
            )

            augmentation = A.Compose(
                [
                    A.RandomSizedBBoxSafeCrop(640, 640, erosion_rate=0.0, interpolation=1, always_apply=True, p=1.0),
                    A.VerticalFlip(always_apply=False, p=0.5),
                    A.HorizontalFlip(always_apply=False, p=0.5),
                    A.Flip(always_apply=False, p=0.5),
                    A.Transpose(always_apply=False, p=0.5),
                    #A.RandomGamma(gamma_limit=(120, 150), eps=None, always_apply=False, p=0.75),
                    A.RandomRotate90(always_apply=False, p=0.5),
                    A.Rotate(limit=(-360, 360), interpolation=1, border_mode=1, value=None, mask_value=None,
                             always_apply=False, p=0.5),
                    #A.GaussNoise(var_limit=(10.0, 150.0), mean=0, always_apply=False, p=0.25),
                    #A.ISONoise(color_shift=(0.0, 0.5), intensity=(0.1, 0.5), always_apply=False, p=0.25),
                    #A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True,
                    #                      elementwise=True, always_apply=False, p=0.25),
                    #A.Blur(blur_limit=3, always_apply=False, p=0.15),
                    #A.ToGray(always_apply=False, p=0.15),
                    #A.OneOf([
                    #    A.HueSaturationValue(hue_shift_limit=(-15, 15), sat_shift_limit=(-15, 15),
                    #                         val_shift_limit=(-15, 15), always_apply=False, p=0.5),
                    #    A.RGBShift(r_shift_limit=3, g_shift_limit=5, b_shift_limit=5, always_apply=False, p=0.5)
                    #], p=0.75),
                    #A.OneOf([
                    #    A.RandomBrightness(limit=0.1, always_apply=False, p=0.5),
                    #    A.RandomContrast(limit=0.1, always_apply=False, p=0.5),
                    #    A.RandomBrightnessContrast(brightness_limit=0.1,
                    #                               contrast_limit=0.1, brightness_by_max=False,
                    #                               always_apply=False, p=0.5),
                    #], p=0.75)
                ],
                bbox_params=bbox_params
            )

            for k in range(2):
                while True:
                    if k == 0:
                        augmented = augmentation_orig(image=image, bboxes=[animal_id_bbox], field_id=['1'])
                    else:
                        augmented = augmentation(image=image, bboxes=[animal_id_bbox], field_id=['1'])
                    if augmented['bboxes']:
                        break
                # show_image(augmented['image'], augmented['bboxes'][0])

                image_path_core = image_path[image_path.find('data/{}'.format(images_dir)):None]
                image_path_core = image_path_core.replace('data/{}'.format(images_dir), images_dir)
                image_path_core = image_path_core.replace(image_ext, '_' + str(k) + image_ext)
                augmented_path = augmented_dir + '/' + dataset_dir + '/' + image_path_core.replace('\\', '/')

                os.makedirs(os.path.split(augmented_path)[0], exist_ok=True)
                if not is_cyrillic_path:
                    cv2.imwrite(augmented_path, augmented['image'])
                else:
                    is_success, im_buf_arr = cv2.imencode(image_ext, augmented['image'])
                    im_buf_arr.tofile(augmented_path)

                writer = Writer(augmented_path, 640, 640)
                bbox_coords = augmented['bboxes'][0]
                bbox_class = read_class_from_xml(xml_path)
                writer.addObject(bbox_class, bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3])
                print(augmented_path.replace(image_ext, '.xml'))
                writer.save(augmented_path.replace(image_ext, '.xml'))

            progress = (i * 100) / images_count
            print("Progress: " + "%.2f" % progress + "%")
            i += 1

    print('\nSuccessfully augmentated images.')


main(['train', 'test'])