import os
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import albumentations as A
import cv2
from pascal_voc_writer import Writer

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

    photos_path = "../train/tensorflow-object-detection-api/Photos"
    labels_path = "../train/tensorflow-object-detection-api/Labels"
    os.makedirs(photos_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    image_index = 0
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

            if not is_cyrillic_path:
                cv2.imwrite(photos_path + '/image_' + str(image_index) + image_ext, image)
            else:
                is_success, im_buf_arr = cv2.imencode(image_ext, image)
                im_buf_arr.tofile(photos_path + '/image_' + str(image_index) + image_ext)

            writer = Writer(photos_path + '/image_' + str(image_index) + image_ext, 300, 300)
            bbox_class = read_class_from_xml(xml_path)
            writer.addObject(bbox_class, animal_id_bbox[0], animal_id_bbox[1], animal_id_bbox[2], animal_id_bbox[3])
            writer.save(labels_path + '/xml_' + str(image_index) + '.xml')
            image_index += 1

            progress = (i * 100) / images_count
            print("Progress: " + "%.2f" % progress + "%")
            i += 1

    print('\nSuccessfully translated images.')


main(['train', 'test'])