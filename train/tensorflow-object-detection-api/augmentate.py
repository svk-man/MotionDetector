import os
import glob
import pandas as pd
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

    dataset_path = 'data/augmented/data'
    os.makedirs(dataset_path, exist_ok=True)

    for images_dir in directory_list:
        images_path = os.path.join(os.getcwd(), 'data/{}'.format(images_dir))

        # Пройти по всем папкам и найти images и их xmls
        images_list = []
        for i in os.walk(images_path):
            if i[2]:
                print("Folder: ")
                print(i[0])
                print("with files:")
                print(i[2])
                print()
                path = i[0]
                path_core = path[path.find(images_path):None] + "/"
                for image_name in i[2]:
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images_list.append(path_core.replace('\\', '/') + image_name)

        for image_path in images_list:
            image = cv2.imread(image_path)
            image_name = os.path.basename(image_path)
            image_ext = os.path.splitext(image_path)[1]
            xml_path = image_path.replace(image_ext, '.xml')
            animal_id_bbox = read_bbox_from_xml(xml_path)
            # show_image(image, bbox=animal_id_bbox)

            # Аугментировать изображение
            augmentation = A.Compose([
                A.RandomSizedBBoxSafeCrop(width=640, height=640, erosion_rate=0.0, interpolation=1, always_apply=True, p=1.0)
            ], bbox_params=bbox_params)

            augmented = augmentation(image=image, bboxes=[animal_id_bbox], field_id=['1'])
            # show_image(augmented['image'], augmented['bboxes'][0])

            image_path_core = image_path[image_path.find('data/{}'.format(images_dir)):None]
            image_path_core = image_path_core.replace('data/{}'.format(images_dir), images_dir)
            augmented_path = dataset_path + '/' + image_path_core.replace('\\', '/')

            os.makedirs(os.path.split(augmented_path)[0], exist_ok=True)
            cv2.imwrite(augmented_path, augmented['image'])

            writer = Writer(augmented_path, 640, 640)
            bbox_coords = augmented['bboxes'][0]
            bbox_class = read_class_from_xml(xml_path)
            writer.addObject(bbox_class, bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3])
            writer.save(augmented_path.replace(image_ext, '.xml'))

    print('Successfully augmentated images.')


main(['train', 'test'])