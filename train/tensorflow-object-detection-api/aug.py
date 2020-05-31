import os
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from pylab import rcParams
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import albumentations as A
import random
import xml.etree.ElementTree as ET

#%matplotlib inline
#%config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BOX_COLOR = (255, 0, 0)


def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
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

# Пройти по всем папкам и найти images и их xmls
IMAGES_PATH = "data/train"

images_list = []
for i in os.walk(IMAGES_PATH):
    if i[2]:
        print("Folder: ")
        print(i[0])
        print("with files:")
        print(i[2])
        print()
        path = i[0]
        path_core = path[path.find(IMAGES_PATH):None] + "/"
        for image_name in i[2]:
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                images_list.append(path_core.replace('\\', '/') + image_name)

bbox_params = A.BboxParams(
    format='pascal_voc',
    min_area=1,
    min_visibility=0.5,
    label_fields=['field_id']
)
for image_path in images_list:
    image = cv2.imread(image_path)
    image_name, image_ext = os.path.splitext(image_path)
    xml_path = image_path.replace(image_ext, '.xml')
    animal_id_bbox = read_bbox_from_xml(xml_path)
    #show_image(image, bbox=animal_id_bbox)

    # Аугментировать изображение
    augmentation = A.Compose([
        A.Flip(always_apply=True)
    ], bbox_params=bbox_params)

    augmented = augmentation(image=image, bboxes=[animal_id_bbox], field_id=['1'])
    show_image(augmented['image'], augmented['bboxes'][0])


form = cv2.imread('data/train/20190400_OPF_mice_/WIN_20190429_14_03_34_Pro/images/image60.jpg')
STUDENT_ID_BBOX = [674, 369, 757, 624]

show_image(form, bbox=STUDENT_ID_BBOX)



# [x_min, y_min, x_max, y_max], e.g. [97, 12, 247, 212].
bbox_params = A.BboxParams(
    format='pascal_voc',
    min_area=1,
    min_visibility=0.5,
    label_fields=['field_id']
)

aug = A.Compose([
    A.Flip(always_apply=True)
], bbox_params=bbox_params)

show_augmented(aug, form, STUDENT_ID_BBOX)

aug = A.Compose([
  A.Rotate(limit=80, always_apply=True)
], bbox_params=bbox_params)

show_augmented(aug, form, STUDENT_ID_BBOX)

aug = A.Compose([
    A.RandomGamma(gamma_limit=(400, 500), always_apply=True)
], bbox_params=bbox_params)

show_augmented(aug, form, STUDENT_ID_BBOX)

aug = A.Compose([
    A.RandomBrightnessContrast(always_apply=True),
], bbox_params=bbox_params)

show_augmented(aug, form, STUDENT_ID_BBOX)

aug = A.Compose([
    A.RGBShift(
      always_apply=True,
      r_shift_limit=100,
      g_shift_limit=100,
      b_shift_limit=100
    ),
], bbox_params=bbox_params)

show_augmented(aug, form, STUDENT_ID_BBOX)

aug = A.Compose([
    A.GaussNoise(
      always_apply=True,
      var_limit=(100, 300),
      mean=150
    ),
], bbox_params=bbox_params)

show_augmented(aug, form, STUDENT_ID_BBOX)

doc_aug = A.Compose([
    A.Flip(p=0.25),
    A.RandomGamma(gamma_limit=(20, 300), p=0.5),
    A.RandomBrightnessContrast(p=0.85),
    A.Rotate(limit=35, p=0.9),
    A.RandomRotate90(p=0.25),
    A.RGBShift(p=0.75),
    A.GaussNoise(p=0.25)
], bbox_params=bbox_params)

show_augmented(doc_aug, form, STUDENT_ID_BBOX)

DATASET_PATH = 'data/augmented'
IMAGES_PATH = f'{DATASET_PATH}/images'

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)

rows = []
for i in tqdm(range(100)):
  augmented = doc_aug(image=form, bboxes=[STUDENT_ID_BBOX], field_id=['1'])
  file_name = f'form_aug_{i}.jpg'
  for bbox in augmented['bboxes']:
    x_min, y_min, x_max, y_max = map(lambda v: int(v), bbox)
    rows.append({
      'file_name': f'images/{file_name}',
      'x_min': x_min,
      'y_min': y_min,
      'x_max': x_max,
      'y_max': y_max,
      'class': 'mouse'
    })

  cv2.imwrite(f'{IMAGES_PATH}/{file_name}', augmented['image'])

pd.DataFrame(rows).to_csv(f'{DATASET_PATH}/annotations.csv', header=True, index=None)

