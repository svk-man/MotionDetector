import xml.etree.ElementTree as ET
import os
import glob
import pandas as pd

script_path = 'C:/Users/kravc/PycharmProjects/MotionDetector/train/tensorflow-yolov3'
data_path = 'data/my_data/'
#names_file_path = "./data/names.txt"
names = []
row = 0


def xml_to_csv(path, type_directory):
    xml_list = []
    path_core = path[path.find('/' + data_path + '{}'.format(type_directory)):None] + "/"
    path_core = script_path + path_core
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            cls = member[0].text
            #if cls not in names:
            #    names.append(cls)
            if cls == 'rat':
                cls_id = 0
            elif cls == 'mouse':
                cls_id = 1
            value = (path_core.replace('\\', '/') + root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     cls,
                     int(float(member[4][0].text)),
                     int(float(member[4][1].text)),
                     int(float(member[4][2].text)),
                     int(float(member[4][3].text))
                     )
            xml_list.append(value)
            if os.path.exists('./' + data_path + type_directory + '.txt'):
                append_write = 'a'
            else:
                append_write = 'w'

            txt_file = open('./' + data_path + type_directory + '.txt', append_write)
            global row
            txt_file.write(str(row) + " " + value[0] + " " + str(value[1]) + " " + str(value[2]) + " " +
                           str(cls_id) + " " + str(value[4]) + " " + str(value[5]) + " " + str(value[6]) + " " + str(value[7]) + "\n")
            row += 1
            txt_file.close()
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# https://stackoverflow.com/questions/17530542/how-to-add-pandas-data-to-an-existing-csv-file
def save_xml_to_txt(df, txt_path, sep=","):
    if not os.path.isfile(txt_path):
        df.to_csv(txt_path, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(txt_path, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. TXT file has " + str(len(pd.read_csv(csv_path, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(txt_path, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and txt file do not match!!")
    else:
        df.to_csv(txt_path, mode='a', index=False, sep=sep, header=False)


def main(directory_list):
    for Image_cat in directory_list:
        txt_file_name = './' + data_path + '{}.txt'.format(Image_cat)
        image_path = script_path + '/' + data_path + '{}'.format(Image_cat)
        print(image_path)
        if os.path.exists(txt_file_name):
            os.remove(txt_file_name)
        for i in os.walk(image_path):
            if i[2]:
                print("Folder: ")
                print(i[0])
                print("with images:")
                print(i[2])
                print()
                xml_df = xml_to_csv(i[0], Image_cat)
                #save_xml_to_txt(xml_df, txt_file_name)

        global row
        row = 0

        print('Successfully converted xml to txt.')


main(['train', 'test', 'val'])

#print("Classes:", names)
#with open(names_file_path, "w") as file:
#    for name in names:
#        file.write(str(name)+'\n')
