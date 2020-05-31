import xml.etree.ElementTree as ET
import os
import glob
import pandas as pd

parent_folder = os.path.basename(os.getcwd())

names_file_path = "names.txt"
names = []


def xml_to_csv(path, type_directory):
    xml_list = []
    path_core = path[path.find('{}'.format(type_directory)):None] + "/"
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            cls = member[0].text
            if cls not in names:
                names.append(cls)
            value = (path_core.replace('\\', '/') + root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     cls,
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
        txt_file_name = '{}.txt'.format(Image_cat)
        image_path = os.path.join(os.getcwd(), '{}'.format(Image_cat))
        if os.path.exists(txt_file_name):
            os.remove(txt_file_name)
        print(image_path)
        for i in os.walk(image_path):
            if i[2]:
                print("Folder: ")
                print(i[0])
                print("with images:")
                print(i[2])
                print()
                xml_df = xml_to_csv(i[0], Image_cat)
                save_xml_to_txt(xml_df, txt_file_name)

        print('Successfully converted xml to txt.')


main(['train', 'test'])

print("Classes:", names)
with open(names_file_path, "w") as file:
    for name in names:
        file.write(str(name)+'\n')
