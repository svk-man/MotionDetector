import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
def xml_to_csv(path, type_directory):
    xml_list = []
    path_core = path[path.find('data/{}'.format(type_directory)):None] + "/"
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (path_core + root.find('filename').text,
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
    for Image_cat in directory_list:
        xml_file_name = 'data/{}_labels.csv'.format(Image_cat)
        image_path = os.path.join(os.getcwd(), 'data/{}'.format(Image_cat))
        if os.path.exists(xml_file_name):
            os.remove(xml_file_name)
        print(image_path)
        for i in os.walk(image_path):
            if i[2]:
                print("Folder: ")
                print(i[0])
                print("with images:")
                print(i[2])
                print()
                xml_df = xml_to_csv(i[0], Image_cat)
                save_sml_to_csv(xml_df, xml_file_name)

        print('Successfully converted xml to csv.')

main(['train','test'])