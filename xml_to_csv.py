import os
import glob #.xml uzantılı dosyaları bulmamızı sağlar.
import pandas as pd
import xml.etree.ElementTree as ET # API'leri kullanarak XML'yi ayrıştırır.


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file) #Hiyerarşik bir ağaç yapısı oluşturur.
        root = tree.getroot() #Kök elemanını döndürür.
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text), #4 düğümleri gösterir. int(member[4][4].text) olsaydı. Düğüm 4 değil 5 olacaktı.
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value) # Value'leri alıp oluşturduğumuz boş listeye atıyor.
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train','test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder)) # os.path.join: Bir veya daha yolu birleştirir.
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/' + folder + '_labels.csv'), index=None) #images klasörüne gider.
        print('Başarıyla XML i CSV ye dönüştürdü.')


main()
