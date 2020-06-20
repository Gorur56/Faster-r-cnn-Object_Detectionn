# Bu program, nesne algılama gerçekleştirmek için TensorFlow tarafından eğitilmiş bir sınıflandırıcı kullanır.
# Sınıflandırıcıyı bir görüntü üzerinde nesne algılama gerçekleştirmek için kullanır.
# Metal plaka üzerindeki kusurları kutu içine alır ve tespit değerini yazar.

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob
from pathlib import Path
import glob
# Localde çalışırken object_detection klasöründe saklandığından bu gereklidir.
sys.path.append("..")

from utils import label_map_util #labelmap.pbtxt içeriğini yüklemek için kullanılır.
from utils import visualization_utils as vis_util

# Modeli kullanmak için oluşturduğumuz frozen_inference_graph.pb'ın oldupuğu klasör. 
MODEL_NAME = 'inference_graph'

# Geçerli çalışma dizinine giden yolu yakalar.
CWD_PATH = os.getcwd()
#Kullanılan modeli içeren frozen_detection_graph.pb dosyasının yolu
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

#Training dosyasından labelmap.pbtxt yolunu alır.
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt') #Hangi id de hangi sınıf var omu söyler.

# Resim yolu
PATH_TO_IMAGE = os.path.join(CWD_PATH,'scratch.jpg')

# Nesne algılayıcısının tanımlayabileceği sınıf sayısı.
NUM_CLASSES = 3

# Etiket haritasını yükler. 
# Etiket, harita indekslerini kategori adlarıyla eşleştirir,
# böylece evrişim ağımız "1" i tahmin ettiğinde, bunun "DefectTypeB" e karşılık geldiğini biliyoruz. 
# Burada dahili yardımcı program işlevlerini kullanıyoruz, 
# ancak uygun dize etiketlerine sözlük eşleme tam sayıları döndüren her şey iyi olurdu.

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Tensorflow modelini belleğe yükleme.
detection_graph = tf.Graph() #Veri akışı grafiği olarak temsil edilen bir TensorFlow hesaplaması
with detection_graph.as_default():
    od_graph_def = tf.GraphDef() 
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph) #Xml belgesine dönüştürür.
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph) #TensorFlow işlemlerini çalıştıran bir sınıf

# Nesne algılama sınıflandırıcısı için giriş ve çıkış tensörlerini (yani veriler) tanımlar.

# Giriş tensörü
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') #Tensorflow tensörü alıyor.

# Çıkış tensörleri algılama çerçeveleri, puanlar ve sınıflardır.
# Her çerçeve, görüntünün belirli bir nesnenin algılandığı bir bölümünü temsil eder.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Her puan nesnelerin her biri için güven seviyesini temsil eder.
# Skor, sınıf etiketi ile birlikte sonuç görüntüsünde gösterilir.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Algılanan nesne sayısı
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# OpenCV kullanarak görüntü okuyun ve şekle sahip olmak için görüntü boyutlarını genişletin: [1, None, None, 3],
# yani sütundaki her öğenin piksel RGB değerine sahip olduğu tek sütunlu bir dizi.
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0) 

# Modeli görüntü olarak girdi olarak çalıştırarak gerçek algılamayı gerçekleştirir.
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# görüntüyü görselleştirir. 

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes), #squeeze: Tek boyutlu girişleri kaldırır.
    np.squeeze(classes).astype(np.int32), #classes'ın kopyasını, belirtilen türe dönüştürür.
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True, #Kusurları kutucuk içine alır.
    line_thickness=5, 
    min_score_thresh=0.80)

# Tüm sonuçlar resim üzerine çizilmiştir. Şimdi görüntüyü göster.
cv2.imshow('Object detector', image)

# Görüntüyü kapatmak için herhangi bir tuşa basın
cv2.waitKey(0)
cv2.destroyAllWindows()
