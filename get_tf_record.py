import tensorflow as tf
                
           
                
                
                
feature_description = {
             "image":tf.io.FixedLenFeature([],tf.string),
             "label":tf.io.FixedLenFeature([],tf.int64)
           }

# 将TFRecord 文件中的每一个序列化的 tf.train.Example 解码
def parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string,feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image']) # 解码JPEG图片
    # feature_dict['image'] = tf.image.resize(feature_dict['image'],[224,224])/ 255.0  # 改变图片尺寸并进行归一化
    return feature_dict['image'],feature_dict['label']     
                
                
def read_TFRecond_file(tfrecord_file):
    # 读取TFRecord 文件
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    # 解码
    dataset = raw_dataset.map(parse_example)
    
    return dataset               
                
                
# Dataset的数据缓冲器大小，和数据集大小及规律有关
buffer_size = 20000
# Dataset的数据批次大小，每批次多少个样本数
batch_size = 8                
                
dataset_train = read_TFRecond_file('/home/xu/MyData/faces_CASIA_112x112_aligned.tfrecords')  # 解码

import PIL
import numpy as np
import os
from tqdm import tqdm
c=0
for i,j in tqdm(dataset_train):
    c+=1
    
    label=str(j.numpy())
    if not os.path.exists(label):
        os.makedirs(label)
    else:
        pic=PIL.Image.fromarray(np.array(i))
        pic.save('./'+label+'/'+str(c)+'.jpg')
       
    # pic=PIL.Image.fromarray(np.array(i))
    

    # print(i.shape)
















        
