import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
import spektral
from spektral.layers import GCNConv, GATConv
from spektral.models.gcn import GCN
from spektral.data import Graph
from spektral.data.loaders import SingleLoader


interpreter = tf.lite.Interpreter(model_path="C:/Users/pm/Desktop/GNN_new/25633(1000)/rgcn_link_pred_int8_edgetpu.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
for i, detail in enumerate(input_details):
    print(f"Input {i}: shape={detail['shape']}")
    # 节点特征（保持不变）
# 节点特征
node_features_data = np.random.randn(1000, 32).astype(np.float32)

# 尾节点索引（用于链接预测）
tail_index_data = np.zeros(1, dtype=np.int32)

# 邻居索引（注意形状变成了[1, 9]）
neighbor_indices_data = np.zeros((1, 9), dtype=np.int32)

# 头节点索引（用于链接预测）
head_index_data = np.zeros(1, dtype=np.int32)

# 关系索引
relation_indices_data = np.zeros((1, 9), dtype=np.int32)

# 邻居掩码（注意这是float32类型）
neighbor_mask_data = np.zeros((1, 9), dtype=np.float32)

# 设置输入张量
interpreter.set_tensor(input_details[0]['index'], node_features_data)
interpreter.set_tensor(input_details[1]['index'], tail_index_data)
interpreter.set_tensor(input_details[2]['index'], neighbor_indices_data)
interpreter.set_tensor(input_details[3]['index'], head_index_data)
interpreter.set_tensor(input_details[4]['index'], relation_indices_data)
interpreter.set_tensor(input_details[5]['index'], neighbor_mask_data)