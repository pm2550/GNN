import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Embedding, Concatenate
from spektral.layers import RGCNConv

def create_multi_rel_gnn_with_link_prediction(
    num_entities,
    num_relations,
    node_features_dim,
    hidden_dim=32,
    use_bases=False,
    num_bases=None
):
    """
    多关系GNN(R-GCN)模型，用于链接预测：
      - 结合节点嵌入和链接预测功能
      - 提供两个模型：预测模型和嵌入模型
    参数:
      num_entities: 实体数量
      num_relations: 关系种类数
      node_features_dim: 节点特征维度 F
      hidden_dim: 隐层大小
      use_bases: 是否使用basis decomposition减少参数
      num_bases: basis个数, 如果use_bases=True, 需设定
    """
    # 节点特征输入 (N, F)
    node_features_input = Input(shape=(node_features_dim,), name="node_features_input")
    
    # 多个adjacency输入 - 每种关系一个邻接矩阵 (N, N)
    adj_inputs = []
    for r in range(num_relations):
        # shape=(None,) + sparse=True => (N, N) 稀疏
        adj_in = Input(shape=(None,), sparse=True, name=f"adj_r{r}")
        adj_inputs.append(adj_in)
    
    # 链接预测需要的额外输入
    head_indices = Input(shape=(), dtype=tf.int32, name="head_indices")
    relation_indices = Input(shape=(), dtype=tf.int32, name="relation_indices")
    tail_indices = Input(shape=(), dtype=tf.int32, name="tail_indices")
    
    # 初始特征变换
    x0 = Dense(hidden_dim, activation='relu')(node_features_input)
    
    # RGCN 第1层
    x1 = RGCNConv(
        channels=hidden_dim,
        num_relations=num_relations,
        num_bases=num_bases if use_bases else None,
        activation='relu'
    )([x0] + adj_inputs)
    x1 = Dropout(0.3)(x1)
    
    # RGCN 第2层
    x2 = RGCNConv(
        channels=hidden_dim,
        num_relations=num_relations,
        num_bases=num_bases if use_bases else None,
        activation='relu'
    )([x1] + adj_inputs)
    x2 = Dropout(0.3)(x2)
    
    # 生成最终实体嵌入
    node_embeddings = Dense(hidden_dim, activation='relu')(x2)
    
    # 链接预测部分 - 获取头尾实体嵌入
    head_embeddings = tf.gather(node_embeddings, head_indices)
    tail_embeddings = tf.gather(node_embeddings, tail_indices)
    
    # 关系嵌入（可学习参数）
    relation_embedding_raw = Embedding(num_relations, hidden_dim)(relation_indices)
    relation_embeddings = Dense(hidden_dim, activation='relu')(relation_embedding_raw)
    
    # 链接预测评分机制（三种方式组合）
    # 1. DistMult方式: head * relation * tail
    head_relation_mul = tf.multiply(head_embeddings, relation_embeddings)
    mult_score = tf.reduce_sum(tf.multiply(head_relation_mul, tail_embeddings), axis=1)
    
    # 2. 双线性方式: head·W·tail
    head_transformed = Dense(hidden_dim, use_bias=False)(head_embeddings)
    bilinear_score = tf.reduce_sum(tf.multiply(head_transformed, tail_embeddings), axis=1)
    
    # 3. 拼接方式: MLP(concat(head, relation, tail))
    concat_embeddings = Concatenate()([head_embeddings, relation_embeddings, tail_embeddings])
    concat_score = Dense(hidden_dim, activation='relu')(concat_embeddings)
    concat_score = Dense(1)(concat_score)
    concat_score = tf.squeeze(concat_score, axis=-1)
    
    # 组合所有评分
    combined_score = mult_score + 0.5 * bilinear_score + 0.3 * concat_score
    
    # 最终链接预测分数
    output = tf.sigmoid(combined_score)
    
    # 构造两个模型
    # 1. 链接预测模型
    prediction_model = Model(
        inputs=[node_features_input] + adj_inputs + [head_indices, relation_indices, tail_indices],
        outputs=output,
        name="RGCN_LinkPrediction"
    )
    
    # 2. 节点嵌入模型（用于提取特征）
    embedding_model = Model(
        inputs=[node_features_input] + adj_inputs,
        outputs=node_embeddings,
        name="RGCN_Embedding"
    )
    
    return prediction_model, embedding_model