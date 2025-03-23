import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Dense, Dropout
from spektral.layers import RGCNConv

def create_rgcn_link_prediction_model(
    num_entities,
    num_relations,
    node_features_dim,
    hidden_dim=32,
):
    """
    R-GCN 多关系版本：
      - 需要给定多个 adjacency inputs，一共 num_relations 个
      - 用 RGCNConv 根据多关系邻接进行消息传递
      - 依旧使用 DistMult 作为打分函数
    """
    # 1) 节点特征输入：形状 (N, F)
    node_features_input = Input(
        shape=(node_features_dim,),
        name="node_features_input"
    )

    # 2) 多个邻接矩阵输入 (num_relations 个)
    #    每个都 shape=(N, N)，可稀疏
    adj_inputs = []
    for r in range(num_relations):
        adj_in = Input(shape=(None,), sparse=True, name=f"adj_input_rel{r}")
        adj_inputs.append(adj_in)
    
    # 三元组 (头、关系、尾) 的索引
    head_indices = Input(shape=(), dtype=tf.int32, name="head_indices")
    relation_indices = Input(shape=(), dtype=tf.int32, name="relation_indices")
    tail_indices = Input(shape=(), dtype=tf.int32, name="tail_indices")

    # ------------------------------------------------
    # R-GCN 卷积部分
    x0 = Dense(hidden_dim)(node_features_input)  # 先做一层Dense
    # RGCNConv 要求输入: [x0] + [adj_0, adj_1, ..., adj_{R-1}]
    x1 = RGCNConv(channels=hidden_dim,
                  num_relations=num_relations,
                  activation='relu')([x0] + adj_inputs)
    x1 = Dropout(0.3)(x1)

    x2 = RGCNConv(channels=hidden_dim,
                  num_relations=num_relations,
                  activation='relu')([x1] + adj_inputs)
    x2 = Dropout(0.3)(x2)

    # 最终节点表示
    node_embeddings = Dense(hidden_dim, activation='relu')(x2)  # (N, hidden_dim)

    # ------------------------------------------------
    # 关系嵌入
    relation_embedding_raw = Embedding(num_relations, hidden_dim)(relation_indices)
    relation_embeddings = Dense(hidden_dim, activation='relu')(relation_embedding_raw)

    # gather 头尾
    head_embeddings = tf.gather(node_embeddings, head_indices)
    tail_embeddings = tf.gather(node_embeddings, tail_indices)

    # DistMult 打分
    distmult_score = tf.reduce_sum(
        head_embeddings * relation_embeddings * tail_embeddings, axis=-1
    )
    output = tf.sigmoid(distmult_score)

    # ------------------------------------------------
    # 构造最终模型
    prediction_model = Model(
        inputs=[node_features_input] + adj_inputs
               + [head_indices, relation_indices, tail_indices],
        outputs=output
    )

    # 仅输出整图节点嵌入
    embedding_model = Model(
        inputs=[node_features_input] + adj_inputs,
        outputs=node_embeddings
    )

    return prediction_model, embedding_model
