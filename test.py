import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Dense, Dropout, Layer

# 自定义多关系卷积层 (简化版)
class MyRelationalConv(Layer):
    def __init__(self, hidden_dim, num_relations, **kwargs):
        super().__init__(**kwargs)
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        # 为每种关系单独的权重矩阵 W_r
        self.relation_weights = []
        for r in range(num_relations):
            w = self.add_weight(
                shape=(hidden_dim, hidden_dim),
                initializer='glorot_uniform',
                trainable=True,
                name=f"W_rel{r}"
            )
            self.relation_weights.append(w)

    def call(self, inputs):
        """
        inputs = [x] + [adj_r0, adj_r1, ..., adj_r{num_relations-1}]
        x: (N, hidden_dim)
        adj_rk: (N, N), sparse or dense
        """
        x = inputs[0]
        adjs = inputs[1:]  # list of adjacency for each relation
        # 合并各关系的卷积结果
        outputs = 0
        for r_idx, adj in enumerate(adjs):
            # message passing from neighbors using W_r
            w_r = self.relation_weights[r_idx]  # shape=(hidden_dim, hidden_dim)
            xw = tf.matmul(x, w_r)  # (N, hidden_dim)
            # A * XW => (N, hidden_dim)
            if isinstance(adj, tf.SparseTensor):
                ax = tf.sparse.sparse_dense_matmul(adj, xw)
            else:
                ax = tf.matmul(adj, xw)
            outputs += ax
        return outputs  # sum over all relations

def create_multi_rel_gnn(
    num_entities,
    num_relations,
    node_features_dim,
    hidden_dim=32
):
    """
    自定义一个“多关系 GNN”:
      - 对每个关系都有一个邻接矩阵
      - 手写多关系卷积 (MyRelationalConv)
      - 最后依旧 DistMult 打分
    """
    node_features_input = Input(shape=(node_features_dim,), name="node_features_input")
    adj_inputs = []
    for r in range(num_relations):
        adj_inputs.append(Input(shape=(None,), sparse=True, name=f"adj_input_r{r}"))

    head_indices = Input(shape=(), dtype=tf.int32, name="head_indices")
    rel_indices  = Input(shape=(), dtype=tf.int32, name="relation_indices")
    tail_indices = Input(shape=(), dtype=tf.int32, name="tail_indices")

    # 1) 初始映射
    x0 = Dense(hidden_dim, activation='relu')(node_features_input)  # (N, hidden_dim)
    
    # 2) 多关系卷积 (第一层)
    x1 = MyRelationalConv(hidden_dim, num_relations)([x0] + adj_inputs)
    x1 = Dropout(0.3)(x1)
    x1 = tf.nn.relu(x1)

    # 3) 多关系卷积 (第二层)
    x2 = MyRelationalConv(hidden_dim, num_relations)([x1] + adj_inputs)
    x2 = Dropout(0.3)(x2)
    x2 = tf.nn.relu(x2)

    # 最终实体表示
    node_embeddings = Dense(hidden_dim, activation='relu')(x2)

    # 关系嵌入 (Emb + Dense)
    rel_emb_raw = Embedding(num_relations, hidden_dim)(rel_indices)
    rel_emb = Dense(hidden_dim, activation='relu')(rel_emb_raw)

    # gather 头尾
    h_emb = tf.gather(node_embeddings, head_indices)
    t_emb = tf.gather(node_embeddings, tail_indices)

    # DistMult
    score = tf.reduce_sum(h_emb * rel_emb * t_emb, axis=-1)
    output = tf.sigmoid(score)

    # 模型
    prediction_model = Model(
        inputs=[node_features_input] + adj_inputs + [head_indices, rel_indices, tail_indices],
        outputs=output
    )
    embedding_model = Model(
        inputs=[node_features_input] + adj_inputs,
        outputs=node_embeddings
    )
    return prediction_model, embedding_model

