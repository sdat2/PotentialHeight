"""TensorFlow implementation of a Graph Attention Network (GAT)."""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np  # For sparse to dense conversion if needed for some ops


class GraphAttentionLayer(tf.keras.layers.Layer):
    """
    Graph Attention Layer (GAT).
    This is a simplified version focusing on the core attention mechanism.
    A production GAT might handle multi-head attention and edge features more explicitly.
    """

    def __init__(
        self,
        output_dim,
        num_heads=1,
        attention_dropout=0.1,
        activation=tf.nn.elu,
        use_bias=True,
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        **kwargs,
    ):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.attn_kernel_regularizer = attn_kernel_regularizer

    def build(self, input_shape):
        node_features_shape, adj_matrix_shape = (
            input_shape  # adj_matrix indicates connectivity
        )

        # Weight matrix for transforming node features (W)
        self.kernels = []
        for i in range(self.num_heads):
            kernel = self.add_weight(
                shape=(node_features_shape[-1], self.output_dim // self.num_heads),
                initializer="glorot_uniform",
                name=f"kernel_head_{i}",
                regularizer=self.kernel_regularizer,
            )
            self.kernels.append(kernel)

        # Attention mechanism weights (a)
        self.attn_kernels_self = []
        self.attn_kernels_neigh = []
        for i in range(self.num_heads):
            attn_kernel_self = self.add_weight(
                shape=(self.output_dim // self.num_heads, 1),
                initializer="glorot_uniform",
                name=f"attn_kernel_self_head_{i}",
                regularizer=self.attn_kernel_regularizer,
            )
            attn_kernel_neigh = self.add_weight(
                shape=(self.output_dim // self.num_heads, 1),
                initializer="glorot_uniform",
                name=f"attn_kernel_neigh_head_{i}",
                regularizer=self.attn_kernel_regularizer,
            )
            self.attn_kernels_self.append(attn_kernel_self)
            self.attn_kernels_neigh.append(attn_kernel_neigh)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer="zeros",
                name="bias",
                regularizer=self.bias_regularizer,
            )
        super(GraphAttentionLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        node_features, adj_matrix = inputs  # adj_matrix (N, N) defines neighbors

        # For sparse adj_matrix, convert to dense for easier neighborhood iteration here.
        # In practice, for large graphs, use tf.gather and sparse operations.
        if isinstance(adj_matrix, tf.SparseTensor):
            adj_matrix_dense = tf.sparse.to_dense(adj_matrix)
        else:
            adj_matrix_dense = adj_matrix

        outputs_all_heads = []
        for i in range(self.num_heads):
            # Transform features: h_i' = X * W_head_i
            transformed_features = tf.matmul(
                node_features, self.kernels[i]
            )  # (N, F_out / num_heads)

            # Compute attention scores: e_ij = LeakyReLU(a_self^T * h_i' + a_neigh^T * h_j')
            attn_self_scores = tf.matmul(
                transformed_features, self.attn_kernels_self[i]
            )  # (N, 1)
            attn_neigh_scores = tf.matmul(
                transformed_features, self.attn_kernels_neigh[i]
            )  # (N, 1)

            # Broadcast to create pairwise scores
            # This is a simplified way; actual GAT uses edge-wise computation.
            # For a node i, its attention score with neighbor j depends on features of i and j.
            # A more direct implementation would iterate over edges or use gather operations.
            # Let's use a common approximation for dense adjacency:
            attention_scores = attn_self_scores + tf.transpose(
                attn_neigh_scores
            )  # (N, N) - e_ij for all pairs

            # Masking non-neighbors
            # Add a large negative number where there's no edge to make softmax output ~0
            mask = (1.0 - adj_matrix_dense) * -1e9
            attention_scores += mask
            attention_scores = tf.nn.softmax(
                attention_scores, axis=1
            )  # Normalize across neighbors (axis=1)

            if training:
                attention_scores = tf.nn.dropout(
                    attention_scores, rate=self.attention_dropout
                )

            # Weighted sum of neighbor features: h_i_final = sum_{j in N_i} (alpha_ij * h_j')
            context = tf.matmul(
                attention_scores, transformed_features
            )  # (N, F_out / num_heads)
            outputs_all_heads.append(context)

        # Concatenate or average outputs from all heads
        if self.num_heads > 1:
            output = tf.concat(outputs_all_heads, axis=-1)  # (N, F_out)
        else:
            output = outputs_all_heads[0]  # (N, F_out)

        if self.use_bias:
            output = tf.add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(GraphAttentionLayer, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": tf.keras.regularizers.serialize(
                    self.bias_regularizer
                ),
                "attn_kernel_regularizer": tf.keras.regularizers.serialize(
                    self.attn_kernel_regularizer
                ),
            }
        )
        return config


def GAT(
    input_dim,
    output_dim,
    num_nodes,
    hidden_units=[64, 32],
    num_heads_per_layer=[4, 4],
    attention_dropout=0.1,
    dropout_rate=0.5,
    l2_reg=5e-4,
):
    """
    Builds a GAT model.

    Args:
        input_dim (int): Dimensionality of input node features.
        output_dim (int): Dimensionality of output.
        num_nodes (int): Number of nodes in the graph.
        hidden_units (list): List of integers for hidden layer dimensions.
        num_heads_per_layer (list): List of int for number of attention heads in each GAT layer.
        attention_dropout (float): Dropout for attention coefficients.
        dropout_rate (float): Dropout rate for hidden layers.
        l2_reg (float): L2 regularization factor.

    Returns:
        tf.keras.Model: The GAT model.
    """
    node_features_input = Input(
        shape=(input_dim,), name="node_features_input", dtype=tf.float32
    )
    adj_matrix_input = Input(
        shape=(num_nodes,), name="adj_matrix_input", sparse=True, dtype=tf.float32
    )  # For connectivity

    x = node_features_input
    adj = adj_matrix_input

    # Hidden GAT layers
    for i, units in enumerate(hidden_units):
        heads = num_heads_per_layer[i] if i < len(num_heads_per_layer) else 1
        x = GraphAttentionLayer(
            units,
            num_heads=heads,
            attention_dropout=attention_dropout,
            activation=tf.nn.elu,  # ELU is common in GAT
            kernel_regularizer=l2(l2_reg),
            attn_kernel_regularizer=l2(l2_reg),
        )([x, adj])
        x = Dropout(dropout_rate)(x)

    # Output layer (can be another GAT layer or a Dense layer)
    # If the last hidden GAT layer has multiple heads, it might average them instead of concatenating
    # For the final output, usually a GAT layer with 1 head is used, or a Dense layer
    final_heads = (
        num_heads_per_layer[-1] if len(num_heads_per_layer) > len(hidden_units) else 1
    )
    output_activation = None  # Linear for regression
    if (
        len(hidden_units) < len(num_heads_per_layer)
        and num_heads_per_layer[len(hidden_units)] > 1
    ):
        # Averaging for the output layer if multi-head
        gat_outs = []
        for h in range(num_heads_per_layer[len(hidden_units)]):
            # This is a simplification. Typically, the GAT layer itself handles multi-head averaging.
            # Here, we'd need to make the GAT layer aware it's an output layer for averaging.
            # Or, use a GAT layer that outputs directly output_dim.
            pass  # This part needs refinement based on how multi-head output is handled.
        # For simplicity, let's assume the last GAT layer before Dense concatenates,
        # and then a Dense layer produces the final output, or a GAT layer with output_dim and 1 head.

    # Let's use a GAT layer with 1 head for the output or average heads.
    # For simplicity, using a Dense layer after the last GAT features
    # If x is (N, F_hidden), a Dense layer can map this to (N, F_out)
    output = Dense(
        output_dim, activation=output_activation, kernel_regularizer=l2(l2_reg)
    )(x)

    # Alternatively, a final GAT layer:
    # output = GraphAttentionLayer(
    #     output_dim,
    #     num_heads=1, # Often 1 head for output, or average if multiple
    #     attention_dropout=attention_dropout,
    #     activation=output_activation,
    #     kernel_regularizer=l2(l2_reg),
    #     attn_kernel_regularizer=l2(l2_reg)
    # )([x, adj])

    model = Model(
        inputs=[node_features_input, adj_matrix_input], outputs=output, name="GAT_Model"
    )
    return model


if __name__ == "__main__":
    # python -m surgenet.gat
    num_nodes_example = 100
    node_feature_dim_example = 10
    output_dim_example = 2

    gat_model = GAT(
        input_dim=node_feature_dim_example,
        output_dim=output_dim_example,
        num_nodes=num_nodes_example,
        hidden_units=[64, 32],  # Output dims for GAT layers
        num_heads_per_layer=[
            8,
            1,
        ],  # 8 heads for first GAT, 1 for the GAT acting as output
        attention_dropout=0.1,
        dropout_rate=0.3,
    )
    gat_model.summary()
