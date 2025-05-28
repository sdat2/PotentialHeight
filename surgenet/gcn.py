"""Graph Convolutional Network (GCN) implementation using TensorFlow/Keras."""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class GCNLayer(tf.keras.layers.Layer):
    """
    A Graph Convolutional Network Layer.
    """

    def __init__(
        self,
        output_dim,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        super(GCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shape):
        node_features_shape, adj_matrix_shape = input_shape
        self.kernel = self.add_weight(
            shape=(node_features_shape[-1], self.output_dim),
            initializer="glorot_uniform",
            name="kernel",
            regularizer=self.kernel_regularizer,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer="zeros",
                name="bias",
                regularizer=self.bias_regularizer,
            )
        super(GCNLayer, self).build(input_shape)

    def call(self, inputs):
        node_features, adj_matrix = (
            inputs  # adj_matrix should be normalized (e.g., D^-0.5 * A * D^-0.5)
        )

        # GCN operation: A_hat * X * W
        # A_hat is the normalized adjacency matrix (often with self-loops)
        # X is the node feature matrix
        # W is the weight matrix

        support = tf.matmul(node_features, self.kernel)
        output = tf.sparse.sparse_dense_matmul(
            adj_matrix, support
        )  # Use sparse_dense_matmul if adj is sparse

        if self.use_bias:
            output = tf.add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(GCNLayer, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": tf.keras.regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config


def GCN(
    input_dim,
    output_dim,
    num_nodes,
    hidden_units=[64, 32],
    dropout_rate=0.5,
    l2_reg=5e-4,
):
    """
    Builds a GCN model.

    Args:
        input_dim (int): Dimensionality of input node features.
        output_dim (int): Dimensionality of output (e.g., 1 for water surface elevation).
        num_nodes (int): Number of nodes in the graph.
        hidden_units (list): List of integers for hidden layer dimensions.
        dropout_rate (float): Dropout rate.
        l2_reg (float): L2 regularization factor.

    Returns:
        tf.keras.Model: The GCN model.
    """
    node_features_input = Input(
        shape=(input_dim,), name="node_features_input", dtype=tf.float32
    )
    # Adjacency matrix (can be sparse). For simplicity, assuming dense here for Input definition,
    # but use tf.sparse.SparseTensor in practice if it's sparse.
    adj_matrix_input = Input(
        shape=(num_nodes,), name="adj_matrix_input", sparse=True, dtype=tf.float32
    )

    x = node_features_input
    adj = adj_matrix_input

    # Hidden GCN layers
    for units in hidden_units:
        x = GCNLayer(units, activation=tf.nn.relu, kernel_regularizer=l2(l2_reg))(
            [x, adj]
        )
        x = Dropout(dropout_rate)(x)

    # Output GCN layer
    # Typically, for node-level prediction, the final layer might be a GCN layer or a Dense layer
    # If using GCNLayer for output, ensure its activation is appropriate (e.g., linear for regression)
    output = GCNLayer(output_dim, activation=None, kernel_regularizer=l2(l2_reg))(
        [x, adj]
    )
    # Or, if predicting a single value per node directly from features:
    # output = Dense(output_dim, activation=None)(x)

    model = Model(
        inputs=[node_features_input, adj_matrix_input], outputs=output, name="GCN_Model"
    )
    return model


if __name__ == "__main__":
    # python -m surgenet.gcn
    num_nodes_example = 100
    node_feature_dim_example = 10  # e.g., bathymetry, friction, forcing terms at t
    output_dim_example = 2  # e.g., water surface elevation and velocity_u at t+dt

    # Create a dummy GCN model
    gcn_model = GCN(
        input_dim=node_feature_dim_example,
        output_dim=output_dim_example,
        num_nodes=num_nodes_example,
        hidden_units=[64, 32],
        dropout_rate=0.3,
    )
    gcn_model.summary()

    # To train this model, you'd need:
    # 1. Node features (N, F_in) - N nodes, F_in features
    # 2. Normalized adjacency matrix (N, N) - sparse or dense
    # 3. Target values (N, F_out) - e.g., water levels at next timestep
