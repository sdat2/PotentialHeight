"""Graph Convolutional Network (GCN) implementation using TensorFlow/Keras."""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2, Regularizer

# Removed: from tensorflow.keras.activations import Activation (causes ImportError)
from typing import List, Optional, Tuple, Callable, Union, Dict, Any


class GCNLayer(tf.keras.layers.Layer):
    """
    A Graph Convolutional Network Layer.
    """

    def __init__(
        self,
        output_dim: int,
        activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = tf.nn.relu,
        use_bias: bool = True,
        kernel_regularizer: Optional[Regularizer] = None,
        bias_regularizer: Optional[Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super(GCNLayer, self).__init__(**kwargs)
        self.output_dim: int = output_dim
        # The type hint for activation is Callable, which is appropriate.
        # No need for a separate Activation type from keras.activations for this.
        self.activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = activation
        self.use_bias: bool = use_bias
        self.kernel_regularizer: Optional[Regularizer] = kernel_regularizer
        self.bias_regularizer: Optional[Regularizer] = bias_regularizer
        self.kernel: Optional[tf.Variable] = None  # Will be defined in build
        self.bias: Optional[tf.Variable] = (
            None  # Will be defined in build if use_bias is True
        )

    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
        """
        Builds the layer.

        This method is called the first time the layer is called on an input.
        It's here that you will define your weights.

        Args:
            input_shape: A tuple of TensorShapes: (node_features_shape, adj_matrix_shape).
                         node_features_shape is (batch_size, num_nodes, num_features_in).
                         adj_matrix_shape is (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes) if shared.
                         More commonly, for GCNs, node_features_shape is (num_nodes, num_features_in)
                         and adj_matrix_shape is (num_nodes, num_nodes).
                         The code assumes the latter, simpler case for feature processing.
        """
        node_features_shape, adj_matrix_shape = input_shape

        # Ensure node_features_shape has at least 1 dimension (features_in)
        # For GCNs, node_features_shape is typically (num_nodes, features_in)
        # So, len(node_features_shape) would be 2.
        # If features are passed as (features_in,), then len would be 1.
        if (
            not isinstance(node_features_shape, tf.TensorShape)
            or len(node_features_shape) < 1
        ):
            raise ValueError(
                f"Node features input shape must be a TensorShape with at least 1 dimension (features_in), "
                f"but got {node_features_shape}"
            )

        last_dim_node_features = node_features_shape[-1]
        if last_dim_node_features is None:
            raise ValueError(
                "The last dimension of the node features input shape cannot be None. "
                "Please provide a defined feature dimension."
            )

        self.kernel = self.add_weight(
            shape=(last_dim_node_features, self.output_dim),
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

    def call(
        self, inputs: Tuple[tf.Tensor, Union[tf.Tensor, tf.SparseTensor]]
    ) -> tf.Tensor:
        """
        Forward pass of the GCN layer.

        Args:
            inputs: A tuple containing:
                - node_features (tf.Tensor): The input node features (shape: [num_nodes, input_dim]).
                - adj_matrix (Union[tf.Tensor, tf.SparseTensor]): The normalized adjacency matrix
                  (shape: [num_nodes, num_nodes]). Should be normalized (e.g., D^-0.5 * A * D^-0.5).

        Returns:
            tf.Tensor: The output features after graph convolution (shape: [num_nodes, output_dim]).
        """
        node_features, adj_matrix = inputs

        if self.kernel is None:
            # This can happen if the layer is called without being built.
            # For dynamic models, build might not be called explicitly before the first call.
            # We can try to infer input_shape and build here, or raise an error.
            # For simplicity, raising an error or ensuring build is called is safer.
            raise RuntimeError(
                "Layer kernel has not been initialized. Ensure build() is called before call()."
            )

        # GCN operation: A_hat * X * W
        # A_hat is the normalized adjacency matrix (often with self-loops)
        # X is the node feature matrix
        # W is the weight matrix

        support = tf.matmul(node_features, self.kernel)  # X * W

        # Handle sparse or dense adjacency matrix
        if isinstance(adj_matrix, tf.SparseTensor):
            output = tf.sparse.sparse_dense_matmul(
                adj_matrix, support
            )  # A_hat * (X * W)
        elif isinstance(adj_matrix, tf.Tensor):
            output = tf.matmul(adj_matrix, support)  # A_hat * (X * W)
        else:
            raise TypeError(
                f"Adjacency matrix must be a tf.Tensor or tf.SparseTensor, got {type(adj_matrix)}"
            )

        if self.use_bias:
            if self.bias is None:
                raise RuntimeError(
                    "Layer bias has not been initialized even though use_bias is True. Ensure build() is called."
                )
            output = tf.add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.
        """
        config = super(GCNLayer, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "activation": (
                    tf.keras.activations.serialize(self.activation)
                    if self.activation
                    else None
                ),
                "use_bias": self.use_bias,
                "kernel_regularizer": (
                    tf.keras.regularizers.serialize(self.kernel_regularizer)
                    if self.kernel_regularizer
                    else None
                ),
                "bias_regularizer": (
                    tf.keras.regularizers.serialize(self.bias_regularizer)
                    if self.bias_regularizer
                    else None
                ),
            }
        )
        return config


def GCN(
    input_dim: int,
    output_dim: int,
    num_nodes: int,
    hidden_units: List[int] = [64, 32],
    dropout_rate: float = 0.5,
    l2_reg: float = 5e-4,
) -> Model:
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
    # Adjacency matrix (can be sparse).
    # For GCN, adj_matrix_input should represent connections for all nodes.
    # So, its shape should be (num_nodes, num_nodes) if it's dense,
    # or it should be a SparseTensor representing the same.
    # The Input layer with sparse=True expects indices and values if you were to feed it directly.
    # However, typically, the sparse tensor is constructed outside and passed in.
    # The shape (num_nodes,) for adj_matrix_input is unconventional for a typical GCN adjacency matrix.
    # If adj_matrix is indeed sparse and has shape (num_nodes, num_nodes),
    # then tf.keras.Input(shape=(num_nodes, num_nodes), sparse=True, ...) would be more explicit.
    # Or, if it's a list of neighbors/edges (not a matrix), the processing in GCNLayer would differ.
    # Assuming the GCNLayer expects a [num_nodes, num_nodes] matrix:
    adj_matrix_input = Input(
        shape=(num_nodes, num_nodes),
        name="adj_matrix_input",
        sparse=True,
        dtype=tf.float32,
        # If it were dense: shape=(num_nodes, num_nodes), sparse=False
        # The original shape=(num_nodes,) was ambiguous for an adjacency matrix.
        # Changed to (num_nodes, num_nodes) for clarity, assuming a standard adj matrix.
    )

    x: tf.Tensor = node_features_input
    adj: Union[tf.Tensor, tf.SparseTensor] = (
        adj_matrix_input  # Type will be SparseTensor due to sparse=True
    )

    # Hidden GCN layers
    for units in hidden_units:
        x = GCNLayer(units, activation=tf.nn.relu, kernel_regularizer=l2(l2_reg))(
            [x, adj]
        )
        x = Dropout(dropout_rate)(x)

    # Output GCN layer
    # Typically, for node-level prediction, the final layer might be a GCN layer or a Dense layer
    # If using GCNLayer for output, ensure its activation is appropriate (e.g., linear for regression)
    output_tensor: tf.Tensor = GCNLayer(output_dim, activation=None, kernel_regularizer=l2(l2_reg))(  # type: ignore
        [x, adj]
    )
    # Or, if predicting a single value per node directly from features:
    # output_tensor = Dense(output_dim, activation=None)(x)

    model = Model(
        inputs=[node_features_input, adj_matrix_input],
        outputs=output_tensor,
        name="GCN_Model",
    )
    return model


if __name__ == "__main__":
    # Example usage:
    # python -m your_module_name.gcn_file_name (replace with actual path)
    num_nodes_example: int = 100
    node_feature_dim_example: int = 10  # e.g., bathymetry, friction, forcing terms at t
    output_dim_example: int = 2  # e.g., water surface elevation and velocity_u at t+dt

    # Create a dummy GCN model
    gcn_model: Model = GCN(
        input_dim=node_feature_dim_example,
        output_dim=output_dim_example,
        num_nodes=num_nodes_example,
        hidden_units=[64, 32],
        dropout_rate=0.3,
        l2_reg=1e-4,  # Adjusted l2_reg for example
    )
    gcn_model.summary()

    # To train this model, you'd need:
    # 1. Node features (tf.Tensor or np.ndarray): Shape (num_nodes, input_dim)
    #    Example:
    #    dummy_node_features = tf.random.normal((num_nodes_example, node_feature_dim_example))
    #
    # 2. Normalized adjacency matrix (tf.SparseTensor or tf.Tensor): Shape (num_nodes, num_nodes)
    #    This is crucial and needs to be preprocessed correctly (e.g., D^-0.5 * A * D^-0.5).
    #    Example (creating a dummy sparse adjacency matrix):
    #    import numpy as np
    #    # Create some random edges for a sparse matrix
    #    num_edges = 300
    #    row_indices = np.random.randint(0, num_nodes_example, num_edges)
    #    col_indices = np.random.randint(0, num_nodes_example, num_edges)
    #    # Ensure no self-loops for this simple example, or handle them appropriately
    #    # For a real GCN, you'd typically add self-loops (A_hat = A + I)
    #    # and then normalize (D_hat^-0.5 * A_hat * D_hat^-0.5).
    #    # Here, we'll just create a sparse tensor with 1s for existing edges.
    #    unique_edges_list = []
    #    for r, c in zip(row_indices, col_indices):
    #        if r != c: # Avoid self-loops for this simple example
    #            unique_edges_list.append((r,c))
    #
    #    # Remove duplicate edges
    #    unique_edges = sorted(list(set(unique_edges_list)))
    #
    #    if not unique_edges: # Handle case with no edges after filtering
    #        print("Warning: No edges created for the dummy adjacency matrix. Using an empty sparse tensor.")
    #        sparse_indices = tf.zeros((0,2), dtype=tf.int64) # Ensure correct rank for indices
    #        sparse_values = tf.zeros((0,), dtype=tf.float32)
    #    else:
    #        sparse_indices_np = np.array(unique_edges)
    #        sparse_indices = tf.convert_to_tensor(sparse_indices_np, dtype=tf.int64)
    #        sparse_values = tf.ones(shape=(tf.shape(sparse_indices)[0],), dtype=tf.float32)
    #
    #    dummy_adj_matrix_sparse = tf.SparseTensor(
    #        indices=sparse_indices,
    #        values=sparse_values,
    #        dense_shape=(num_nodes_example, num_nodes_example)
    #    )
    #    # It's important to reorder the sparse tensor for TF operations
    #    dummy_adj_matrix_sparse = tf.sparse.reorder(dummy_adj_matrix_sparse)

    # 3. Target values (tf.Tensor or np.ndarray): Shape (num_nodes, output_dim)
    #    Example:
    #    dummy_target_values = tf.random.normal((num_nodes_example, output_dim_example))

    # Compile the model
    # gcn_model.compile(optimizer='adam', loss='mse') # Or another appropriate loss/optimizer

    # Fit the model (example, won't run without actual data)
    # print("\n--- Example Training (Conceptual) ---")
    # print("Note: This requires actual data and proper adjacency matrix normalization.")
    # try:
    #     # Ensure dummy_node_features and dummy_adj_matrix_sparse are defined
    #     if 'dummy_node_features' in locals() and 'dummy_adj_matrix_sparse' in locals() and 'dummy_target_values' in locals():
    #          print("Attempting to fit model with dummy data...")
    #          # history = gcn_model.fit(
    #          # [dummy_node_features, dummy_adj_matrix_sparse],
    #          # dummy_target_values,
    #          # epochs=2, # Reduced epochs for quick test
    #          # batch_size=num_nodes_example, # Often full-batch for GCNs
    #          # verbose=1
    #          # )
    #          # print("Conceptual training would proceed here if data were real and model compiled.")
    #          print("Conceptual training step skipped (model not compiled, data is dummy).")
    #     else:
    #         print("Skipping conceptual training: dummy data not fully initialized.")
    # except Exception as e:
    #     print(f"Skipping conceptual training due to placeholder data setup: {e}")
    #     print("To run training, uncomment, compile the model, and provide real, normalized data.")

    print(f"\nModel Input Names: {gcn_model.input_names}")
    print(f"Model Output Names: {gcn_model.output_names}")
    print(f"Model Input shapes: {[inp.shape for inp in gcn_model.inputs]}")
    print(f"Model Output shapes: {[out.shape for out in gcn_model.outputs]}")

    # Example of making a prediction (requires correctly formatted inputs)
    # try:
    #     if 'dummy_node_features' in locals() and 'dummy_adj_matrix_sparse' in locals():
    #         # Ensure inputs match the model's expected input shapes
    #         # The node features are typically (num_nodes, input_dim)
    #         # The adjacency matrix is (num_nodes, num_nodes)
    #         # If batching is used, an outer batch dimension would be present.
    #         # For this model, inputs are [node_features_input, adj_matrix_input]
    #         # node_features_input has shape (None, input_dim) after build
    #         # adj_matrix_input has shape (None, num_nodes, num_nodes) after build if batching,
    #         # or (num_nodes, num_nodes) if not batching (as defined in Input layer)

    #         # If predicting for a single graph (no batch dim):
    #         # dummy_node_features_pred = tf.random.normal((num_nodes_example, node_feature_dim_example))
    #         # dummy_adj_matrix_sparse_pred = tf.sparse.reorder(tf.SparseTensor(
    #         #    indices=dummy_adj_matrix_sparse.indices, # Use indices from above
    #         #    values=dummy_adj_matrix_sparse.values,   # Use values from above
    #         #    dense_shape=(num_nodes_example, num_nodes_example)
    #         # ))
    #         # predictions = gcn_model.predict([dummy_node_features_pred, dummy_adj_matrix_sparse_pred])
    #         # print(f"\nShape of predictions: {predictions.shape}") # (num_nodes_example, output_dim_example)
    #         print("Prediction step skipped (requires compiled model and potentially batched input).")
    # except Exception as e:
    #     print(f"Could not make dummy predictions: {e}")
