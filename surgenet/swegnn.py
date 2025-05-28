"""Initial SWE GNN implementation based on Bentivoglio et al. (2023). (Gemini 2.5 Pro)"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Dense,
    Input,
    Concatenate,
)  # Import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class SWEGNNLayer(Layer):
    """
    A Hydraulics-Based Graph Neural Network Layer based on Bentivoglio et al. (2023).
    This layer computes messages based on differences in water surface elevation
    and other hydraulic properties, then aggregates them to update node (cell) states.

    Key node features assumed (can be extended):
    - h: water depth (prognostic)
    - z_b: bed elevation (static)

    Key edge features assumed (can be extended):
    - L_ij: length of the interface
    - n_ij: Manning's roughness at the interface (or derived from cells)
    - z_b_ij: Bed elevation at the interface (or average/min of cell bed elevations)
    """

    def __init__(
        self,
        edge_mlp_units=[32, 32],
        node_mlp_units=[32],
        activation=tf.nn.relu,
        use_bias=True,
        kernel_regularizer=None,
        predict_delta=True,
        **kwargs,
    ):
        super(SWEGNNLayer, self).__init__(**kwargs)
        self.edge_mlp_units = edge_mlp_units
        self.node_mlp_units = node_mlp_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.predict_delta = predict_delta

    def build(self, input_shape):
        node_features_shape, edge_index_shape, edge_attributes_shape = input_shape
        edge_mlp_in_dim = 2 * 2 + edge_attributes_shape[-1]

        layers = []
        for units in self.edge_mlp_units:
            layers.append(
                Dense(
                    units,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.kernel_regularizer,
                )
            )
        layers.append(
            Dense(
                1,
                activation=None,
                use_bias=self.use_bias,
                kernel_regularizer=self.kernel_regularizer,
                name="edge_message",
            )
        )
        self.edge_message_mlp = tf.keras.Sequential(layers, name="EdgeMessageMLP")

        node_mlp_input_dim = 1 + 1 + 1

        layers = []
        for units in self.node_mlp_units:
            layers.append(
                Dense(
                    units,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.kernel_regularizer,
                )
            )
        layers.append(
            Dense(
                1,
                activation=None,
                use_bias=self.use_bias,
                kernel_regularizer=self.kernel_regularizer,
                name="node_update",
            )
        )
        self.node_update_mlp = tf.keras.Sequential(layers, name="NodeUpdateMLP")
        super(SWEGNNLayer, self).build(input_shape)

    def call(self, inputs):
        node_features, edge_index, edge_attributes = inputs
        h_t = node_features[:, 0:1]
        zb_cell = node_features[:, 1:2]

        idx_sender = edge_index[0]
        idx_receiver = edge_index[1]

        h_sender = tf.gather(h_t, idx_sender)
        zb_sender = tf.gather(zb_cell, idx_sender)
        h_receiver = tf.gather(h_t, idx_receiver)
        zb_receiver = tf.gather(zb_cell, idx_receiver)

        edge_mlp_inputs = tf.concat(
            [h_sender, zb_sender, h_receiver, zb_receiver, edge_attributes],
            axis=-1,
        )
        edge_messages = self.edge_message_mlp(edge_mlp_inputs)

        num_nodes = tf.shape(node_features)[0]
        aggregated_messages = tf.math.unsorted_segment_sum(
            data=edge_messages, segment_ids=idx_receiver, num_segments=num_nodes
        )

        node_update_mlp_inputs = tf.concat([h_t, zb_cell, aggregated_messages], axis=-1)
        update_value = self.node_update_mlp(node_update_mlp_inputs)

        if self.predict_delta:
            h_next = h_t + update_value
        else:
            h_next = update_value
        h_next = tf.maximum(h_next, 0.0)
        return h_next

    def get_config(self):
        config = super(SWEGNNLayer, self).get_config()
        config.update(
            {
                "edge_mlp_units": self.edge_mlp_units,
                "node_mlp_units": self.node_mlp_units,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "predict_delta": self.predict_delta,
            }
        )
        return config


def SWEGNN(
    input_node_feat_dim,
    input_edge_feat_dim,
    output_dim=1,
    num_message_passing_steps=1,
    edge_mlp_units=[64, 32],
    node_mlp_units=[32],
    predict_delta=True,
    l2_reg=1e-5,
):
    """
    Builds the Hydraulics-Based Graph Neural Network (SWEGNN) model.
    Corrected to use keras.layers.Concatenate for symbolic tensors.
    """
    node_features_input = Input(
        shape=(input_node_feat_dim,), name="node_features_input", dtype=tf.float32
    )
    edge_index_input = Input(shape=(2,), name="edge_index_input", dtype=tf.int32)
    edge_attributes_input = Input(
        shape=(input_edge_feat_dim,), name="edge_attributes_input", dtype=tf.float32
    )

    processed_node_features = node_features_input

    for i in range(num_message_passing_steps):
        h_next_step = SWEGNNLayer(
            edge_mlp_units=edge_mlp_units,
            node_mlp_units=node_mlp_units,
            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
            predict_delta=predict_delta,
            name=f"swegnn_layer_{i}",  # Unique name for the layer in the loop
        )([processed_node_features, edge_index_input, edge_attributes_input])

        # Extract static features (zb_cell and others) from the *current* processed_node_features
        # These features should not be affected by h_next_step directly, but carried over.
        current_zb_cell = processed_node_features[
            :, 1:2
        ]  # Assuming zb is the second feature

        if (
            input_node_feat_dim > 2
        ):  # If there are other static features beyond h and zb
            other_static_features = processed_node_features[:, 2:]
            processed_node_features = Concatenate(
                axis=-1, name=f"concatenate_nodes_{i}"
            )(
                [  # Unique name for Concatenate
                    h_next_step,
                    current_zb_cell,
                    other_static_features,
                ]
            )
        else:  # Only h and zb
            processed_node_features = Concatenate(
                axis=-1, name=f"concatenate_nodes_{i}"
            )(
                [h_next_step, current_zb_cell]  # Unique name for Concatenate
            )

    final_predicted_h = processed_node_features[:, 0:output_dim]

    model = Model(
        inputs=[node_features_input, edge_index_input, edge_attributes_input],
        outputs=final_predicted_h,
        name="SWEGNN_Model",
    )
    return model


if __name__ == "__main__":
    # python -m surgenet.swegnn
    input_node_dim_example = 2
    input_edge_dim_example = 3
    output_dim_example = 1

    SWEGNN_model = SWEGNN(
        input_node_feat_dim=input_node_dim_example,
        input_edge_feat_dim=input_edge_dim_example,
        output_dim=output_dim_example,
        num_message_passing_steps=17,
        edge_mlp_units=[32, 16],
        node_mlp_units=[16],
        predict_delta=True,
        l2_reg=1e-5,
    )
    SWEGNN_model.summary()

    # Dummy data for testing compilation (optional)
    # num_nodes = 10
    # num_edges = 20
    # dummy_node_features = tf.random.uniform(shape=(num_nodes, input_node_dim_example))
    # dummy_edge_index = tf.stack([
    #     tf.random.uniform(shape=(num_edges,), minval=0, maxval=num_nodes, dtype=tf.int32),
    #     tf.random.uniform(shape=(num_edges,), minval=0, maxval=num_nodes, dtype=tf.int32)
    # ], axis=0)
    # dummy_edge_attributes = tf.random.uniform(shape=(num_edges, input_edge_dim_example))
    # dummy_target_h = tf.random.uniform(shape=(num_nodes, output_dim_example))

    # SWEGNN_model.compile(optimizer='adam', loss='mse')
    # print("\nModel compiled successfully.")
