"""Model implementation of mSWE-GNN (Multiscale Shallow Water Equations Graph Neural Network). May not work with older versions of TensorFlow/Keras due to API changes. Initially based on Gemini 2.5 Pro's implementation. Need to further assess the faithfullness of this implementation."""

import tensorflow as tf
from tensorflow import keras
from keras import layers


def create_mlp(hidden_units, activation="relu", final_activation=None, name=None):
    """Creates a Multi-Layer Perceptron (MLP)."""
    mlp_layers = []
    for i, units in enumerate(hidden_units):
        if units is None:
            raise ValueError(
                f"MLP layer units cannot be None. Problem in MLP: {name}, layer {i}"
            )
        mlp_layers.append(
            layers.Dense(
                units,
                activation=activation,
                name=(f"{name}_dense_{i}" if name else f"dense_{i}"),
            )
        )
    if final_activation is not None and mlp_layers:
        last_layer_activation = keras.activations.get(final_activation)
        mlp_layers[-1].activation = last_layer_activation
    return keras.Sequential(mlp_layers, name=name)


class StaticNodeEncoder(layers.Layer):
    """Encodes static node features. Shared across all scales. (phi_s)"""

    def __init__(
        self, embedding_dim, mlp_hidden_units, name="static_node_encoder", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.mlp = create_mlp(
            list(mlp_hidden_units) + [embedding_dim], name=f"{name}_phi_s_mlp"
        )

    def call(self, static_node_features, training=False):
        return self.mlp(static_node_features, training=training)


class DynamicNodeEncoder(layers.Layer):
    """Encodes initial dynamic node features. Applied at finest scale. (phi_d)"""

    def __init__(
        self, embedding_dim, mlp_hidden_units, name="dynamic_node_encoder", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.mlp = create_mlp(
            list(mlp_hidden_units) + [embedding_dim], name=f"{name}_phi_d_mlp"
        )

    def call(self, dynamic_node_features_current_t, training=False):
        return self.mlp(dynamic_node_features_current_t, training=training)


class EdgeEncoder(layers.Layer):
    """Encodes edge features. Shared across all edges. (phi_epsilon)"""

    def __init__(self, embedding_dim, mlp_hidden_units, name="edge_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mlp = create_mlp(
            list(mlp_hidden_units) + [embedding_dim], name=f"{name}_phi_epsilon_mlp"
        )

    def call(self, edge_features, training=False):
        return self.mlp(edge_features, training=training)


class mSWEGNNLayer(layers.Layer):
    """Implements a single GNN message passing layer from mSWE-GNN. (Eq. 3, 4)"""

    def __init__(
        self, output_dim, mlp_psi_hidden_units, name="mswegnn_layer", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.psi_mlp = create_mlp(
            list(mlp_psi_hidden_units) + [output_dim], name=f"{self.name}_psi_mlp"
        )
        self.W = layers.Dense(output_dim, use_bias=False, name=f"{self.name}_W_matrix")

    def call(self, inputs, training=False):
        h_d_prev, h_s, edge_features_embedded, adjacency_info = inputs
        sender_indices, receiver_indices = adjacency_info

        h_di_prev = tf.gather(h_d_prev, sender_indices)
        h_dj_prev = tf.gather(h_d_prev, receiver_indices)
        h_si = tf.gather(h_s, sender_indices)
        h_sj = tf.gather(h_s, receiver_indices)

        psi_input = tf.concat(
            [h_si, h_sj, h_di_prev, h_dj_prev, edge_features_embedded], axis=-1
        )
        psi_output = self.psi_mlp(psi_input, training=training)

        diff_h_d = h_dj_prev - h_di_prev
        s_ij = psi_output * diff_h_d

        num_nodes = tf.shape(h_d_prev)[0]
        aggregated_fluxes = tf.math.unsorted_segment_sum(
            s_ij, receiver_indices, num_segments=num_nodes
        )

        transformed_fluxes = self.W(aggregated_fluxes)
        h_d_new = h_d_prev + transformed_fluxes
        return h_d_new


class MeanPoolDownsample(layers.Layer):
    """Downsamples node features using mean pooling. (Eq. 5)"""

    def __init__(self, target_num_segments, name="mean_pool_downsample", **kwargs):
        super().__init__(name=name, **kwargs)
        self.target_num_segments = tf.constant(
            target_num_segments, dtype=tf.int32
        )  # Store as tensor

    def call(self, inputs):
        fine_scale_h_d, prolongation_map_fine_to_coarse = inputs

        num_fine_nodes = tf.shape(fine_scale_h_d)[0]
        num_features = tf.shape(fine_scale_h_d)[1]

        # Function to return if the target number of segments is 0
        def empty_coarse_output():
            return tf.zeros([0, num_features], dtype=fine_scale_h_d.dtype)

        # Function to return if the input (fine_scale_h_d) is empty
        def if_fine_empty():
            # Output should be zeros, shaped by target_num_segments
            return tf.zeros(
                [self.target_num_segments, num_features], dtype=fine_scale_h_d.dtype
            )

        # Function to process normally if inputs are not empty and target segments > 0
        def if_fine_not_empty():
            # Sort segment IDs and gather data accordingly for segment_mean
            sorted_indices = tf.argsort(prolongation_map_fine_to_coarse)

            sorted_fine_scale_h_d = tf.gather(fine_scale_h_d, sorted_indices)
            sorted_prolongation_map = tf.gather(
                prolongation_map_fine_to_coarse, sorted_indices
            )

            return tf.math.segment_mean(
                data=sorted_fine_scale_h_d,
                segment_ids=sorted_prolongation_map,
                num_segments=self.target_num_segments,  # Use the stored target_num_segments
            )

        # Main control flow using tf.cond
        return tf.cond(
            tf.equal(self.target_num_segments, 0),
            true_fn=empty_coarse_output,
            false_fn=lambda: tf.cond(
                tf.equal(num_fine_nodes, 0),
                true_fn=if_fine_empty,
                false_fn=if_fine_not_empty,
            ),
        )


class LearnableUpsample(layers.Layer):
    """Upsamples node features using a learnable mechanism. (Eq. 6)"""

    def __init__(
        self,
        output_dim_psi_mlp,
        mlp_psi_up_hidden_units,
        name="learnable_upsample",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.psi_up_mlp = create_mlp(
            list(mlp_psi_up_hidden_units) + [output_dim_psi_mlp],
            name=f"{self.name}_psi_up_mlp",
        )

    def call(self, inputs, training=False):
        (
            h_dk_coarse,
            h_si_fine,
            h_sk_coarse,
            h_di_fine_current,
            (fine_node_indices_for_edges, coarse_node_indices_for_edges),
        ) = inputs

        h_dk_coarse_gathered = tf.gather(h_dk_coarse, coarse_node_indices_for_edges)
        h_si_fine_gathered = tf.gather(h_si_fine, fine_node_indices_for_edges)
        h_sk_coarse_gathered = tf.gather(h_sk_coarse, coarse_node_indices_for_edges)
        h_di_fine_current_gathered = tf.gather(
            h_di_fine_current, fine_node_indices_for_edges
        )

        psi_up_input = tf.concat(
            [
                h_si_fine_gathered,
                h_sk_coarse_gathered,
                h_di_fine_current_gathered,
                h_dk_coarse_gathered,
            ],
            axis=-1,
        )
        psi_up_output = self.psi_up_mlp(psi_up_input, training=training)

        contributions = psi_up_output * h_dk_coarse_gathered

        num_fine_nodes = tf.shape(h_si_fine)[0]
        h_di_fine_upsampled = tf.math.unsorted_segment_sum(
            contributions, fine_node_indices_for_edges, num_segments=num_fine_nodes
        )
        return h_di_fine_upsampled


class OutputDecoder(layers.Layer):
    """Decodes final embeddings to hydraulic variables. (Eq. 8)"""

    def __init__(
        self,
        output_dynamic_features,
        mlp_phi_hidden_units,
        history_length,
        name="output_decoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.output_dynamic_features = output_dynamic_features
        self.temporal_weighting = layers.Dense(
            1, use_bias=False, name=f"{self.name}_w_p_temporal_weights"
        )
        self.phi_mlp = create_mlp(
            list(mlp_phi_hidden_units) + [output_dynamic_features],
            name=f"{self.name}_phi_output_mlp",
        )
        self.projection_U_history_layer = None

    def call(self, inputs, training=False):
        final_h_d_fine, U_history_fine = inputs

        permuted_U_history = tf.transpose(U_history_fine, perm=[0, 2, 1])
        weighted_U_processed = self.temporal_weighting(permuted_U_history)
        weighted_U_squeezed = tf.squeeze(weighted_U_processed, axis=-1)

        phi_output = self.phi_mlp(final_h_d_fine, training=training)

        projected_weighted_U = weighted_U_squeezed
        if weighted_U_squeezed.shape[-1] != self.output_dynamic_features:
            if self.projection_U_history_layer is None:
                self.projection_U_history_layer = layers.Dense(
                    self.output_dynamic_features,
                    name=f"{self.name}_projection_U_history",
                )
            projected_weighted_U = self.projection_U_history_layer(weighted_U_squeezed)

        output_sum = projected_weighted_U + phi_output
        predictions = tf.nn.relu(output_sum)
        return predictions


class mSWEGNNModel(keras.Model):
    def __init__(
        self,
        num_scales,
        gnn_layers_per_block,
        h_s_dim,
        h_d_dim,
        h_edge_dim,
        mlp_hidden_units,
        output_dynamic_features,
        history_length,
        # Added num_nodes_per_scale_list to pass to MeanPoolDownsample
        num_nodes_per_scale_list,
        name="mswegnn_model",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_scales = num_scales
        self.h_d_dim = h_d_dim
        self.mlp_hidden_units = list(mlp_hidden_units)

        self.static_node_encoder = StaticNodeEncoder(
            h_s_dim, self.mlp_hidden_units, name=f"{self.name}_static_encoder"
        )
        self.dynamic_node_encoder = DynamicNodeEncoder(
            h_d_dim, self.mlp_hidden_units, name=f"{self.name}_dynamic_encoder"
        )
        self.edge_encoder = EdgeEncoder(
            h_edge_dim, self.mlp_hidden_units, name=f"{self.name}_edge_encoder"
        )

        self.encoder_gnn_blocks = []
        self.downsamplers = []
        for i in range(num_scales - 1):
            block_layers = [
                mSWEGNNLayer(
                    self.h_d_dim,
                    self.mlp_hidden_units,
                    name=f"{self.name}_encoder_scale{i}_gnn{j}",
                )
                for j in range(gnn_layers_per_block)
            ]
            self.encoder_gnn_blocks.append(block_layers)

            # Pass the number of nodes for the coarser scale (i+1) as target_num_segments
            coarse_nodes_count_for_downsampler = num_nodes_per_scale_list[i + 1]
            self.downsamplers.append(
                MeanPoolDownsample(
                    target_num_segments=coarse_nodes_count_for_downsampler,
                    name=f"{self.name}_downsampler_scale{i}",
                )
            )

        self.bottleneck_gnn_block = [
            mSWEGNNLayer(
                self.h_d_dim,
                self.mlp_hidden_units,
                name=f"{self.name}_bottleneck_gnn{j}",
            )
            for j in range(gnn_layers_per_block)
        ]

        self.decoder_gnn_blocks = []
        self.upsamplers = []
        for i in range(num_scales - 1):
            self.upsamplers.append(
                LearnableUpsample(
                    self.h_d_dim,
                    self.mlp_hidden_units,
                    name=f"{self.name}_upsampler_scale{i}",
                )
            )
            block_layers = [
                mSWEGNNLayer(
                    self.h_d_dim,
                    self.mlp_hidden_units,
                    name=f"{self.name}_decoder_scale{i}_gnn{j}",
                )
                for j in range(gnn_layers_per_block)
            ]
            self.decoder_gnn_blocks.append(block_layers)

        self.output_decoder = OutputDecoder(
            output_dynamic_features,
            self.mlp_hidden_units,
            history_length,
            name=f"{self.name}_output_decoder",
        )

    def call(self, inputs, training=False):
        static_node_features_ps = inputs["static_node_features_per_scale"]
        dynamic_node_features_hist_finest = inputs[
            "dynamic_node_features_history_finest"
        ]
        edge_features_ps = inputs["edge_features_per_scale"]
        adjacencies_ps = inputs["adjacencies_per_scale"]
        prolong_maps_down = inputs["prolongation_maps_down"]
        prolong_maps_up = inputs["prolongation_maps_up"]

        h_s_all_scales = [
            self.static_node_encoder(s_feat, training=training)
            for s_feat in static_node_features_ps
        ]
        h_edge_all_scales = [
            self.edge_encoder(e_feat, training=training) for e_feat in edge_features_ps
        ]

        current_dynamic_features_finest = dynamic_node_features_hist_finest[:, -1, :]
        h_d_current_scale = self.dynamic_node_encoder(
            current_dynamic_features_finest, training=training
        )

        skip_connections_data = []

        for i in range(self.num_scales - 1):
            h_s_scale = h_s_all_scales[i]
            h_edge_scale = h_edge_all_scales[i]
            adj_scale = adjacencies_ps[i]

            current_block_h_d = h_d_current_scale
            for gnn_layer in self.encoder_gnn_blocks[i]:
                current_block_h_d = gnn_layer(
                    (current_block_h_d, h_s_scale, h_edge_scale, adj_scale),
                    training=training,
                )
            h_d_current_scale = current_block_h_d

            skip_connections_data.append(h_d_current_scale)
            h_d_current_scale = self.downsamplers[i](
                (h_d_current_scale, prolong_maps_down[i])
            )

        coarsest_scale_idx = self.num_scales - 1
        h_s_coarsest = h_s_all_scales[coarsest_scale_idx]
        h_edge_coarsest = h_edge_all_scales[coarsest_scale_idx]
        adj_coarsest = adjacencies_ps[coarsest_scale_idx]

        current_bottleneck_h_d = h_d_current_scale
        for gnn_layer in self.bottleneck_gnn_block:
            current_bottleneck_h_d = gnn_layer(
                (current_bottleneck_h_d, h_s_coarsest, h_edge_coarsest, adj_coarsest),
                training=training,
            )
        h_d_current_scale = current_bottleneck_h_d

        for i in range(self.num_scales - 1):
            scale_idx_fine = self.num_scales - 2 - i
            scale_idx_coarse = self.num_scales - 1 - i

            h_s_fine_scale_for_up = h_s_all_scales[scale_idx_fine]
            h_s_coarse_scale_for_up = h_s_all_scales[scale_idx_coarse]
            skip_data_fine_scale = skip_connections_data[scale_idx_fine]
            prolong_map_c_to_f_edges = prolong_maps_up[i]

            h_d_upsampled = self.upsamplers[i](
                (
                    h_d_current_scale,
                    h_s_fine_scale_for_up,
                    h_s_coarse_scale_for_up,
                    skip_data_fine_scale,
                    prolong_map_c_to_f_edges,
                ),
                training=training,
            )

            h_d_current_scale = layers.Add(
                name=f"{self.name}_decoder_add_skip_{scale_idx_fine}"
            )([h_d_upsampled, skip_data_fine_scale])

            h_s_scale_decoder = h_s_all_scales[scale_idx_fine]
            h_edge_scale_decoder = h_edge_all_scales[scale_idx_fine]
            adj_scale_decoder = adjacencies_ps[scale_idx_fine]

            current_block_h_d = h_d_current_scale
            for gnn_layer in self.decoder_gnn_blocks[i]:
                current_block_h_d = gnn_layer(
                    (
                        current_block_h_d,
                        h_s_scale_decoder,
                        h_edge_scale_decoder,
                        adj_scale_decoder,
                    ),
                    training=training,
                )
            h_d_current_scale = current_block_h_d

        predictions = self.output_decoder(
            (h_d_current_scale, dynamic_node_features_hist_finest), training=training
        )
        return predictions


if __name__ == "__main__":
    num_nodes_per_scale_list = [200, 100, 50]
    # Example with a scale having 0 nodes to test MeanPoolDownsample robustness
    # num_nodes_per_scale_list = [200, 0, 50] # This would be problematic for subsequent layers if not handled
    # num_nodes_per_scale_list = [200, 100, 0] # Test 0 nodes at coarsest trainable scale

    num_edges_per_scale_list = [300, 150, 70]
    # if num_nodes_per_scale_list = [200, 0, 50], then num_edges_per_scale_list should reflect this, e.g. [300, 0, 70]

    model_params = {
        "num_scales": 3,
        "gnn_layers_per_block": 2,
        "h_s_dim": 32,
        "h_d_dim": 64,
        "h_edge_dim": 16,
        "mlp_hidden_units": [64, 32],
        "output_dynamic_features": 2,
        "history_length": 3,
        "num_nodes_per_scale_list": num_nodes_per_scale_list,  # Add this
    }

    static_feature_dim_example = 3
    dynamic_feature_dim_initial_example = 2
    edge_feature_dim_example = 1

    mswe_gnn_model = mSWEGNNModel(**model_params)

    dummy_static_feats_ps = [
        (
            tf.random.normal(
                (num_nodes_per_scale_list[i], static_feature_dim_example),
                dtype=tf.float32,
            )
            if num_nodes_per_scale_list[i] > 0
            else tf.zeros((0, static_feature_dim_example), dtype=tf.float32)
        )
        for i in range(model_params["num_scales"])
    ]
    dummy_dyn_hist_finest = (
        tf.random.normal(
            (
                num_nodes_per_scale_list[0],
                model_params["history_length"],
                dynamic_feature_dim_initial_example,
            ),
            dtype=tf.float32,
        )
        if num_nodes_per_scale_list[0] > 0
        else tf.zeros(
            (0, model_params["history_length"], dynamic_feature_dim_initial_example),
            dtype=tf.float32,
        )
    )

    dummy_edge_feats_ps = [
        (
            tf.random.normal(
                (num_edges_per_scale_list[i], edge_feature_dim_example),
                dtype=tf.float32,
            )
            if num_edges_per_scale_list[i] > 0
            else tf.zeros((0, edge_feature_dim_example), dtype=tf.float32)
        )
        for i in range(model_params["num_scales"])
    ]

    dummy_adj_ps = []
    for i in range(model_params["num_scales"]):
        if num_nodes_per_scale_list[i] > 0 and num_edges_per_scale_list[i] > 0:
            senders = tf.random.uniform(
                (num_edges_per_scale_list[i],),
                maxval=num_nodes_per_scale_list[i],
                dtype=tf.int32,
            )
            receivers = tf.random.uniform(
                (num_edges_per_scale_list[i],),
                maxval=num_nodes_per_scale_list[i],
                dtype=tf.int32,
            )
        else:  # Handle cases with zero nodes or edges
            senders = tf.zeros((0,), dtype=tf.int32)
            receivers = tf.zeros((0,), dtype=tf.int32)
        dummy_adj_ps.append((senders, receivers))

    dummy_prolong_down = []
    for i in range(model_params["num_scales"] - 1):
        fine_nodes_count = num_nodes_per_scale_list[i]
        coarse_nodes_count = num_nodes_per_scale_list[i + 1]
        if fine_nodes_count > 0 and coarse_nodes_count > 0:
            # prolongation map must have fine_nodes_count entries
            # each entry is an index into the coarse_nodes_count nodes
            dummy_prolong_down.append(
                tf.random.uniform(
                    (fine_nodes_count,), maxval=coarse_nodes_count, dtype=tf.int32
                )
            )
        else:
            # If fine_nodes_count is 0, map is empty.
            # If coarse_nodes_count is 0 (but fine_nodes_count > 0), all fine nodes map to "non-existent" coarse nodes.
            # The MeanPoolDownsample layer handles target_num_segments=0 by returning empty.
            # The prolongation map itself just needs to be a placeholder of correct shape if fine_nodes_count > 0.
            # For tf.argsort to not error on this map if it's used by MeanPoolDownsample when target_num_segments=0
            # (though it shouldn't be), providing a map of zeros is okay.
            dummy_prolong_down.append(tf.zeros((fine_nodes_count,), dtype=tf.int32))

    dummy_prolong_up = []
    for i in range(model_params["num_scales"] - 1):
        idx_coarse = model_params["num_scales"] - 1 - i
        idx_fine = model_params["num_scales"] - 2 - i

        num_nodes_at_fine_scale_for_up = num_nodes_per_scale_list[idx_fine]
        num_nodes_at_coarse_scale_for_up = num_nodes_per_scale_list[idx_coarse]

        num_inter_edges = (
            num_nodes_at_fine_scale_for_up  # Example: one "up-edge" per fine node
        )

        if num_nodes_at_fine_scale_for_up > 0 and num_nodes_at_coarse_scale_for_up > 0:
            fine_indices = tf.range(num_nodes_at_fine_scale_for_up, dtype=tf.int32)
            coarse_indices = tf.random.uniform(
                (num_inter_edges,),
                maxval=num_nodes_at_coarse_scale_for_up,
                dtype=tf.int32,
            )
            dummy_prolong_up.append((fine_indices, coarse_indices))
        else:  # If either scale has 0 nodes, no upsampling edges.
            dummy_prolong_up.append(
                (tf.zeros((0,), dtype=tf.int32), tf.zeros((0,), dtype=tf.int32))
            )

    dummy_inputs = {
        "static_node_features_per_scale": dummy_static_feats_ps,
        "dynamic_node_features_history_finest": dummy_dyn_hist_finest,
        "edge_features_per_scale": dummy_edge_feats_ps,
        "adjacencies_per_scale": dummy_adj_ps,
        "prolongation_maps_down": dummy_prolong_down,
        "prolongation_maps_up": dummy_prolong_up,
    }

    print("Building model with dummy inputs...")
    try:
        _ = mswe_gnn_model(dummy_inputs, training=False)
        mswe_gnn_model.summary()
        print(
            f"mSWE-GNN Model with {model_params['num_scales']} scales instantiated and built."
        )
    except Exception as e:
        print(f"Error during model build or summary: {e}")
        import traceback

        traceback.print_exc()
