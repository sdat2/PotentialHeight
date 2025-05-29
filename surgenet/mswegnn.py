"""Model implementation of mSWE-GNN (Multiscale Shallow Water Equations Graph Neural Network).
May not work with older versions of TensorFlow/Keras due to API changes.
Initially based on Gemini 2.5 Pro's implementation.
Need to further assess the faithfullness of this implementation."""

import tensorflow as tf
from tensorflow import keras
from keras import (
    layers,
)  # Keep this for now, but tf.keras.layers is preferred for type hints
from typing import List, Optional, Tuple, Dict, Any, Union

def create_mlp(
    hidden_units: List[int],
    activation: str = "relu",
    final_activation: Optional[str] = None,
    name: Optional[str] = None,
) -> keras.Sequential:  # Or tf.keras.Sequential
    """Creates a Multi-Layer Perceptron (MLP)."""
    mlp_layers: List[tf.keras.layers.Dense] = []
    for i, units in enumerate(hidden_units):
        if units is None:  # Should not happen if hidden_units is List[int]
            raise ValueError(
                f"MLP layer units cannot be None. Problem in MLP: {name}, layer {i}"
            )
        mlp_layers.append(
            layers.Dense(  # Or tf.keras.layers.Dense
                units,
                activation=activation,
                name=(f"{name}_dense_{i}" if name else f"dense_{i}"),
            )
        )
    if final_activation is not None and mlp_layers:
        last_layer_activation = keras.activations.get(
            final_activation
        )  # Or tf.keras.activations.get
        mlp_layers[-1].activation = last_layer_activation
    return keras.Sequential(mlp_layers, name=name)  # Or tf.keras.Sequential


class StaticNodeEncoder(layers.Layer):  # Or tf.keras.layers.Layer
    """Encodes static node features. Shared across all scales. (phi_s)"""

    mlp: keras.Sequential  # Or tf.keras.Sequential

    def __init__(
        self,
        embedding_dim: int,
        mlp_hidden_units: List[int],
        name: str = "static_node_encoder",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.mlp = create_mlp(
            list(mlp_hidden_units) + [embedding_dim], name=f"{name}_phi_s_mlp"
        )

    def call(
        self, static_node_features: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        return self.mlp(static_node_features, training=training)


class DynamicNodeEncoder(layers.Layer):  # Or tf.keras.layers.Layer
    """Encodes initial dynamic node features. Applied at finest scale. (phi_d)"""

    mlp: keras.Sequential  # Or tf.keras.Sequential

    def __init__(
        self,
        embedding_dim: int,
        mlp_hidden_units: List[int],
        name: str = "dynamic_node_encoder",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.mlp = create_mlp(
            list(mlp_hidden_units) + [embedding_dim], name=f"{name}_phi_d_mlp"
        )

    def call(
        self, dynamic_node_features_current_t: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        return self.mlp(dynamic_node_features_current_t, training=training)


class EdgeEncoder(layers.Layer):  # Or tf.keras.layers.Layer
    """Encodes edge features. Shared across all edges. (phi_epsilon)"""

    mlp: keras.Sequential  # Or tf.keras.Sequential

    def __init__(
        self,
        embedding_dim: int,
        mlp_hidden_units: List[int],
        name: str = "edge_encoder",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.mlp = create_mlp(
            list(mlp_hidden_units) + [embedding_dim], name=f"{name}_phi_epsilon_mlp"
        )

    def call(self, edge_features: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.mlp(edge_features, training=training)


class mSWEGNNLayer(layers.Layer):  # Or tf.keras.layers.Layer
    """Implements a single GNN message passing layer from mSWE-GNN. (Eq. 3, 4)"""

    output_dim: int
    psi_mlp: keras.Sequential  # Or tf.keras.Sequential
    W: layers.Dense  # Or tf.keras.layers.Dense

    def __init__(
        self,
        output_dim: int,
        mlp_psi_hidden_units: List[int],
        name: str = "mswegnn_layer",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.psi_mlp = create_mlp(
            list(mlp_psi_hidden_units) + [output_dim], name=f"{self.name}_psi_mlp"
        )
        self.W = layers.Dense(  # Or tf.keras.layers.Dense
            output_dim, use_bias=False, name=f"{self.name}_W_matrix"
        )

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        training: bool = False,
    ) -> tf.Tensor:
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


class MeanPoolDownsample(layers.Layer):  # Or tf.keras.layers.Layer
    """Downsamples node features using mean pooling. (Eq. 5)"""

    target_num_segments: tf.Tensor

    def __init__(
        self,
        target_num_segments: int,
        name: str = "mean_pool_downsample",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.target_num_segments = tf.constant(
            target_num_segments, dtype=tf.int32
        )  # Store as tensor

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        fine_scale_h_d, prolongation_map_fine_to_coarse = inputs

        num_fine_nodes = tf.shape(fine_scale_h_d)[0]
        num_features = tf.shape(fine_scale_h_d)[1]

        # Function to return if the target number of segments is 0
        def empty_coarse_output() -> tf.Tensor:
            return tf.zeros([0, num_features], dtype=fine_scale_h_d.dtype)

        # Function to return if the input (fine_scale_h_d) is empty
        def if_fine_empty() -> tf.Tensor:
            # Output should be zeros, shaped by target_num_segments
            return tf.zeros(
                [self.target_num_segments, num_features], dtype=fine_scale_h_d.dtype
            )

        # Function to process normally if inputs are not empty and target segments > 0
        def if_fine_not_empty() -> tf.Tensor:
            # Sort segment IDs and gather data accordingly for segment_mean
            sorted_indices = tf.argsort(prolongation_map_fine_to_coarse)

            sorted_fine_scale_h_d = tf.gather(fine_scale_h_d, sorted_indices)
            sorted_prolongation_map = tf.gather(
                prolongation_map_fine_to_coarse, sorted_indices
            )

            return tf.math.segment_mean(
                data=sorted_fine_scale_h_d,
                segment_ids=sorted_prolongation_map,
                num_segments=self.target_num_segments,
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


class LearnableUpsample(layers.Layer):  # Or tf.keras.layers.Layer
    """Upsamples node features using a learnable mechanism. (Eq. 6)"""

    psi_up_mlp: keras.Sequential  # Or tf.keras.Sequential

    def __init__(
        self,
        output_dim_psi_mlp: int,
        mlp_psi_up_hidden_units: List[int],
        name: str = "learnable_upsample",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.psi_up_mlp = create_mlp(
            list(mlp_psi_up_hidden_units) + [output_dim_psi_mlp],
            name=f"{self.name}_psi_up_mlp",
        )

    def call(
        self,
        inputs: Tuple[
            tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Tuple[tf.Tensor, tf.Tensor]
        ],
        training: bool = False,
    ) -> tf.Tensor:
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


class OutputDecoder(layers.Layer):  # Or tf.keras.layers.Layer
    """Decodes final embeddings to hydraulic variables. (Eq. 8)"""

    output_dynamic_features: int
    temporal_weighting: layers.Dense  # Or tf.keras.layers.Dense
    phi_mlp: keras.Sequential  # Or tf.keras.Sequential
    projection_U_history_layer: Optional[
        layers.Dense
    ]  # Or Optional[tf.keras.layers.Dense]

    def __init__(
        self,
        output_dynamic_features: int,
        mlp_phi_hidden_units: List[int],
        history_length: int,  # history_length is not directly used in this version of __init__
        name: str = "output_decoder",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.output_dynamic_features = output_dynamic_features
        self.temporal_weighting = layers.Dense(  # Or tf.keras.layers.Dense
            1, use_bias=False, name=f"{self.name}_w_p_temporal_weights"
        )
        self.phi_mlp = create_mlp(
            list(mlp_phi_hidden_units) + [output_dynamic_features],
            name=f"{self.name}_phi_output_mlp",
        )
        self.projection_U_history_layer = None

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        final_h_d_fine, U_history_fine = inputs

        permuted_U_history = tf.transpose(U_history_fine, perm=[0, 2, 1])
        weighted_U_processed = self.temporal_weighting(permuted_U_history)
        weighted_U_squeezed = tf.squeeze(weighted_U_processed, axis=-1)

        phi_output = self.phi_mlp(final_h_d_fine, training=training)

        projected_weighted_U = weighted_U_squeezed
        if weighted_U_squeezed.shape[-1] != self.output_dynamic_features:
            if self.projection_U_history_layer is None:
                self.projection_U_history_layer = (
                    layers.Dense(  # Or tf.keras.layers.Dense
                        self.output_dynamic_features,
                        name=f"{self.name}_projection_U_history",
                    )
                )
            projected_weighted_U = self.projection_U_history_layer(weighted_U_squeezed)

        output_sum = projected_weighted_U + phi_output
        predictions = tf.nn.relu(output_sum)
        return predictions


InputDictType = Dict[
    str,
    Union[
        List[tf.Tensor],  # For _per_scale lists of Tensors
        tf.Tensor,  # For _finest Tensor
        List[
            Tuple[tf.Tensor, tf.Tensor]
        ],  # For adjacencies_per_scale, prolongation_maps_up
    ],
]


class mSWEGNNModel(keras.Model):  # Or tf.keras.Model
    num_scales: int
    h_d_dim: int
    mlp_hidden_units: List[int]
    static_node_encoder: StaticNodeEncoder
    dynamic_node_encoder: DynamicNodeEncoder
    edge_encoder: EdgeEncoder
    encoder_gnn_blocks: List[List[mSWEGNNLayer]]
    downsamplers: List[MeanPoolDownsample]
    bottleneck_gnn_block: List[mSWEGNNLayer]
    decoder_gnn_blocks: List[List[mSWEGNNLayer]]
    upsamplers: List[LearnableUpsample]
    output_decoder: OutputDecoder

    def __init__(
        self,
        num_scales: int,
        gnn_layers_per_block: int,
        h_s_dim: int,
        h_d_dim: int,
        h_edge_dim: int,
        mlp_hidden_units: List[int],
        output_dynamic_features: int,
        history_length: int,
        num_nodes_per_scale_list: List[int],
        name: str = "mswegnn_model",
        **kwargs: Any,
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
            block_layers: List[mSWEGNNLayer] = [
                mSWEGNNLayer(
                    self.h_d_dim,
                    self.mlp_hidden_units,
                    name=f"{self.name}_encoder_scale{i}_gnn{j}",
                )
                for j in range(gnn_layers_per_block)
            ]
            self.encoder_gnn_blocks.append(block_layers)

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
                    self.h_d_dim,  # output_dim_psi_mlp for upsampler
                    self.mlp_hidden_units,  # mlp_psi_up_hidden_units
                    name=f"{self.name}_upsampler_scale{i}",
                )
            )
            block_layers_dec: List[mSWEGNNLayer] = [
                mSWEGNNLayer(
                    self.h_d_dim,
                    self.mlp_hidden_units,
                    name=f"{self.name}_decoder_scale{i}_gnn{j}",
                )
                for j in range(gnn_layers_per_block)
            ]
            self.decoder_gnn_blocks.append(block_layers_dec)

        self.output_decoder = OutputDecoder(
            output_dynamic_features,
            self.mlp_hidden_units,
            history_length,
            name=f"{self.name}_output_decoder",
        )

    def call(self, inputs: InputDictType, training: bool = False) -> tf.Tensor:
        static_node_features_ps: List[tf.Tensor] = inputs["static_node_features_per_scale"]  # type: ignore
        dynamic_node_features_hist_finest: tf.Tensor = inputs["dynamic_node_features_history_finest"]  # type: ignore
        edge_features_ps: List[tf.Tensor] = inputs["edge_features_per_scale"]  # type: ignore
        adjacencies_ps: List[Tuple[tf.Tensor, tf.Tensor]] = inputs["adjacencies_per_scale"]  # type: ignore
        prolong_maps_down: List[tf.Tensor] = inputs["prolongation_maps_down"]  # type: ignore
        prolong_maps_up: List[Tuple[tf.Tensor, tf.Tensor]] = inputs["prolongation_maps_up"]  # type: ignore

        h_s_all_scales: List[tf.Tensor] = [
            self.static_node_encoder(s_feat, training=training)
            for s_feat in static_node_features_ps
        ]
        h_edge_all_scales: List[tf.Tensor] = [
            self.edge_encoder(e_feat, training=training) for e_feat in edge_features_ps
        ]

        current_dynamic_features_finest: tf.Tensor = dynamic_node_features_hist_finest[
            :, -1, :
        ]
        h_d_current_scale: tf.Tensor = self.dynamic_node_encoder(
            current_dynamic_features_finest, training=training
        )

        skip_connections_data: List[tf.Tensor] = []

        # Encoder path
        for i in range(self.num_scales - 1):
            h_s_scale: tf.Tensor = h_s_all_scales[i]
            h_edge_scale: tf.Tensor = h_edge_all_scales[i]
            adj_scale: Tuple[tf.Tensor, tf.Tensor] = adjacencies_ps[i]

            current_block_h_d: tf.Tensor = h_d_current_scale
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

        # Bottleneck
        coarsest_scale_idx: int = self.num_scales - 1
        h_s_coarsest: tf.Tensor = h_s_all_scales[coarsest_scale_idx]
        h_edge_coarsest: tf.Tensor = h_edge_all_scales[coarsest_scale_idx]
        adj_coarsest: Tuple[tf.Tensor, tf.Tensor] = adjacencies_ps[coarsest_scale_idx]

        current_bottleneck_h_d: tf.Tensor = h_d_current_scale
        for gnn_layer in self.bottleneck_gnn_block:
            current_bottleneck_h_d = gnn_layer(
                (current_bottleneck_h_d, h_s_coarsest, h_edge_coarsest, adj_coarsest),
                training=training,
            )
        h_d_current_scale = current_bottleneck_h_d

        # Decoder path
        for i in range(self.num_scales - 1):
            scale_idx_fine: int = self.num_scales - 2 - i
            scale_idx_coarse: int = self.num_scales - 1 - i

            h_s_fine_scale_for_up: tf.Tensor = h_s_all_scales[scale_idx_fine]
            h_s_coarse_scale_for_up: tf.Tensor = h_s_all_scales[scale_idx_coarse]
            skip_data_fine_scale: tf.Tensor = skip_connections_data[scale_idx_fine]
            prolong_map_c_to_f_edges: Tuple[tf.Tensor, tf.Tensor] = prolong_maps_up[i]

            h_d_upsampled: tf.Tensor = self.upsamplers[i](
                (
                    h_d_current_scale,  # h_dk_coarse
                    h_s_fine_scale_for_up,  # h_si_fine
                    h_s_coarse_scale_for_up,  # h_sk_coarse
                    skip_data_fine_scale,  # h_di_fine_current (from skip connection)
                    prolong_map_c_to_f_edges,
                ),
                training=training,
            )

            h_d_current_scale = layers.Add(  # Or tf.keras.layers.Add
                name=f"{self.name}_decoder_add_skip_{scale_idx_fine}"
            )([h_d_upsampled, skip_data_fine_scale])

            h_s_scale_decoder: tf.Tensor = h_s_all_scales[scale_idx_fine]
            h_edge_scale_decoder: tf.Tensor = h_edge_all_scales[scale_idx_fine]
            adj_scale_decoder: Tuple[tf.Tensor, tf.Tensor] = adjacencies_ps[
                scale_idx_fine
            ]

            current_block_h_d_dec: tf.Tensor = h_d_current_scale
            for gnn_layer in self.decoder_gnn_blocks[i]:
                current_block_h_d_dec = gnn_layer(
                    (
                        current_block_h_d_dec,
                        h_s_scale_decoder,
                        h_edge_scale_decoder,
                        adj_scale_decoder,
                    ),
                    training=training,
                )
            h_d_current_scale = current_block_h_d_dec

        predictions: tf.Tensor = self.output_decoder(
            (h_d_current_scale, dynamic_node_features_hist_finest), training=training
        )
        return predictions


if __name__ == "__main__":
    num_nodes_per_scale_list_ex: List[int] = [200, 100, 50]
    num_edges_per_scale_list_ex: List[int] = [300, 150, 70]

    model_params_ex: Dict[str, Any] = {
        "num_scales": 3,
        "gnn_layers_per_block": 2,
        "h_s_dim": 32,
        "h_d_dim": 64,
        "h_edge_dim": 16,
        "mlp_hidden_units": [64, 32],
        "output_dynamic_features": 2,
        "history_length": 3,
        "num_nodes_per_scale_list": num_nodes_per_scale_list_ex,
    }

    static_feature_dim_example: int = 3
    dynamic_feature_dim_initial_example: int = 2
    edge_feature_dim_example: int = 1

    mswe_gnn_model_ex = mSWEGNNModel(**model_params_ex)

    dummy_static_feats_ps_ex: List[tf.Tensor] = [
        (
            tf.random.normal(
                (num_nodes_per_scale_list_ex[i], static_feature_dim_example),
                dtype=tf.float32,
            )
            if num_nodes_per_scale_list_ex[i] > 0
            else tf.zeros((0, static_feature_dim_example), dtype=tf.float32)
        )
        for i in range(model_params_ex["num_scales"])
    ]
    dummy_dyn_hist_finest_ex: tf.Tensor = (
        tf.random.normal(
            (
                num_nodes_per_scale_list_ex[0],
                model_params_ex["history_length"],
                dynamic_feature_dim_initial_example,
            ),
            dtype=tf.float32,
        )
        if num_nodes_per_scale_list_ex[0] > 0
        else tf.zeros(
            (0, model_params_ex["history_length"], dynamic_feature_dim_initial_example),
            dtype=tf.float32,
        )
    )

    dummy_edge_feats_ps_ex: List[tf.Tensor] = [
        (
            tf.random.normal(
                (num_edges_per_scale_list_ex[i], edge_feature_dim_example),
                dtype=tf.float32,
            )
            if num_edges_per_scale_list_ex[i] > 0
            else tf.zeros((0, edge_feature_dim_example), dtype=tf.float32)
        )
        for i in range(model_params_ex["num_scales"])
    ]

    dummy_adj_ps_ex: List[Tuple[tf.Tensor, tf.Tensor]] = []
    for i in range(model_params_ex["num_scales"]):
        senders_ex: tf.Tensor
        receivers_ex: tf.Tensor
        if num_nodes_per_scale_list_ex[i] > 0 and num_edges_per_scale_list_ex[i] > 0:
            senders_ex = tf.random.uniform(
                (num_edges_per_scale_list_ex[i],),
                maxval=num_nodes_per_scale_list_ex[i],
                dtype=tf.int32,
            )
            receivers_ex = tf.random.uniform(
                (num_edges_per_scale_list_ex[i],),
                maxval=num_nodes_per_scale_list_ex[i],
                dtype=tf.int32,
            )
        else:
            senders_ex = tf.zeros((0,), dtype=tf.int32)
            receivers_ex = tf.zeros((0,), dtype=tf.int32)
        dummy_adj_ps_ex.append((senders_ex, receivers_ex))

    dummy_prolong_down_ex: List[tf.Tensor] = []
    for i in range(model_params_ex["num_scales"] - 1):
        fine_nodes_count_ex: int = num_nodes_per_scale_list_ex[i]
        coarse_nodes_count_ex: int = num_nodes_per_scale_list_ex[i + 1]
        if fine_nodes_count_ex > 0 and coarse_nodes_count_ex > 0:
            dummy_prolong_down_ex.append(
                tf.random.uniform(
                    (fine_nodes_count_ex,), maxval=coarse_nodes_count_ex, dtype=tf.int32
                )
            )
        else:
            dummy_prolong_down_ex.append(
                tf.zeros((fine_nodes_count_ex,), dtype=tf.int32)
            )

    dummy_prolong_up_ex: List[Tuple[tf.Tensor, tf.Tensor]] = []
    for i in range(model_params_ex["num_scales"] - 1):
        idx_coarse_ex: int = model_params_ex["num_scales"] - 1 - i
        idx_fine_ex: int = model_params_ex["num_scales"] - 2 - i

        num_nodes_at_fine_scale_for_up_ex: int = num_nodes_per_scale_list_ex[
            idx_fine_ex
        ]
        num_nodes_at_coarse_scale_for_up_ex: int = num_nodes_per_scale_list_ex[
            idx_coarse_ex
        ]

        num_inter_edges_ex: int = num_nodes_at_fine_scale_for_up_ex

        fine_indices_ex: tf.Tensor
        coarse_indices_ex: tf.Tensor
        if (
            num_nodes_at_fine_scale_for_up_ex > 0
            and num_nodes_at_coarse_scale_for_up_ex > 0
        ):
            fine_indices_ex = tf.range(
                num_nodes_at_fine_scale_for_up_ex, dtype=tf.int32
            )
            coarse_indices_ex = tf.random.uniform(
                (num_inter_edges_ex,),
                maxval=num_nodes_at_coarse_scale_for_up_ex,
                dtype=tf.int32,
            )
            dummy_prolong_up_ex.append((fine_indices_ex, coarse_indices_ex))
        else:
            dummy_prolong_up_ex.append(
                (tf.zeros((0,), dtype=tf.int32), tf.zeros((0,), dtype=tf.int32))
            )

    dummy_inputs_ex: InputDictType = {
        "static_node_features_per_scale": dummy_static_feats_ps_ex,
        "dynamic_node_features_history_finest": dummy_dyn_hist_finest_ex,
        "edge_features_per_scale": dummy_edge_feats_ps_ex,
        "adjacencies_per_scale": dummy_adj_ps_ex,
        "prolongation_maps_down": dummy_prolong_down_ex,
        "prolongation_maps_up": dummy_prolong_up_ex,
    }

    print("Building model with dummy inputs...")
    try:
        _ = mswe_gnn_model_ex(dummy_inputs_ex, training=False)
        mswe_gnn_model_ex.summary()
        print(
            f"mSWE-GNN Model with {model_params_ex['num_scales']} scales instantiated and built."
        )
    except Exception as e:
        print(f"Error during model build or summary: {e}")
        import traceback

        traceback.print_exc()
