"""
Graph dataset creation for autoregressive GNN training from ADCIRC NetCDF files.
"""

import tensorflow as tf
import xarray as xr
import numpy as np
import os

# Placeholder for actual graph representation, e.g., TensorFlow GNN's GraphTensor
# from tensorflow_gnn import GraphTensor # Or your preferred GNN library


def create_adcirc_graph_dataset(
    file_pattern: str,
    node_feature_vars: list[str],
    edge_index_var: str,  # Name of the variable in NetCDF holding edge connectivity (e.g., element_connectivity or a precomputed dual graph)
    target_vars: list[str],
    num_input_steps: int = 1,
    num_target_steps: int = 1,  # For multi-step ahead prediction
    batch_size: int = 32,
    shuffle_buffer_size: int = 1000,
    prefetch_buffer_size: tf.data.AUTOTUNE = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """
    Creates a TensorFlow Dataset for autoregressive GNN training from ADCIRC NetCDF files.

    Args:
        file_pattern: Glob pattern to find NetCDF files (e.g., "/path/to/adcirc_runs/*.nc").
        node_feature_vars: List of variable names in the NetCDF to be used as node features.
                           These are expected to be time-varying.
        edge_index_var: Variable name in the NetCDF representing the graph connectivity (e.g., element triangles,
                        from which a dual graph's edges can be derived, or precomputed edge indices).
                        This is assumed to be static across time for a given mesh.
        target_vars: List of variable names in the NetCDF to be used as target node features
                     (e.g., 'zeta' for water surface elevation).
        num_input_steps: Number of past time steps to use as input node features.
        num_target_steps: Number of future time steps to predict.
        batch_size: Batch size for the dataset.
        shuffle_buffer_size: Buffer size for shuffling the data. If 0, no shuffling.
        prefetch_buffer_size: Buffer size for prefetching.

    Returns:
        A tf.data.Dataset yielding tuples of (input_graph_features, target_graph_features).
        The exact structure of input_graph_features will depend on how you represent
        graphs (e.g., a dictionary of node sets, edge sets, and context, or a GraphTensor).
        target_graph_features will typically be a tensor of future node values.
    """
    filepaths = tf.io.gfile.glob(file_pattern)
    if not filepaths:
        raise ValueError(f"No files found for pattern: {file_pattern}")

    print(f"Found {len(filepaths)} NetCDF files.")

    # --- 1. Define a generator function to load and process data ---
    def _data_generator():
        for filepath in filepaths:
            try:
                # Using xarray to open NetCDF, can also use netCDF4 library
                # ds = xr.open_dataset(filepath, engine="netcdf4") # Specify engine if needed
                # For ADCIRC, often files are grouped (e.g., fort.63.nc, fort.64.nc)
                # If dealing with a single file per simulation run time series:
                with xr.open_dataset(filepath, engine="netcdf4") as ds:
                    # --- a. Extract Static Graph Structure (Edges) ---
                    # This needs to be adapted based on how your dual graph is stored.
                    # Example: If edge_index_var stores [2, num_edges] tensor
                    # Or if it's element connectivity from which you derive dual graph edges.
                    # For this example, let's assume it's precomputed and available.
                    # This might be loaded once if it's the same for all files, or per file.
                    if edge_index_var not in ds:
                        print(
                            f"Warning: Edge index variable '{edge_index_var}' not found in {filepath}. Skipping."
                        )
                        continue
                    edge_indices = ds[
                        edge_index_var
                    ].values  # Shape: [2, num_edges] or similar
                    # Ensure correct dtype for TensorFlow GNN if used
                    edge_indices = tf.convert_to_tensor(edge_indices, dtype=tf.int32)

                    # --- b. Extract Time-Varying Node Features and Targets ---
                    # Assuming node features and targets are variables with dimensions (time, num_nodes)
                    # Concatenate specified node feature variables
                    input_node_data_list = []
                    for var_name in node_feature_vars:
                        if var_name not in ds:
                            print(
                                f"Warning: Node feature variable '{var_name}' not found in {filepath}. Skipping this variable."
                            )
                            continue
                        # Ensure data is (time, num_nodes, feature_dim_for_var)
                        data = ds[var_name].values
                        if data.ndim == 2:  # (time, num_nodes)
                            data = data[..., np.newaxis]  # Add feature dimension
                        input_node_data_list.append(data)

                    if not input_node_data_list:
                        print(
                            f"No node feature variables found or loaded for {filepath}. Skipping file."
                        )
                        continue
                    all_input_node_features = np.concatenate(
                        input_node_data_list, axis=-1
                    )  # (time, num_nodes, num_node_features)

                    target_node_data_list = []
                    for var_name in target_vars:
                        if var_name not in ds:
                            print(
                                f"Warning: Target variable '{var_name}' not found in {filepath}. Skipping this variable."
                            )
                            continue
                        data = ds[var_name].values
                        if data.ndim == 2:  # (time, num_nodes)
                            data = data[..., np.newaxis]
                        target_node_data_list.append(data)

                    if not target_node_data_list:
                        print(
                            f"No target variables found or loaded for {filepath}. Skipping file."
                        )
                        continue
                    all_target_node_features = np.concatenate(
                        target_node_data_list, axis=-1
                    )  # (time, num_nodes, num_target_features)

                    num_timesteps = all_input_node_features.shape[0]
                    num_nodes = all_input_node_features.shape[1]

                    # --- c. Create Input/Target Sequences for Autoregression ---
                    for t in range(
                        num_timesteps - num_input_steps - num_target_steps + 1
                    ):
                        input_start_idx = t
                        input_end_idx = t + num_input_steps
                        target_start_idx = input_end_idx
                        target_end_idx = target_start_idx + num_target_steps

                        current_input_node_features = all_input_node_features[
                            input_start_idx:input_end_idx, ...
                        ]  # (num_input_steps, num_nodes, num_node_features)
                        current_target_node_features = all_target_node_features[
                            target_start_idx:target_end_idx, ...
                        ]  # (num_target_steps, num_nodes, num_target_features)

                        # --- d. Construct Graph Representation for GNN ---
                        # This is a placeholder. You'll need to adapt this to your GNN library's expected input format.
                        # For tf_geometric or spektral, it might be a tuple of (node_features, edge_index, edge_features)
                        # For TensorFlow GNN (TF-GNN), you'd create a GraphTensor.

                        # Example: Simple dictionary representation (adapt as needed)
                        # If num_input_steps > 1, you might flatten/process time into features or handle it in the GNN
                        if num_input_steps == 1:
                            # Squeeze the time dimension if only one input step, common for many GNNs
                            processed_input_node_features = tf.convert_to_tensor(
                                current_input_node_features[0], dtype=tf.float32
                            )
                        else:
                            # Handle multiple input steps, e.g., flatten or keep as sequence
                            # Flattening: (num_nodes, num_input_steps * num_node_features)
                            # processed_input_node_features = tf.reshape(current_input_node_features, [num_nodes, -1])
                            # Or keep as is and let the GNN handle the time dimension
                            processed_input_node_features = tf.convert_to_tensor(
                                current_input_node_features, dtype=tf.float32
                            )

                        # For TF-GNN, you might structure it like this:
                        # input_graph = tfgnn.GraphTensor.from_pieces(
                        #     node_sets={
                        #         "mesh_nodes": tfgnn.NodeSet.from_fields(
                        #             sizes=[num_nodes],
                        #             features={'features': processed_input_node_features}
                        #         )
                        #     },
                        #     edge_sets={
                        #         "mesh_edges": tfgnn.EdgeSet.from_fields(
                        #             sizes=[tf.shape(edge_indices)[1]],
                        #             features={}, # Add edge features if any
                        #             adjacency=tfgnn.Adjacency.from_indices(
                        #                 source=("mesh_nodes", edge_indices[0, :]),
                        #                 target=("mesh_nodes", edge_indices[1, :])
                        #             )
                        #         )
                        #     }
                        # )
                        # yield input_graph, tf.convert_to_tensor(current_target_node_features, dtype=tf.float32)

                        # Simplified yield for now, assuming GNN takes node features and edge indices
                        # The GNN model will need to know how to interpret these.
                        # Input structure: (node_features_at_t, edge_indices)
                        # Target structure: node_features_at_t+1...t+num_target_steps
                        yield (
                            processed_input_node_features,
                            edge_indices,
                        ), tf.convert_to_tensor(
                            current_target_node_features, dtype=tf.float32
                        )

            except Exception as e:
                print(f"Error processing file {filepath}: {e}")
                continue
            # finally:
            # if 'ds' in locals() and ds is not None: # Ensure ds is defined
            # ds.close() # xarray closes with 'with' statement, but good practice if not using 'with'

    # --- 2. Determine Output Signature ---
    # This needs to precisely match what the _data_generator yields.
    # It's crucial for tf.data.Dataset.from_generator.
    # Get a sample output to infer shapes and types. This is a bit fragile.
    # A more robust way is to predefine them if known.

    # Create a temporary generator to get one sample for signature
    temp_gen = _data_generator()
    try:
        sample_input, sample_target = next(temp_gen)
        sample_input_node_features, sample_edge_indices = sample_input

        # If using TF-GNN GraphTensor, the signature would be (graph_tensor_spec, target_spec)
        # For the simplified version:
        output_signature = (
            (
                tf.TensorSpec(
                    shape=sample_input_node_features.shape,
                    dtype=sample_input_node_features.dtype,
                ),  # Node features
                tf.TensorSpec(
                    shape=sample_edge_indices.shape, dtype=sample_edge_indices.dtype
                ),  # Edge indices
            ),
            tf.TensorSpec(
                shape=sample_target.shape, dtype=sample_target.dtype
            ),  # Target node features
        )
        del temp_gen  # clean up
        del sample_input
        del sample_target
        del sample_input_node_features
        del sample_edge_indices

    except StopIteration:
        raise ValueError(
            "The data generator did not produce any data. Check file paths, variable names, and data processing logic."
        )
    except Exception as e:
        raise ValueError(f"Could not determine output signature from generator: {e}")

    # --- 3. Create the TensorFlow Dataset ---
    dataset = tf.data.Dataset.from_generator(
        _data_generator, output_signature=output_signature
    )

    # --- 4. Shuffle, Batch, and Prefetch ---
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    return dataset


if __name__ == "__main__":
    # --- Example Usage ---
    # Note: You'll need to create some dummy NetCDF files or point to your actual data
    # and adjust variable names accordingly.

    # Create dummy NetCDF data for demonstration
    def create_dummy_netcdf(filepath, num_timesteps=20, num_nodes=50, num_edges=100):
        if os.path.exists(filepath):
            return  # Don't recreate if it exists

        times = np.arange(num_timesteps)
        nodes = np.arange(num_nodes)

        # Simulate some node features (e.g., velocity_u, velocity_v, elevation)
        zeta = np.random.rand(num_timesteps, num_nodes).astype(np.float32)
        u_vel = np.random.rand(num_timesteps, num_nodes).astype(np.float32)

        # Simulate edge connectivity (dual graph)
        # Typically, for a triangular mesh, the dual graph connects centers of adjacent elements.
        # Here, we just create random connections between nodes for simplicity.
        # In a real scenario, this would come from the mesh definition.
        # Shape: [2, num_edges] where row 0 is source, row 1 is target
        edge_idx = np.random.randint(0, num_nodes, size=(2, num_edges), dtype=np.int32)

        ds = xr.Dataset(
            {
                "zeta": (("time", "node"), zeta),
                "u_vel": (("time", "node"), u_vel),
                "dual_edge_indices": (
                    ("two", "num_edges"),
                    edge_idx,
                ),  # "two" is a common dimension name for pairs
            },
            coords={
                "time": times,
                "node": nodes,
                "two": [0, 1],
                "num_edges": np.arange(num_edges),
            },
        )
        ds.to_netcdf(filepath)
        print(f"Created dummy NetCDF: {filepath}")

    # Create a couple of dummy files
    DUMMY_DATA_DIR = "dummy_adcirc_data"
    os.makedirs(DUMMY_DATA_DIR, exist_ok=True)
    create_dummy_netcdf(os.path.join(DUMMY_DATA_DIR, "run1.nc"))
    create_dummy_netcdf(os.path.join(DUMMY_DATA_DIR, "run2.nc"))

    try:
        print("\nAttempting to create dataset...")
        # Adjust these parameters based on your NetCDF structure
        # For the dummy data:
        node_features = ["u_vel"]  # Using u_vel as an example input feature
        target_features = ["zeta"]  # Predicting zeta
        edge_var = "dual_edge_indices"  # How we named it in the dummy file

        dataset = create_adcirc_graph_dataset(
            file_pattern=os.path.join(DUMMY_DATA_DIR, "*.nc"),
            node_feature_vars=node_features,
            edge_index_var=edge_var,
            target_vars=target_features,
            num_input_steps=3,  # Use 3 previous steps as input
            num_target_steps=2,  # Predict 2 steps ahead
            batch_size=4,
            shuffle_buffer_size=10,  # Small for demo
        )

        print("\nSuccessfully created dataset. Taking one batch to inspect...")
        for i, ((input_feats, edges), targets) in enumerate(
            dataset.take(2)
        ):  # Take 2 batches
            print(f"\nBatch {i+1}:")
            print(
                f"  Input Node Features Shape: {input_feats.shape}, Dtype: {input_feats.dtype}"
            )  # (batch, num_input_steps, num_nodes, num_node_features)
            print(
                f"  Edge Indices Shape: {edges.shape}, Dtype: {edges.dtype}"
            )  # (batch, 2, num_edges)
            print(
                f"  Target Node Features Shape: {targets.shape}, Dtype: {targets.dtype}"
            )  # (batch, num_target_steps, num_nodes, num_target_features)
            # If num_input_steps was 1 and squeezed:
            # print(f"  Input Node Features Shape: {input_feats.shape}") # (batch, num_nodes, num_node_features)

        # Example of how you might adapt for TensorFlow GNN GraphTensor (conceptual)
        # print("\n--- Conceptual TF-GNN GraphTensor structure (not fully implemented in example) ---")
        # If the generator yielded GraphTensors:
        # for i, (input_graph_tensor, targets) in enumerate(dataset.take(1)):
        #     print(f"Batch {i+1}:")
        #     print(f"  Input GraphTensor: {input_graph_tensor.node_sets['mesh_nodes'].features.shape}")
        #     print(f"  Target Node Features Shape: {targets.shape}")

    except ValueError as ve:
        print(f"ValueError during dataset creation or iteration: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up dummy data
        # import shutil
        # if os.path.exists(DUMMY_DATA_DIR):
        #     shutil.rmtree(DUMMY_DATA_DIR)
        #     print(f"\nCleaned up dummy data directory: {DUMMY_DATA_DIR}")
        pass
