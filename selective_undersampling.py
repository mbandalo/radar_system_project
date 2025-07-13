import os
import json
import h5py
import numpy as np
import tqdm
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_selective_undersampling(
    input_data_dir: str,
    output_data_dir: str,
    k_neighbors: int = 5,
    random_class11_keep_ratio: float = 0.01,
    dynamic_class_ids: list = None
):
    """
    Performs selective undersampling of static points (Class 11) based on proximity
    to dynamic points, and a random subset of remaining static points.
    """
    logger.info(f"Initiating selective undersampling for {input_data_dir}")
    os.makedirs(output_data_dir, exist_ok=True)

    radar_path = os.path.join(input_data_dir, 'radar_data.h5')
    scenes_path = os.path.join(input_data_dir, 'scenes.json')

    if not os.path.exists(radar_path) or not os.path.exists(scenes_path):
        logger.error(f"Input files not found in {input_data_dir}. Exiting.")
        return

    with h5py.File(radar_path, 'r') as h5f:
        raw_data_combined = h5f['radar_data'][:]
    with open(scenes_path, 'r') as f:
        scenes_metadata_combined = json.load(f)

    logger.info(f"Loaded {len(raw_data_combined)} total points from combined data.")

    new_raw_data_segments = []
    new_scenes_metadata = {}
    new_offset = 0

    if dynamic_class_ids is None:
        dynamic_class_ids = list(range(11))
        logger.info(f"Using default dynamic class IDs: {dynamic_class_ids}")

    for combined_scene_key, meta in tqdm.tqdm(scenes_metadata_combined.items(), desc="Undersampling scenes"):
        if 'radar_indices' not in meta:
            logger.warning(f"Scene '{combined_scene_key}' has no 'radar_indices'. Skipping.")
            continue
        
        start_idx, end_idx = meta['radar_indices']
        scene_segment = raw_data_combined[start_idx:end_idx]

        if len(scene_segment) == 0:
            logger.warning(f"Empty segment for scene '{combined_scene_key}'. Skipping.")
            continue

        points_features = np.stack([scene_segment['x_cc'], scene_segment['y_cc']], axis=1).astype(np.float32)
        point_labels = scene_segment['label_id'].astype(np.int64)

        dynamic_mask = np.isin(point_labels, dynamic_class_ids)
        static_mask = (point_labels == 11)

        dynamic_segment_full = scene_segment[dynamic_mask]
        static_class11_segment_full = scene_segment[static_mask]
        static_class11_points = points_features[static_mask]

        if len(dynamic_segment_full) == 0 or len(static_class11_segment_full) == 0:
            logger.debug(f"Scene '{combined_scene_key}': No dynamic or static points. Retaining all {len(scene_segment)}.")
            new_selected_segment = scene_segment
        else:
            if len(static_class11_points) >= k_neighbors:
                nn_model = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(static_class11_points)
                _, indices = nn_model.kneighbors(points_features[dynamic_mask])
                indices_to_keep_static = np.unique(indices.flatten())
            else:
                indices_to_keep_static = np.arange(len(static_class11_points))
                logger.debug(f"Scene '{combined_scene_key}': Insufficient static points ({len(static_class11_points)}) for {k_neighbors} neighbors. Retaining all static points.")

            boundary_static_segment = static_class11_segment_full[indices_to_keep_static]

            remaining_static_indices = np.setdiff1d(np.arange(len(static_class11_points)), indices_to_keep_static)
            num_to_keep_random = int(len(remaining_static_indices) * random_class11_keep_ratio)
            
            random_static_segment = (
                static_class11_segment_full[np.random.choice(remaining_static_indices, num_to_keep_random, replace=False)]
                if num_to_keep_random > 0 else np.array([], dtype=static_class11_segment_full.dtype)
            )
            
            new_selected_segment = np.concatenate(
                [dynamic_segment_full, boundary_static_segment, random_static_segment],
                axis=0
            )
            logger.debug(f"Scene '{combined_scene_key}': Dynamic: {len(dynamic_segment_full)}, Boundary Static: {len(boundary_static_segment)}, Random Static: {len(random_static_segment)}. Total: {len(new_selected_segment)}")

        if len(new_selected_segment) == 0:
            logger.warning(f"Scene '{combined_scene_key}': No points selected. Skipping frame for new dataset.")
            continue

        new_meta = meta.copy()
        new_meta['radar_indices'] = [new_offset, new_offset + len(new_selected_segment)]
        new_scenes_metadata[combined_scene_key] = new_meta
        
        new_raw_data_segments.append(new_selected_segment)
        new_offset += len(new_selected_segment)

    if not new_raw_data_segments:
        logger.error("No data selected for the undersampled dataset. Exiting.")
        return

    final_concatenated_data = np.concatenate(new_raw_data_segments, axis=0)
    logger.info(f"Total points in new undersampled dataset: {len(final_concatenated_data)}")

    output_radar_path = os.path.join(output_data_dir, 'radar_data.h5')
    output_scenes_path = os.path.join(output_data_dir, 'scenes.json')

    with h5py.File(output_radar_path, 'w') as f:
        f.create_dataset('radar_data', data=final_concatenated_data, dtype=final_concatenated_data.dtype)
    logger.info(f"Undersampled radar_data.h5 saved to: {output_radar_path}")

    with open(output_scenes_path, 'w') as f:
        json.dump(new_scenes_metadata, f, indent=4)
    logger.info(f"Undersampled scenes.json saved to: {output_scenes_path}")

    logger.info("Selective undersampling complete.")

if __name__ == '__main__':
    INPUT_COMBINED_DATA_DIR = ""
    OUTPUT_UNDERSAMPLED_DIR_NAME = "sequence_undersampled_for_training"
    
    OUTPUT_UNDERSAMPLED_FULL_PATH = os.path.join(
        os.path.dirname(INPUT_COMBINED_DATA_DIR), 
        OUTPUT_UNDERSAMPLED_DIR_NAME
    )

    K_NEIGHBORS = 5
    RANDOM_CLASS11_KEEP_RATIO = 0.005
    DYNAMIC_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if not DYNAMIC_CLASS_IDS:
        logger.error("Dynamic class IDs not specified.")
    elif not SEQUENCES_TO_MERGE: 
        logger.error("No sequences specified for merge. Ensure `SEQUENCES_TO_MERGE` is defined if intended to be used here.")
    else:
        perform_selective_undersampling(
            INPUT_COMBINED_DATA_DIR,
            OUTPUT_UNDERSAMPLED_FULL_PATH,
            k_neighbors=K_NEIGHBORS,
            random_class11_keep_ratio=RANDOM_CLASS11_KEEP_RATIO,
            dynamic_class_ids=DYNAMIC_CLASS_IDS
        )