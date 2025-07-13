import os
import json
import h5py
import numpy as np
import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_radar_scenes(
    base_data_dir: str,
    sequences_to_merge: list,
    output_dir_name: str = "sequence_combined"
):
    """
    Merges multiple RadarScenes sequences into a single combined sequence.
    Recalculates radar_indices for the combined dataset.
    """
    logger.info(f"Initiating merge for sequences: {sequences_to_merge}")
    logger.info(f"Base data directory: {base_data_dir}")

    output_path = os.path.join(base_data_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Combined data output directory: {output_path}")

    all_raw_data_segments_for_concat = []
    combined_scenes_metadata = {}
    current_raw_data_offset = 0

    for seq_name in tqdm.tqdm(sequences_to_merge, desc="Processing Sequences"):
        seq_path = os.path.join(base_data_dir, seq_name)
        radar_path = os.path.join(seq_path, 'radar_data.h5')
        scenes_path = os.path.join(seq_path, 'scenes.json')

        if not os.path.exists(radar_path) or not os.path.exists(scenes_path):
            logger.warning(f"Skipping sequence {seq_name}: Missing data files.")
            continue

        with open(scenes_path, 'r') as f:
            raw_scenes_seq = json.load(f)
        frames_dict_seq = raw_scenes_seq.get('scenes', raw_scenes_seq)

        with h5py.File(radar_path, 'r') as h5f:
            raw_data_seq = h5f['radar_data'][:]

        logger.info(f"Loaded {len(raw_data_seq)} points from {seq_name}. {len(frames_dict_seq)} scenes found.")

        all_raw_data_segments_for_concat.append(raw_data_seq)

        for ts_str, meta in frames_dict_seq.items():
            try:
                ts_int = int(ts_str)
            except (TypeError, ValueError):
                logger.warning(f"Skipping non-numeric scene key '{ts_str}' in {seq_name}.")
                continue

            new_scene_key = f"{seq_name}_{ts_int}"
            new_meta = meta.copy()

            if 'radar_indices' in meta:
                start_idx, end_idx = meta['radar_indices']
                new_meta['radar_indices'] = [current_raw_data_offset + start_idx, current_raw_data_offset + end_idx]
            else:
                logger.error(f"Scene {ts_int} in {seq_name} lacks 'radar_indices'. Slicing may be inaccurate.")
            
            new_meta['original_timestamp_from_source_file'] = ts_int
            new_meta['original_sequence_name'] = seq_name

            combined_scenes_metadata[new_scene_key] = new_meta
            
        current_raw_data_offset += len(raw_data_seq)

    if not all_raw_data_segments_for_concat:
        logger.error("No data loaded from specified sequences. Exiting.")
        return

    logger.info("Concatenating raw radar data arrays.")
    concatenated_raw_data = np.concatenate(all_raw_data_segments_for_concat, axis=0)
    logger.info(f"Total points in combined dataset: {len(concatenated_raw_data)}")

    combined_radar_path = os.path.join(output_path, 'radar_data.h5')
    with h5py.File(combined_radar_path, 'w') as f:
        f.create_dataset('radar_data', data=concatenated_raw_data, dtype=concatenated_raw_data.dtype)
    logger.info(f"Combined radar_data.h5 saved to: {combined_radar_path}")

    combined_scenes_path = os.path.join(output_path, 'scenes.json')
    with open(combined_scenes_path, 'w') as f:
        json.dump(combined_scenes_metadata, f, indent=4)
    logger.info(f"Combined scenes.json saved to: {combined_scenes_path}")

    logger.info("Merge operation complete.")

if __name__ == '__main__':
    RADAR_SCENES_BASE_DIR = ""
    
    SEQUENCES_TO_MERGE = [
        
    ]
    
    OUTPUT_SEQUENCE_DIR_NAME = "sequence_merged_subset_for_training" 

    if not SEQUENCES_TO_MERGE:
        logger.error("No sequences specified for merge.")
    else:
        merge_radar_scenes(RADAR_SCENES_BASE_DIR, SEQUENCES_TO_MERGE, OUTPUT_SEQUENCE_DIR_NAME)