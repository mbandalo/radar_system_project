# datasets/radar_scenes.py

import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging 

logger = logging.getLogger(__name__)

from datasets.calibration import transform_to_common_frame, calibrate_uncertainties

class RadarScenesDataset(Dataset):
    def __init__(self, data_dir: str, calibration_params: dict, num_classes: int, transforms=None): 
        self.data_dir = data_dir
        self.cal_params = calibration_params
        self.num_classes = num_classes
        self.transforms = transforms

        self.radar_path = os.path.join(self.data_dir, 'radar_data.h5')
        self.scenes_path = os.path.join(self.data_dir, 'scenes.json')
        logger.info(f"Dataset initialized for data_dir: {self.data_dir}")
        logger.info(f"Looking for radar_data.h5 at: {self.radar_path}")
        logger.info(f"Looking for scenes.json at: {self.scenes_path}")

        if not os.path.exists(self.radar_path):
            raise FileNotFoundError(f"Missing radar_data.h5 in {self.data_dir}")
        if not os.path.exists(self.scenes_path):
            raise FileNotFoundError(f"Missing scenes.json in {self.data_dir}")

        with open(self.scenes_path, 'r') as f:
            self.scenes_raw_dict = json.load(f) 
        
        self.scenes = self.scenes_raw_dict
        
        if not self.scenes:
            raise RuntimeError(f"No valid scene entries in {self.scenes_path}")
        logger.info(f"Found {len(self.scenes)} valid scene entries in {self.scenes_path}.")


        with h5py.File(self.radar_path, 'r') as h5f:
            self.raw_data_combined = h5f['radar_data'][:] 
        logger.info(f"Loaded {len(self.raw_data_combined)} raw radar points from {self.radar_path}")

        self.sensor_ids = np.array(list(self.cal_params.get('sensor_poses', {}).keys()), dtype=self.raw_data_combined['sensor_id'].dtype)
        logger.info(f"Expected sensor IDs from config: {self.sensor_ids}")

        self.frames = []
        for combined_scene_key, meta in self.scenes.items(): 
            if 'radar_indices' in meta:
                start, end = meta['radar_indices']
                segment = self.raw_data_combined[start:end] 
            else:
                logger.error(f"Scene '{combined_scene_key}' has no 'radar_indices' in its metadata. Cannot safely load frame. Skipping.")
                continue 

            if len(segment) == 0:
                logger.warning(f"No raw radar points found for scene '{combined_scene_key}'. Skipping.")
                continue

            segment = segment[np.isin(segment['sensor_id'], self.sensor_ids)]
            
            if len(segment) == 0:
                logger.warning(f"No points remaining after sensor_id filter for scene '{combined_scene_key}'. Skipping frame.")
                continue

            points = np.stack([
                segment['x_cc'], segment['y_cc'], segment['vr_compensated'],
                segment['range_sc'], segment['azimuth_sc'], segment['rcs']
            ], axis=1).astype(np.float32)

            if 'uncertainty' in segment.dtype.names:
                uncertainties = segment['uncertainty'].astype(np.float32)
            else:
                uncertainties = np.zeros(points.shape[0], dtype=np.float32)

            points = transform_to_common_frame(points, segment['sensor_id'].astype(np.int64), self.cal_params)
            uncertainties = calibrate_uncertainties(uncertainties, segment['sensor_id'].astype(np.int64), self.cal_params)

            point_labels = segment['label_id'].astype(np.int64)
            
            if self.transforms:
                points, uncertainties, point_labels = self.transforms(points, uncertainties, point_labels)

            original_timestamp = meta.get('original_timestamp_from_source_file', -1) 
            if original_timestamp == -1: 
                try:
                    original_timestamp = int(combined_key.rsplit('_', 1)[-1]) 
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse timestamp from combined_key '{combined_key}'. Using 0.")
                    original_timestamp = 0 

            self.frames.append((points, uncertainties, point_labels, original_timestamp))

        if not self.frames:
            raise RuntimeError(f"No frames extracted from the combined data.")
        else:
            logger.info(f"\nSuccessfully extracted {len(self.frames)} frames from the combined dataset.")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        points, uncertainties, point_labels, timestamp = self.frames[idx]
        return (
            torch.from_numpy(points),
            torch.from_numpy(uncertainties),
            torch.from_numpy(point_labels),
            timestamp
        )