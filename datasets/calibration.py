import numpy as np

def transform_to_common_frame(
    signals: np.ndarray,
    sensor_ids: np.ndarray,
    calibration_params: dict
) -> np.ndarray:
    poses = calibration_params.get('sensor_poses', {})
    transformed = signals.copy()

    for sid in np.unique(sensor_ids):
        key_str = str(int(sid))
        if key_str in poses:
            pose = poses[key_str]
        elif int(sid) in poses:
            pose = poses[int(sid)]
        else:
            raise KeyError(
                f"Sensor pose for sensor_id {sid} not found "
                f"(tried '{key_str}' and {int(sid)})"
            )
        R = np.asarray(pose['R'], dtype=np.float32)
        t = np.asarray(pose['t'], dtype=np.float32)
        mask = (sensor_ids == sid)
        coords = signals[mask, :3]
        transformed[mask, :3] = coords.dot(R.T) + t

    return transformed


def calibrate_uncertainties(
    uncertainties: np.ndarray,
    sensor_ids: np.ndarray,
    calibration_params: dict
) -> np.ndarray:
    """
    Calibrate per-point uncertainty estimates per sensor.

    Args:
        uncertainties: np.ndarray of shape (N,), raw uncertainty per point.
        sensor_ids: np.ndarray of shape (N,) with sensor_id for each point.
        calibration_params: dict containing:
            - 'noise_model': dict mapping sensor_id (as string or int) -> float scale
            - 'hetero_model': optional callable(uncertainties: np.ndarray, scale: float) -> np.ndarray
    Returns:
        calibrated: np.ndarray same shape as uncertainties.
    """
    noise_model = calibration_params.get('noise_model', {})
    hetero = calibration_params.get('hetero_model', None)
    calibrated = np.empty_like(uncertainties, dtype=np.float32)
    for sid in np.unique(sensor_ids):
        key_str = str(int(sid))
        if key_str in noise_model:
            scale = noise_model[key_str]
        elif int(sid) in noise_model:
            scale = noise_model[int(sid)]
        else:
            scale = 1.0
        mask = (sensor_ids == sid)
        raw_unc = uncertainties[mask]
        if hetero is not None:
            cal_unc = hetero(raw_unc, scale)
        else:
            cal_unc = raw_unc * scale
        calibrated[mask] = cal_unc
    return calibrated
