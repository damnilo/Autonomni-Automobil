import numpy as np

class ObservationBuilder:

    def build(self, env, raw_obs, info):

        lidar = self.extract_lidar(raw_obs) / 50.0

        speed = np.array([info.get("speed", 0.0)]) / 120.0

        heading = np.array([self.compute_heading_error(info)]) / np.pi

        lane_offset = np.array([self.compute_lane_offset(info)]) / 5.0

        return np.concatenate([
            lidar, speed, heading, lane_offset
        ]).astype(np.float32)
    
    def extract_lidar(self, raw_obs):

        if isinstance(raw_obs, dict):
            lidar = raw_obs.get("lidar", raw_obs.get("cloud_points", np.array([])))
            return np.array(lidar, dtype=np.float32).flatten()

        return np.array(raw_obs, dtype=np.float32).flatten()
    
    def compute_heading_error(self, info):
        return info.get("heading_diff", 0.0)
    
    def compute_lane_offset(self, info):
        return info.get("lateral", 0.0)
    
    def obs_size(self, raw_obs: np.ndarray, info: dict) -> int:
        return len(self.build(raw_obs, info))