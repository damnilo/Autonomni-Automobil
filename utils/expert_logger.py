# utils/expert_dataset_logger.py

import json
import os


class ExpertDatasetLogger:

    def __init__(self, filepath):

        self.filepath = filepath

        self.records = []

        os.makedirs(
            os.path.dirname(filepath),
            exist_ok=True
        )

    def log(
        self,
        observation,
        steering,
        throttle
    ):

        self.records.append({

            "observation": observation,

            "action_steering": steering,

            "action_throttle": throttle
        })

    def save(self):

        with open(self.filepath, "w") as f:

            json.dump(
                self.records,
                f
            )

        print(
            f"[ExpertDatasetLogger] "
            f"Saved {len(self.records)} samples."
        )