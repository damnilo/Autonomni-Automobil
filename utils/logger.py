import csv
import os
from datetime import datetime


class Logger:
    """
    Logs training and evaluation metrics to:
        - Console (stdout)
        - CSV file  (logs/training_TIMESTAMP.csv)

    Training columns:
        episode, reward, epsilon, avg_loss, steps

    Evaluation columns:
        episode, reward, success, collision
    """

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        train_path = os.path.join(log_dir, f"training_{timestamp}.csv")
        eval_path  = os.path.join(log_dir, f"evaluation_{timestamp}.csv")

        self._train_file = open(train_path, "w", newline="")
        self._eval_file  = open(eval_path,  "w", newline="")

        self._train_writer = csv.writer(self._train_file)
        self._eval_writer  = csv.writer(self._eval_file)

        self._train_writer.writerow(
            ["episode", "reward", "epsilon", "avg_loss", "global_step"]
        )
        self._eval_writer.writerow(
            ["episode", "reward", "success", "collision"]
        )

        print(f"[Logger] Training log  → {train_path}")
        print(f"[Logger] Evaluation log → {eval_path}")

    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode: int,
        reward: float,
        epsilon: float,
        avg_loss: float,
        global_step: int = 0,
    ):
        self._train_writer.writerow(
            [episode, f"{reward:.2f}", f"{epsilon:.4f}", f"{avg_loss:.6f}", global_step]
        )
        self._train_file.flush()

        print(
            f"[Train] Ep {episode:>4d} | "
            f"Reward {reward:>8.2f} | "
            f"ε {epsilon:.3f} | "
            f"Loss {avg_loss:.5f} | "
            f"Step {global_step}"
        )

    def log_episode_result(
        self,
        episode: int,
        reward: float,
        out_of_road: bool,
        success: bool,
        collision: bool,
    ):
        self._eval_writer.writerow(
            [episode, f"{reward:.2f}", int(success), int(collision), int(out_of_road)]
        )
        self._eval_file.flush()

        status = "✓ SUCCESS" if success else ("✗ CRASH" if collision or out_of_road else "— timeout")
        print(
            f"[Eval]  Ep {episode:>4d} | "
            f"Reward {reward:>8.2f} | "
            f"{status}"
        )

    def close(self):
        self._train_file.close()
        self._eval_file.close()