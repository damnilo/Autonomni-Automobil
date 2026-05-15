from manual_control.game import Game
from utils.logger import ActionLogger
from utils.path_manager import generate_log_path
from utils.env_randomizer import get_random_metadrive_config
from configs.manual_control_config import MANUAL_CONTROL_CONFIG

def run_dataset_collection():

    log_path = generate_log_path()

    config = {
        **MANUAL_CONTROL_CONFIG,

        "ENV_CONFIG": get_random_metadrive_config()
    }

    game = Game(config)

    game.subscribe_logger(
        ActionLogger(log_path)
    )

    game.start()