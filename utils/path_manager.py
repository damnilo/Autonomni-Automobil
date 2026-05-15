from datetime import datetime
import os

def generate_log_path():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("logs", exist_ok=True)
    PREFIX = 'drive_log_'

    return os.path.join(
        "logs",
        f"{PREFIX}_{timestamp}.json"
    )