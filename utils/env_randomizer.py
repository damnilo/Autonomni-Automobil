import random

def get_random_metadrive_config():
    map_elements = ["S", "C", "R", "O", "T", "X"]
    random_map = "".join(random.choices(map_elements, k=random.randint(3, 6)))

    return {
        "use_render": False,
        "manual_control": False,
        "traffic_density": round(random.uniform(0.05, 0.4), 2),
        "num_scenarios": 1,
        "start_seed": random.randint(0, 10000),
        "map": random_map,
        "daytime": random.choice(["08:00", "12:00", "17:30", "20:00"]),
        "accident_prob": round(random.uniform(0.0, 0.2), 2),
        "vehicle_config": {
            "show_lidar": True,
            # "vehicle_model": "default", # Neki MetaDrive verzije zahtevaju specifične modele, ostavi default ako pravi problem
        },
        "agent_policy": None,
        # ISPRAVLJENI KLJUČEVI:
        "on_continuous_line_done": True, 
        "crash_vehicle_done": True,      # Sudar sa drugim vozilom
        "crash_object_done": True,       # Sudar sa objektom (ogradom, čunjem)
        "out_of_road_done": True,        # Kraj ako skreneš sa puta
    }