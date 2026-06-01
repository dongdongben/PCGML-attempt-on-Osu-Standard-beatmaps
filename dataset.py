# Turns all_maps and audio_cache into PyTorch-ready examples

import pickle
from pathlib import Path

import torch

def save_tensors(maps_tensor, audio_tensor, output_path="processed/tensors.pt"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "maps_tensor": maps_tensor,
        "audio_tensor": audio_tensor,
    }

    torch.save(data, output_path)

def load_tensors(path="processed/tensors.pt"):
    return torch.load(path)

def get_audioTensor(audio_cache):
    songs_tensor = {}
    for key, audio_data in audio_cache.items():
        mel = torch.tensor(audio_data["mel"], dtype=torch.float32)
        onset = torch.tensor(audio_data["onset"], dtype=torch.float32)
        onset = onset.unsqueeze(1)
        
        song_tensor = torch.cat([mel, onset], dim=1)
        songs_tensor[key] = song_tensor
    return songs_tensor

def get_snap_grid(map_data, audio_data, divisor=4):
    uninherited_points = []
    grid = []
    duration = audio_data["duration"]*1000

    for timing_point in map_data["timing_points"]:
        if timing_point["uninherited"] == 1:
            uninherited_points.append(timing_point)
    

    for i, point in enumerate(uninherited_points):        
        start_time = point["time"]
        
        if i+1 < len(uninherited_points):
            end_time = uninherited_points[i+1]["time"]
        else:
            end_time = duration
            
        snap_length = point["beat_length"]/divisor
        
        time = start_time 
        while time < end_time and time <= duration:
            grid.append(time)
            time += snap_length
    return grid

def get_timing_points(map_data):
    uninherited_points = []

    for timing_point in map_data["timing_points"]:
        if timing_point["uninherited"] == 1:
            point = [
                timing_point["time"],
                timing_point["beat_length"],
                timing_point["meter"],
                timing_point["sampleSet"],
                timing_point["sampleIndex"],
                timing_point["volume"],
                timing_point["uninherited"],
                timing_point["effects"],
            ]
            uninherited_points.append(point)
    if not uninherited_points:
        return torch.empty((0, 8), dtype=torch.float32)        

    return torch.tensor(uninherited_points, dtype=torch.float32)

def get_slider_velocity_points(map_data):
    inherited_points = []

    for timing_point in map_data["timing_points"]:
        if timing_point["uninherited"] == 0:
            point = [
                timing_point["time"],
                timing_point["beat_length"],
                timing_point["meter"],
                timing_point["sampleSet"],
                timing_point["sampleIndex"],
                timing_point["volume"],
                timing_point["uninherited"],
                timing_point["effects"],
            ]
            inherited_points.append(point)
    if not inherited_points:
        return torch.empty((0, 8), dtype=torch.float32)        

    return torch.tensor(inherited_points, dtype=torch.float32)
    

def hit_object_type_decoder(object_type):
    object_type = int(object_type)
    is_circle = object_type & 1 != 0
    is_slider = object_type & 2 != 0
    is_new_combo = object_type & 4 != 0
    is_spinner = object_type & 8 != 0
    
    if is_circle:
        real_type = 0
    elif is_slider:
        real_type = 1
    elif is_spinner:
        real_type = 2
    else:
        real_type = -1
    if is_new_combo:
        is_new_combo = 1
    else:
        is_new_combo = 0
    return real_type, is_new_combo

def parse_slider_params(object_params, max_points):
    curve = object_params[0]      # e.g. "B|300:192|350:240"
    repeat = int(object_params[1])
    pixel_length = float(object_params[2])
    
    CURVE_TYPES = {
    "L": 0,
    "P": 1,
    "B": 2,
    "C": 3,
    }
    parts = curve.split("|")
    curve_type = parts[0]         # "B", "L", "P", "C"
    curve_type_id = CURVE_TYPES.get(curve_type, -1)

    points = []
    for point in parts[1:]:
        x, y = point.split(":")
        points.append([float(x) / 512, float(y) / 384])

    point_tensor = torch.zeros((max_points, 2), dtype=torch.float32)
    point_mask = torch.zeros(max_points, dtype=torch.float32)

    for i, point in enumerate(points[:max_points]):
        point_tensor[i] = torch.tensor(point)
        point_mask[i] = 1
    return {
        "curve_type": torch.tensor(curve_type_id, dtype=torch.long),
        "repeat": torch.tensor(repeat, dtype=torch.long),
        "pixel_length": torch.tensor(pixel_length, dtype=torch.float32),
        "points": point_tensor,
        "point_mask": point_mask,
    }

def get_slider_features(map_data, objects, max_points=6):
    slider_features = []

    for i, obj in enumerate(objects):
        kind_id = int(obj[3].item())

        if kind_id == 1:
            object_params = map_data["hit_objects"][i]["object_params"]
            slider_tensor = parse_slider_params(object_params, max_points)
            slider_features.append(slider_tensor)

    return slider_features

def get_objects_tensor(map_data):
    hit_objects = []
    for hit_object in map_data["hit_objects"]:
        real_type, is_new_combo = hit_object_type_decoder(hit_object["type"])
        
        object_features = [
            hit_object["x"] / 512,
            hit_object["y"] / 384,
            hit_object["time_ms"],
            real_type,
            is_new_combo,
        ]
        object_features = torch.tensor(object_features, dtype=torch.float32)
        hit_objects.append(object_features)
       
    return hit_objects

def get_mapTensor(all_maps, audio_cache):
    maps_tensor = {}
    for beatmap_id, map_data in all_maps.items():
        # audio key
        map_tensor = {}
        map_tensor["audio_key"] = map_data["audio_key"]
        
        # difficulty
        difficulty_keys = [
            "HPDrainRate",
            "CircleSize",
            "OverallDifficulty",
            "ApproachRate",
            "SliderMultiplier",
            "SliderTickRate",
        ]
        difficulty = [map_data["difficulty"].get(key, 0.0) for key in difficulty_keys]
        map_tensor["difficulty"] = torch.tensor(difficulty, dtype=torch.float32)
        
        # get snap grid for map
        audio_data = audio_cache[map_data["audio_key"]]
        grid = get_snap_grid(map_data, audio_data)
        map_tensor["grid"] = torch.tensor(grid, dtype=torch.float32)
    
        # object locations
        objects = get_objects_tensor(map_data)
        if objects:
            map_tensor["objects"] = torch.stack(objects)
        else:
            map_tensor["objects"] = torch.empty((0, 5), dtype=torch.float32)
        
        # slider features
        slider_tensor = get_slider_features(map_data, objects)
        map_tensor["slider_features"] = slider_tensor
            
        # uniherited timing points
        uninherited_tp = get_timing_points(map_data)
        map_tensor["uninherited_tp"] = uninherited_tp
        
        # inherited tp
        inherited_tp = get_slider_velocity_points(map_data)
        map_tensor["inherited_tp"] = inherited_tp
        
        maps_tensor[beatmap_id] = map_tensor
        
    return maps_tensor

if __name__ == "__main__":
    input_path = Path("processed/extracted.pkl")
    output_path = Path("processed/tensors.pt")

    with open(input_path, 'rb') as file:
        data = pickle.load(file)
        all_maps =  data["all_maps"]
        audio_cache = data["audio_cache"]

    audio_tensor = get_audioTensor(audio_cache)
    maps_tensor = get_mapTensor(all_maps, audio_cache)
    save_tensors(maps_tensor, audio_tensor, output_path)

    print("saved tensor dataset:", output_path)
    print("maps:", len(maps_tensor))
    print("audio files:", len(audio_tensor))
    
