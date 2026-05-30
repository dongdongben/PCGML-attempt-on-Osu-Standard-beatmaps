# Turns all_maps and audio_cache into PyTorch-ready examples

import pickle
import torch

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
        while time <= end_time and time <= duration:
            grid.append(time)
            time += snap_length
    return grid


def get_mapTensor(all_maps):
    maps_tensor = {}
    for beatmap_id, map_data in all_maps.items():
        # audio key
        map_tensor = {}
        map_tensor["audio_key"] = map_data["audio_key"]
        
        # difficulty
        difficulty = []       
        for values in map_data["difficulty"]:
            difficulty.append(values)
        difficulty = torch.tensor(difficulty, dtype=torch.float32)
        map_tensor["difficulty"] = difficulty
        
        # get snap grid for map
        grid = get_snap_grid(map_data)
        grid = torch.tensor(grid, dtype=torch.float32)
    
        # object locations
        
        # timing points
    return maps_tensor

if __name__ == "__main__":
    with open('extracted.pkl', 'rb') as file:
        data = pickle.load(file)
        all_maps =  data["all_maps"]
        audio_cache = data["audio_cache"]
    
