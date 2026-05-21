# imports
from pathlib import Path

import librosa
import numpy as np

import pickle

def save_extracted(all_maps, audio_cache, output_path="processed/extracted.pkl"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "all_maps": all_maps,
        "audio_cache": audio_cache
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_extracted(path = "processed/extracted.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["all_maps"], data["audio_cache"]

def extract_audio(path):
    # sr and other configuration comes from chat
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    
    hop_length = 512
    n_fft = 2048
    n_mels = 80
    
    # Mel spectrogram: shape = [n_mels, time_frames]  
    mel = librosa.feature.melspectrogram(
        y = y,
        sr = sr,
        n_fft = n_fft,
        hop_length = hop_length,
        n_mels = n_mels
    )
    
    # convert power values to decibels
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.T
    
    # Onset strength: shape = [time_features]
    onset = librosa.onset.onset_strength(
        y = y,
        sr = sr,
        hop_length = hop_length
    )
    
    duration = librosa.get_duration(y=y, sr=sr)
    
    print("Loaded audio:", path)
    print("mel shape:", mel_db.shape)
    print("onset shape:", onset.shape)
    print("duration:", duration)

    
    return {
        "mel": mel_db,
        "onset": onset,
        "sr": sr,
        "hop_length": hop_length,
        "duration": duration        
    }
    
    
def parse_beatmaps(file):
    beatmap_id = None
    # initialize per map data
    map_data = {
        "audio_path": None,
        "audio_filename": None,
        "mode": None,
        "difficulty": {},
        "timing_points": [],
        "hit_objects": []
    }
    
    # BEGIN PARSING PER-MAP FEATURE 
    with open(file, 'r', encoding="utf-8-sig") as beatmap:
        current_section = ""
        for line in beatmap:
            line = line.strip() # strip each line so we remove \n
            # skip empty lines
            if not line:
                continue
            # extract mapID
            if line.startswith("BeatmapID:"):
                beatmap_id = line.split(":", 1)[1].strip()
                #use beatmap_id later for all_maps construction
            # extract audio file name
            if line.startswith("AudioFilename:"):
                audio_filename = line.split(":", 1)[1].strip()
                map_data["audio_filename"] = audio_filename
            # get gamemode
            if line.startswith("Mode:"):
                mode = line.split(":", 1)[1].strip()
                map_data["mode"] = int(mode)
            # check current section
            if line.startswith("[") and line.endswith("]"):
                current_section = line
                continue
            # get difficulty settings
            if current_section == "[Difficulty]":
                key, value = line.split(":", 1)
                map_data["difficulty"][key] = float(value)
            
            # get timing
            if current_section == "[TimingPoints]":
                parts = line.split(",")
                
                time = float(parts[0])
                beat_length = float(parts[1])
                meter = int(parts[2])
                sampleSet = int(parts[3])
                sampleIndex = int(parts[4])
                volume = int(parts[5])
                uninherited = int(parts[6])
                effects = int(parts[7])
                                    
                timing_point = {
                    "time": time,
                    "beat_length": beat_length,
                    "meter": meter,
                    "sampleSet": sampleSet,
                    "sampleIndex": sampleIndex,
                    "volume": volume,
                    "uninherited": uninherited,
                    "effects": effects
                }
                map_data["timing_points"].append(timing_point)

            # get hit objects
            if current_section == "[HitObjects]":
                parts = line.split(",")
                
                x = int(parts[0])
                y = int(parts[1])
                time_ms = float(parts[2])
                object_type = int(parts[3])
                hitsound = int(parts[4])
                object_params = parts[5:]
                
                hit_object = {
                    "x": x,
                    "y": y,
                    "time_ms": time_ms,
                    "type": object_type,
                    "hitsound": hitsound,
                    "object_params": object_params,
                }  
                map_data["hit_objects"].append(hit_object)
    return map_data, beatmap_id   


def add_audio_frames(map_data, audio_data):
    sr = audio_data["sr"]
    hop_length = audio_data["hop_length"]
    for hit_object in map_data["hit_objects"]:
        time_ms = hit_object["time_ms"]
        audio_frame = int((time_ms / 1000) * sr / hop_length)
        hit_object["audio_frame"] = audio_frame
    return map_data

               
    
def extract_dataset(folder_path="songs"):
    # initiate path
    folder_path = Path(folder_path)
    # Construct data set
    all_maps = {}
    #used for storing all audio
    audio_cache = {}

    # begin looping through the folder and files
    for folder in folder_path.iterdir():
        if not folder.is_dir():
            continue

        for file in folder.glob("*.osu"):
            
            map_data, beatmap_id = parse_beatmaps(file)
            if map_data["mode"] != 0:
                continue
            if beatmap_id is None or beatmap_id == "0":
                beatmap_id = str(file)
            
            # get audio path
            if map_data["audio_filename"] is None:
                print("Missing AudioFilename:", file)
                continue
            audio_path = folder / map_data["audio_filename"]
                
            if not audio_path.exists():
                print("Missing audio:", audio_path)
                continue

            #############################
            audio_key = str(audio_path)
            
            map_data["audio_path"] = audio_key
            map_data["audio_key"] = audio_key
                                
            if audio_key not in audio_cache:
                audio_cache[audio_key] = extract_audio(audio_path)
            
            audio_data = audio_cache[audio_key]  
            map_data = add_audio_frames(map_data, audio_data)
            # put current map into the the grand dataset            
            all_maps[beatmap_id] = map_data    
            
    return all_maps, audio_cache
            
            # THIS MARKS the end of the extraction for features of a map in the song folder

if __name__ == "__main__":
    all_maps, audio_cache = extract_dataset("songs")
    save_extracted(all_maps, audio_cache)
    
    print("maps:", len(all_maps))
    print("audio files:", len(audio_cache))


