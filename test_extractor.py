from Extractor import extract_dataset


def print_one_map(path, id):
    dataset, audio_cache = extract_dataset(path)
    map_data = dataset[id]
    
    audio_path = map_data["audio_path"]
    print("audio path", audio_path)
    
    audio_key = map_data["audio_key"]
    print("audio key:", audio_key)
    
    audio_filename = map_data["audio_filename"]
    print("audio file name:", audio_filename)
    
if __name__ == "__main__":
    
    folder_path = "songs"
    print_one_map(folder_path, "690982")
    
    
    