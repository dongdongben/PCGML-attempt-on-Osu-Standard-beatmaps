# PCGML Attempt on osu!standard Beatmaps

An early attempt to generate osu!standard beatmaps with procedural content generation via machine learning.

## Expected Input

The extractor expects a local `songs/` folder containing osu! beatmap folders:

```text
songs/
  beatmapset_folder/
    difficulty_1.osu
    difficulty_2.osu
    audio.mp3
```

Each `.osu` file should declare its audio file through `AudioFilename`.

## Output Structure

```python
all_maps = {
    beatmap_id: map_data,
    ...
}
```

```python
map_data = {
    "audio_path": str,
    "audio_key": str,
    "audio_filename": str,
    "mode": int,
    "difficulty": difficulty,
    "timing_points": [timing_point, ...],
    "hit_objects": [hit_object, ...],
}
```

```python
difficulty = {
    "HPDrainRate": float,
    "CircleSize": float,
    "OverallDifficulty": float,
    "ApproachRate": float,
    "SliderMultiplier": float,
    "SliderTickRate": float,
}
```

```python
timing_point = {
    "time": float,
    "beat_length": float,
    "meter": int,
    "sampleSet": int,
    "sampleIndex": int,
    "volume": int,
    "uninherited": int,
    "effects": int,
}
```

```python
hit_object = {
    "x": int,
    "y": int,
    "time_ms": float,
    "type": int,
    "hitsound": int,
    "object_params": list[str],
    "audio_frame": int,
}
```

```python
audio_cache = {
    audio_key: audio_data,
    ...
}
```

```python
audio_data = {
    "mel": np.ndarray,      # shape: [frames, 80]
    "onset": np.ndarray,    # shape: [frames]
    "sr": int,
    "hop_length": int,
    "duration": float,
}
```

## Notes

The extractor currently keeps slider and spinner-specific data inside `object_params`. Later stages can parse those fields more explicitly once the model representation is settled.

The returned data is still an intermediate representation. Before training, it should be converted into fixed-size chunks, arrays, or tensors.
