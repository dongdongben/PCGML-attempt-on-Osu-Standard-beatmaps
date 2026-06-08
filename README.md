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

## Extractor Output

Running the extractor produces `processed/extracted.pkl`. This is the
intermediate representation used to build the PyTorch dataset.

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

## PyTorch Dataset

Run the dataset conversion after creating `processed/extracted.pkl`:

```powershell
python dataset.py
```

This creates `processed/tensors.pt` with the following top-level structure:

```python
tensor_dataset = {
    "maps_tensor": maps_tensor,
    "audio_tensor": audio_tensor,
}
```

Load the saved dataset with:

```python
from dataset import load_tensors

data = load_tensors("processed/tensors.pt")
maps_tensor = data["maps_tensor"]
audio_tensor = data["audio_tensor"]
```

### Audio Tensors

Audio features are stored once per unique audio file:

```python
audio_tensor = {
    audio_key: song_tensor,
    ...
}
```

```python
song_tensor: torch.float32  # shape: [audio_frames, 81]
```

Each audio frame contains:

```text
columns 0-79: mel-spectrogram features
column 80:    onset-strength feature
```

### Map Tensors

Map features are stored by beatmap ID:

```python
maps_tensor = {
    beatmap_id: map_tensor,
    ...
}
```

```python
map_tensor = {
    "audio_key": str,
    "difficulty": torch.Tensor,
    "grid": torch.Tensor,
    "objects": torch.Tensor,
    "slider_features": list[slider_feature],
    "uninherited_tp": torch.Tensor,
    "inherited_tp": torch.Tensor,
}
```

#### Difficulty

```python
map_tensor["difficulty"]: torch.float32  # shape: [6]
```

The columns use this fixed order:

```text
0: HPDrainRate
1: CircleSize
2: OverallDifficulty
3: ApproachRate
4: SliderMultiplier
5: SliderTickRate
```

#### Snap Grid

```python
map_tensor["grid"]: torch.float32  # shape: [number_of_snaps]
```

Each value is a legal snap time in milliseconds. The current default grid uses
a divisor of `4`.

#### Hit Objects

```python
map_tensor["objects"]: torch.float32  # shape: [number_of_objects, 5]
```

Object columns:

```text
0: normalized x position, x / 512
1: normalized y position, y / 384
2: object time in milliseconds
3: object kind ID
4: new-combo flag, 0 or 1
```

Object kind IDs:

```text
0: hit circle
1: slider
2: spinner
```

The compact kind IDs are intended for model training. They must be converted
back into osu! type bit flags when reconstructing a `.osu` file.

Maps without objects use an empty tensor with shape `[0, 5]`.

#### Slider Features

Only slider objects receive slider-specific features:

```python
slider_feature = {
    "curve_type": torch.long,       # scalar
    "repeat": torch.long,           # scalar
    "pixel_length": torch.float32,  # scalar
    "points": torch.float32,        # shape: [6, 2]
    "point_mask": torch.float32,    # shape: [6]
}
```

Slider control points are normalized by the osu!standard playfield dimensions.
Unused point slots are zero-padded and marked with `0` in `point_mask`.

Curve type IDs:

```text
0: linear
1: perfect circle
2: Bezier
3: Catmull
```

`slider_features` is currently a structured list rather than one stacked
tensor because its fields have different shapes and data types.

#### Timing Points

```python
map_tensor["uninherited_tp"]: torch.float32  # shape: [number_of_points, 8]
map_tensor["inherited_tp"]: torch.float32    # shape: [number_of_points, 8]
```

Timing-point columns:

```text
0: time
1: beat_length
2: meter
3: sampleSet
4: sampleIndex
5: volume
6: uninherited
7: effects
```

`uninherited_tp` contains BPM/timing-section points. `inherited_tp` contains
slider-velocity and other inherited timing changes. Empty timing-point groups
use tensors with shape `[0, 8]`.

## Pipeline

```text
songs/
  -> Extractor.py
  -> processed/extracted.pkl
  -> dataset.py
  -> processed/tensors.pt
  -> training code
```

Rerun `Extractor.py` when raw parsing or audio extraction changes. Rerun
`dataset.py` when the tensor representation changes. Model-only changes can
reuse `processed/tensors.pt`.

## Notes

The extractor currently keeps slider and spinner-specific data inside `object_params`. Later stages can parse those fields more explicitly once the model representation is settled.

The saved tensor dataset is still a map-level representation. Training code may
need to convert maps into fixed-size chunks, padded batches, or snap-level
examples depending on the model.
