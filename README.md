# How to use:

in the same directory as this code file, make a folder called `Input Videos`, and then place all videos inside.
Run the code, and a progress bar show its progress, it will run the entire folder, and save each rectified video in the output `Rect Videos`. (it will make it automatically if its not already detected).

Special options:
`overwrite_existing` = True / False, if you want only to rectify new vides then keep this False and it will skip any files that appear in input and output folder.

------

# `rectify_pipeline.py`

A standalone, single-file GoPro video rectification pipeline. It combines
what used to be split across `test.py` (the undistortion math +
`rectify_video` / `rectify_video_widen`) and `main.py` (folder
orchestration) into one file — with `FastHorizonAlg` deliberately left out,
so this file *only* rectifies videos, nothing else.

Verified to produce **pixel-for-pixel identical output** to `main.py`'s
rectification stage when using matching parameters (see "Verified
equivalence" below).

## What it does

- Accepts either **a single video file** or **a folder of videos** as input
  (auto-detected — no separate "single" vs "batch" function to choose between).
- Undistorts ("flattens") each GoPro video using the calibrated camera model
  in `gopro_DG_rectilinear_model_lrv.json`.
- Skips videos whose output already exists, unless told to overwrite.
- Saves each rectified video under its original filename into the output folder.

## Quick start

```bash
python rectify_pipeline.py "path\to\video.mp4"          # rectify one video
python rectify_pipeline.py "path\to\folder_of_videos"    # rectify every video in a folder
python rectify_pipeline.py                                # no arg -> uses "Input Videos" -> "Rect Videoss2"
```

Or from Python:

```python
from pathlib import Path
from rectify_pipeline import RunConfig, run

run(RunConfig(
    input_path=Path("Input Videos"),      # a folder, or a single .mp4/.mov/.avi/.mkv file
    output_folder=Path("Rect Videos"),
    overwrite_existing=False,
))
```

## The two rectification functions

Only two of the four variants that used to exist in `test.py` were kept here
(the other two — `rectify_video_no_crop` and
`rectify_video_gopro_inverse_zhangK_no_crop` — are not included):

| Function | Output size | Behavior |
|---|---|---|
| `rectify_video(input_video_path, output_video_path=None)` | same as input | The original, faithful rectification: calibrated K unchanged, no zoom, no canvas change. |
| `rectify_video_widen(input_video_path, output_video_path=None, target_aspect_ratio=None, crop_percentile=1.0, crop_top_px=0, crop_bottom_px=0, crop_left_px=0, crop_right_px=0)` | **wider than input** | **The default.** Keeps the calibrated K exactly as-is (no zoom) and widens the canvas instead, since undistorting a wide-angle lens naturally produces a wider image. Can center-crop top/bottom to `target_aspect_ratio`, plus exact manual pixel crop bars applied after that. |

You select between them via `RunConfig.rectify_mode` (`"widen"` or `"original"`)
— see below.

## `RunConfig` — one object instead of scattered globals

You use this class to edit the rectification parameters, you can change them in 'main' :)

```python
@dataclass
class RunConfig:
    # --- input/output ---
    input_path: Path                      # a single video file OR a folder of videos
    output_folder: Path                   # where the rectified video(s) are saved
    overwrite_existing: bool = False      # re-rectify + overwrite existing outputs, or skip them

    # --- which rectification function to use ---
    rectify_mode: str = "widen"           # "widen" -> rectify_video_widen; "original" -> rectify_video

    # --- rectify_video_widen-specific parameters (ignored when rectify_mode="original") ---
    target_aspect_ratio: Optional[float] = None
    crop_percentile: float = 1.0
    crop_top_px: int = 145
    crop_bottom_px: int = 100
    crop_left_px: int = 300
    crop_right_px: int = 300

    # --- misc / local config ---
    video_extensions: frozenset = {".mp4", ".mov", ".avi", ".mkv"}
    tqdm_enabled: bool = True
    timing_enabled: bool = True
    stats_enabled: bool = True
    dprint_enabled: bool = False
```

Notes:

- The `crop_top_px`/`crop_bottom_px`/`crop_left_px`/`crop_right_px` defaults
  (`145, 100, 300, 300`) 
- `rectify_mode="original"` switches to `rectify_video` — in that mode, the
  widen-specific fields (`target_aspect_ratio`, `crop_percentile`, the four
  crop-px fields) are simply ignored.
- `tqdm_enabled`, `timing_enabled`, `stats_enabled`, `dprint_enabled` map
  directly onto the matching flags in `general_useful_functions.py`
  (`guf.TIMING_ENABLED`, `guf.STATS_ENABLED`, `guf.DPRINT_ENABLED`); `run()`
  applies them at the start of every call via `_apply_global_toggles()`.

## Functions

- **`run(config: RunConfig)`** — the main entry point. Resolves
  `config.input_path` to one or more video files, rectifies each with
  `config.rectify_mode`'s function, skips existing outputs unless
  `overwrite_existing=True`, shows a `tqdm` progress bar (togglable), and
  prints a timing summary at the end via `general_useful_functions`.
  Returns the list of output video paths, in input order.
- **`rectify_video(...)`**, **`rectify_video_widen(...)`** — same signatures
  and behavior as their `test.py` counterparts (this file contains its own
  copy of the math so it doesn't need to import `test.py`).


## What's intentionally *not* here

- No horizon detection — this file only rectifies videos.
- No grid collage video builder.
- No multithreading/multiprocessing — this was tried and benchmarked
  (thread pool gave ~1.9x speedup, process pool ~1.55x, on a 4-video/4-worker
  test) but was reverted; `run()` processes videos strictly one at a time.
- No `rectify_video_no_crop` / `rectify_video_gopro_inverse_zhangK_no_crop` —
  only the two functions actually in use (`rectify_video`,
  `rectify_video_widen`) were carried over from `test.py`.

