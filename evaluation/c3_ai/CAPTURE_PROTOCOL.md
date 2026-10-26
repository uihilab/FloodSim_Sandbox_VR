# Scene capture protocol

Goal: 27 screenshots taken with the app's own capture path, each with ground truth read from the simulation. These are the only inputs to `run_eval.py`. Allow about half a day.

## 0. Fixed settings for every capture

- Project: `FloodSim_Sandbox_VR 5.7`, the same level used for the paper's figures.
- Discharges: exactly 171, 568 and 705 m3/s (6,045 / 20,068 / 24,922 cfs), as stated in the manuscript. **The `BP_SimControl` buttons currently say Low 9,600 / Medium 22,700 / High 55,000 cfs.** Set the inflow to the manuscript values before capturing, and write in `notes` how you set it.
- NPCs and vehicles off (SimConfiguration toggles). They move between captures and would differ across repeats.
- Flood-map overlay (`BP_Floodmap`) hidden, damage overlay off, no debug drawing, no UI. The overlays label flooded areas, which would leak the answer.
- Same time of day and weather (SkyCreator) for every scene. Write the preset name in `notes`.
- Capture with the app's `BP_DataAnalysis` actor, calling `TakeScreenshot(FileName)`. It renders its own SceneCapture2D at 1920x1080 (FinalColorLDR) and writes a PNG to `<LaunchDir>/Screenshots`, which is the image the app sends to GPT-4o-mini. Do not use HighResShot or the editor's screenshot tool. Leave its projection as it is and record it in `projection` (perspective or orthographic).

## 1. Camera bookmarks

Place six empty actors (or CineCameraActors used only as transform markers) named `CAM1` to `CAM6` in the level. Save the level, and record each marker's location and rotation in `scenes/camera_bookmarks.csv`, so the views can be reproduced.

| Bookmark | View | Intended content (substitute if not in the domain) |
|---|---|---|
| CAM1 | aerial | The aerial pose used for Figure 8 (the app's default analysis view) |
| CAM2 | street | Riverside Drive, looking along the street toward the river |
| CAM3 | street | Iowa Memorial Union riverfront / entrance |
| CAM4 | street | Burlington Street bridge approach |
| CAM5 | street | A second bridge or underpass in the domain |
| CAM6 | street | A downtown street that stays dry at 568 m3/s (tests over-attribution) |

If a location is not in the simulated domain, pick another named place and update `gazetteer.csv` (and set `in_domain`).

Simple capture Blueprint (Level Blueprint): on key 1 to 6, `SetActorTransform(BP_DataAnalysis, CAMn.GetActorTransform)`, wait 2 frames, then `TakeScreenshot("<scenario>_CAMn.png")`.

## 2. Scenes (27)

| Group | Scenes | How |
|---|---|---|
| Flood scenarios | `normal_CAM1..6`, `flood_CAM1..6`, `extreme_CAM1..6` (18) | Set the discharge, run to steady state (below), pause, capture all six bookmarks |
| Dry controls | `dry_CAM1`, `dry_CAM2`, `dry_CAM3` (3) | Fluid Flux domain empty (no inflow, no water anywhere), same lighting |
| Perturbations | `flood_CAM1_P1..P3`, `extreme_CAM2_P1..P3` (6) | Same paused state as the base scene; P1 = move camera 10 m (1000 uu) sideways, P2 = yaw +10 degrees, P3 = zoom in 15% (FOV or ortho width) |

**Steady state:** run the simulation and note the gage stage (`StageHeight` in `BP_SimControl`) every 30 s. Steady means it changes by less than 0.03 ft over 60 s. Record the elapsed simulation time in `sim_time_s`. Pause the game (`pause` console command) before capturing, so the six views and the perturbations share one water state.

## 3. Ground truth (one manifest row per scene)

Copy `scenes/manifest_template.csv` to `scenes/manifest.csv` and put the 27 PNGs in `scenes/`. Fill these columns while the game is paused:

- `gage_stage_ft`: `StageHeight` at capture.
- `visible_features`: gazetteer names (`|`-separated) that are in frame and identifiable.
- `wet_features`: the subset of `visible_features` with water deeper than 0.05 m on the street surface or touching the building footprint.
- `probe_ids`, `probe_depths_m`, `probe_max_depth_m`: water depth at fixed probe points (below) that are in frame. `probe_max_depth_m` is the largest value, or 0.
- `severity_tier_true`: an integer, using the same definitions the model gets:
  - 0 = no floodwater in frame
  - 1 = water only in the river channel
  - 2 = overbank water on some streets, ground or building perimeters, but most of the visible urban area is dry
  - 3 = widespread inundation; many streets and buildings in view are surrounded by water

  Two authors label tiers independently. Record both in `notes` (e.g. `tier_rater2=2`), and resolve any disagreement before running. Report the inter-rater agreement on truth labels in the paper.
- `capture_width`, `capture_height`, `projection`, `captured_by`, `capture_date`.

**Probes:** place about 8 actors carrying `BP_FluxDataComponent` (the readback component `AC_DamageCalculator` uses) at fixed named points: street centrelines and building entrances. Name them `PRB1..PRB8` and list them in `scenes/camera_bookmarks.csv`. Before trusting the numbers, check what `FluxFluidHeight` / `GetLocalHeight` returns. A probe on dry ground should read 0 if it is depth. If it reads terrain or surface elevation, subtract the ground elevation. Values are in Unreal units (cm), so divide by 100 for metres.

## 4. Optional: in-app end-to-end latency

With an API key in `DataAnalysis.cpp`, trigger the in-app analysis 10 times at CAM1 for each scenario. Note the `UE_LOG` timestamps of "Image saved succesfully." and "Message Content:". This gives capture-to-answer latency inside the engine. `run_eval.py` only measures the API round trip.

## 5. Hand-off

Zip `scenes/` (27 PNGs, `manifest.csv`, `camera_bookmarks.csv`) and record the commit or date of the Unreal project used.
