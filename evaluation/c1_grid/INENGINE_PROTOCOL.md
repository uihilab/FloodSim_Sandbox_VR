# In-engine grid sensitivity (Fluid Flux): protocol for the co-author

The independent solver study (`run_experiment.py`, `metrics.py`) shows how a heightfield shallow-water
model of this reach responds to cell size. This protocol repeats the key part inside FloodSim Sandbox
itself, so the response can also report how Fluid Flux behaves and what each cell size costs in frame time.

Budget: about 2-3 hours at the workstation used in the paper (Alienware Aurora R15, i9-13900F, RTX 4070 Ti).

## 1. Settings to change

`BP_FluxSimulationDomain` (Content/FluidFlux/Simulation/Blueprints) in the Iowa City level:

1. Note the current **Area World Size** and **Resolution**. Pixel size = Area World Size / Resolution
   (Unreal units are cm, so 100 uu = 1 m). Record these values exactly; the paper says "roughly 1 m".
2. Keep Area World Size fixed. Change only Resolution so that the pixel size is **0.5 m, 1 m and 2 m**
   (for example, a 4096 m area needs 8192 / 4096 / 2048). If 0.5 m does not fit in GPU memory, record the
   error and use 0.75 m as the finest level instead.
3. Keep every other setting unchanged (inflow source, outlet, iteration settings, landscape, buildings).

Do the three discharges at each resolution: Normal 171, Flood 568, Extreme 705 m³/s. **First check what
the Low/Medium/High buttons in `BP_SimControl` send to the inflow source: they are labelled 9,600 /
22,700 / 55,000 cfs, which does not match the manuscript.** Set the inflow to the manuscript values for
these runs and tell Ram which values the paper figures were made with.

## 2. Each run (9 runs)

1. Start PIE in desktop (not VR) mode, select the discharge, and let the simulation run until the gage
   height stops changing (watch the stage readout; typically several minutes of simulated time). Record
   the time it took.
2. In the console: `stat unit`, then `csvprofile start`, wait 60 s, `csvprofile stop`. Keep the CSV from
   `Saved/Profiling/CSV/`. Also note the average FPS shown by `stat fps`.
3. Run `export_building_heights.py` (this folder) from the editor's Python console
   (`py "<path>/export_building_heights.py" --tag flood_1m`). It writes one row per building with the
   Fluid Flux local water height and the building mesh height that `AC_DamageCalculator` already reads,
   plus the gage stage. If the script cannot find the properties, see the note at the top of the script.
4. Take a top-down orthographic screenshot from the same camera bookmark every time (the AI capture
   bookmark `AERIAL` from `../c3_ai/CAPTURE_PROTOCOL.md` works). Name it `<discharge>_<cellsize>.png`.

## 3. Hand back

Put everything in `evaluation/data/c1/inengine/`: the 9 building CSVs, 9 profiler CSVs, 9 screenshots,
and a short `settings.txt` with Area World Size, Resolution values, inflow values actually used, and
the time to steady state for each run. Then run:

```
python analyze_inengine.py
```

It reports, against the finest run: per-building depth differences, changes in wet/dry classification,
gage stage differences, and frame time / FPS for each cell size.
