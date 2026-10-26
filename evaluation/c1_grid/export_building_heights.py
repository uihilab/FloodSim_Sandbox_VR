"""Run inside the Unreal Editor Python console during PIE:
    py "<path>/export_building_heights.py" --tag flood_1m

Writes evaluation/data/c1/inengine/buildings_<tag>.csv with, for every actor that carries
AC_DamageCalculator, the values that component already computes: the Fluid Flux local water
height and the mesh height (both in Unreal units, cm), plus the actor bounds.

If a property name is not found, open AC_DamageCalculator and check the variable names
(the compiled Blueprint lists LocalHeight, MeshHeight, LocalDamage, FluxData); adjust PROPS.
"""
import argparse
import csv
import os

import unreal

PROPS = ["LocalHeight", "MeshHeight", "LocalDamage"]

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
args = parser.parse_args()

worlds = unreal.EditorLevelLibrary.get_pie_worlds(False) if hasattr(unreal.EditorLevelLibrary, "get_pie_worlds") else []
if not worlds:
    worlds = [unreal.UnrealEditorSubsystem().get_game_world()]
world = worlds[0]
if world is None:
    raise RuntimeError("Start Play In Editor first; the script reads the running simulation.")

dmg_class = unreal.load_object(None, "/Game/Blueprints/AC_DamageCalculator.AC_DamageCalculator_C")
out_dir = os.path.join(unreal.Paths.project_dir(), "evaluation", "data", "c1", "inengine")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, f"buildings_{args.tag}.csv")

rows = []
for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
    comp = actor.get_component_by_class(dmg_class) if dmg_class else None
    if comp is None:
        continue
    origin, extent = actor.get_actor_bounds(False)
    row = dict(actor=actor.get_name(), x=origin.x, y=origin.y, z_min=origin.z - extent.z, z_max=origin.z + extent.z)
    for p in PROPS:
        try:
            row[p] = comp.get_editor_property(p)
        except Exception:
            row[p] = ""
    rows.append(row)

with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["actor"])
    w.writeheader()
    w.writerows(rows)
unreal.log(f"Wrote {len(rows)} buildings to {path}")
