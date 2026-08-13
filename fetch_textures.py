"""Download CC0 PBR texture sets from ambientCG into textures_pbr/.

    python3 fetch_textures.py            # download everything that's missing
    python3 fetch_textures.py --list     # just show what would be fetched

Each set ends up as

    textures_pbr/<AssetId>/Color.jpg
                          /Roughness.jpg
                          /Normal.jpg

script.py picks these up automatically: parcel
sets are used as package surfaces (colour + roughness + normal), floor sets
feed the PBR floor materials. If this directory is missing the script silently
falls back to the plain photo textures in textures/.

The prefix of the folder name decides how a set is used, so keep the ambientCG
naming: script.py routes Cardboard/Chipboard/Cork -> cardboard,
Paper/Wallpaper -> paper, Plastic, Foil, Fabric/Wicker/Leather -> fabric,
Styrofoam -> foam, Wood -> timber (crates), and everything in
_FLOOR_PREFIXES -> floors.

ambientCG assets are CC0 (public domain) - no attribution required.
"""

import argparse
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

OUT = Path(__file__).resolve().parent / "textures_pbr"
RES = "1K-JPG"          # 2K-JPG also works; 1K keeps RAM/VRAM sane over 3000 frames

# Package surfaces. The prefix decides which parcel "kind" the set is used for
# in the render script (cardboard / paper / plastic / fabric / foam), so keep
# the naming - script.py maps prefixes to kinds.
PARCEL_SETS = [
    # corrugated / board
    "Cardboard001", "Cardboard002", "Cardboard003", "Cardboard004",
    "Chipboard001", "Chipboard002", "Chipboard003", "Chipboard004",
    "Chipboard005", "Chipboard006", "Chipboard007", "Chipboard008",
    "Cork001", "Cork002", "Cork003", "Cork004",
    # paper / printed wraps
    "Paper001", "Paper002", "Paper003", "Paper004", "Paper005", "Paper006",
    # poly bags / shrink wrap / taped plastic
    "Plastic001", "Plastic002", "Plastic003", "Plastic004", "Plastic005",
    "Plastic006", "Plastic007", "Plastic008", "Plastic009", "Plastic010",
    "Plastic011", "Plastic012A", "Plastic013A", "Plastic014A", "Plastic015A",
    "Plastic016A", "Plastic017A",
    "Foil001", "Foil002", "Foil003",
    # woven sacks / bags
    "Fabric001", "Fabric004", "Fabric008", "Fabric010", "Fabric012",
    "Fabric015", "Fabric019", "Fabric020", "Fabric023", "Fabric024",
    "Fabric028", "Fabric029", "Fabric031", "Fabric034", "Fabric036",
    "Fabric041", "Fabric048",
    "Wicker001", "Wicker002", "Wicker005", "Wicker006",
    # foam boxes
    "Styrofoam001", "Styrofoam002", "Styrofoam003", "Styrofoam004",
    "Styrofoam005",
    # shipping labels
    "Sticker001",

    # ---- the wider surface pool ----------------------------------------
    # board / paper wraps
    "CardboardSet001",
    "Wallpaper001A", "Wallpaper001B", "Wallpaper001C",
    "Wallpaper002A", "Wallpaper002B", "Wallpaper002C",
    # more poly / film / foil
    "Plastic012B", "Plastic013B", "Plastic014B", "Plastic015B",
    "Plastic016B", "Plastic017B", "Plastic018A", "Plastic018B",
    # more woven / textile sacks
    "Fabric002", "Fabric003", "Fabric005", "Fabric006", "Fabric007",
    "Fabric009", "Fabric011", "Fabric013", "Fabric014", "Fabric016",
    "Fabric017", "Fabric021", "Fabric025", "Fabric026", "Fabric027",
    "Fabric030", "Fabric032", "Fabric033", "Fabric035", "Fabric037",
    "Fabric044", "Fabric047", "Fabric052", "Fabric055", "Fabric060",
    "Fabric063", "Fabric068", "Fabric072", "Fabric077", "Fabric083",
    "Wicker003", "Wicker004", "Wicker007A", "Wicker008A", "Wicker009A",
    "Wicker010A", "Wicker011A", "Wicker012A", "Wicker013",
    "Wicker007B", "Wicker008B", "Wicker009B",
    "Fabric018", "Fabric022", "Fabric038", "Fabric042", "Fabric050",
    "Fabric058", "Fabric065", "Fabric070",
    # leather-ish wraps (used as another "fabric" family)
    "Leather002", "Leather005", "Leather008", "Leather011", "Leather014",
    "Leather017", "Leather020", "Leather026", "Leather031", "Leather037",
    # bare timber, for wooden crates / trays
    "Wood001", "Wood008", "Wood014", "Wood022", "Wood027", "Wood033",
    "Wood041", "Wood048", "Wood053", "Wood060", "Wood066", "Wood071",
    "Wood077", "Wood085A", "Wood092",
]

# Floors
FLOOR_SETS = [
    "Concrete031", "Concrete034", "Concrete036", "Concrete042A",
    "Concrete045", "Concrete047A",
    "Concrete014", "Concrete016", "Concrete018", "Concrete020",
    "Concrete023", "Concrete025", "Concrete029", "Concrete033",
    "Tiles074", "Tiles107", "Tiles138", "Tiles139", "Tiles141",
    "Tiles012", "Tiles028", "Tiles045", "Tiles059", "Tiles093",
    "Tiles101", "Tiles118", "Tiles126",
    "Asphalt031", "Asphalt033", "Road012A", "Road006", "Road007",
    "PavingStones128", "PavingStones138", "PavingStones070",
    "PavingStones092", "PavingStones112",
    "WoodFloor051", "WoodFloor064", "WoodFloor007", "WoodFloor033",
    "WoodFloor043",
    "Planks011", "Planks016",
    "Metal032", "Metal048A", "CorrugatedSteel009",
    "Metal007", "Metal022", "Metal038",
    "MetalPlates006", "MetalPlates008",
    "Terrazzo003", "Terrazzo009", "Terrazzo017",
    "Marble006", "Marble012",
    "Bricks023", "Bricks051",
    "Ground054", "Gravel022", "Gravel023",
    "Rubber002", "Rubber004",
    "Carpet012", "Carpet016", "Carpet004", "Carpet009", "Carpet013",

    # ---- the wider floor pool -------------------------------------------
    "Tiles003", "Tiles017", "Tiles036", "Tiles052", "Tiles066",
    "Tiles081", "Tiles096", "Tiles110", "Tiles124", "Tiles131", "Tiles144",
    "Concrete002", "Concrete008", "Concrete012", "Concrete019",
    "Concrete027", "Concrete038", "Concrete041A", "Concrete044A",
    "Concrete048",
    "PavingStones004", "PavingStones019", "PavingStones037",
    "PavingStones055", "PavingStones078", "PavingStones099",
    "PavingStones121", "PavingStones140",
    "WoodFloor005", "WoodFloor018", "WoodFloor027", "WoodFloor039",
    "WoodFloor048", "WoodFloor057", "WoodFloor068",
    "Planks003", "Planks009", "Planks019", "Planks028", "Planks038",
    "Carpet001", "Carpet003", "Carpet006", "Carpet010", "Carpet014",
    "Ground010", "Ground027", "Ground047", "Ground068", "Ground087",
    "Gravel005", "Gravel013", "Gravel028", "Gravel039",
    "Bricks011", "Bricks029", "Bricks047", "Bricks066", "Bricks084",
    "Bricks101",
    "Terrazzo005", "Terrazzo011", "Terrazzo018",
    "Marble008", "Marble016", "Marble023",
    "Onyx004", "Onyx011",
    "Travertine006", "Travertine012",
    "Rock022", "Rocks014",
    "Asphalt008", "Asphalt016", "Asphalt026A",
    # painted warehouse / road markings
    "RoadLines003", "RoadLines009", "RoadLines015", "RoadLines021A",
    "RoadLines028A",
    "OfficeCeiling002", "OfficeCeiling005",
    "Facade007", "Facade016",
    # industrial gratings / plate floors
    "MetalWalkway003", "MetalWalkway008", "MetalWalkway012",
    "DiamondPlate002", "DiamondPlate007A",
    "Metal015", "Metal029", "Metal045A", "MetalPlates011",
    "Rubber001", "Rubber003",
    "CorrugatedSteel004", "CorrugatedSteel008A",
]

# zip member suffix -> file we save
WANTED = {
    "_Color.jpg": "Color.jpg",
    "_Roughness.jpg": "Roughness.jpg",
    "_NormalGL.jpg": "Normal.jpg",
}


def fetch(asset_id):
    dst = OUT / asset_id
    if all((dst / n).exists() for n in WANTED.values()):
        return f"= {asset_id} (already present)"
    url = f"https://ambientcg.com/get?file={asset_id}_{RES}.zip"
    try:
        req = Request(url, headers={"User-Agent": "dataset-fetch/1.0"})
        with urlopen(req, timeout=180) as r:
            blob = r.read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
        dst.mkdir(parents=True, exist_ok=True)
        got = []
        for member in zf.namelist():
            for suffix, out_name in WANTED.items():
                if member.endswith(suffix):
                    (dst / out_name).write_bytes(zf.read(member))
                    got.append(out_name)
        if not got:
            return f"! {asset_id}: zip had none of {list(WANTED)}"
        return f"+ {asset_id}: {', '.join(sorted(got))}"
    except Exception as exc:
        return f"! {asset_id}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", choices=["parcels", "floors"])
    args = ap.parse_args()

    sets = []
    if args.only != "floors":
        sets += PARCEL_SETS
    if args.only != "parcels":
        sets += FLOOR_SETS

    if args.list:
        print("\n".join(sets))
        print(f"\n{len(sets)} sets -> {OUT}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SOURCE.txt").write_text(
        "PBR texture sets downloaded from https://ambientcg.com (CC0 / public "
        "domain).\nRe-fetch or extend the list with fetch_textures.py.\n")

    ok = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(fetch, sets):
            print(line, flush=True)
            ok += line.startswith(("+", "="))
    print(f"\n{ok}/{len(sets)} sets available in {OUT}")
    return 0 if ok == len(sets) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
