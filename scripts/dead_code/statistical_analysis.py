# general imports
import os
import json

# plots
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

# utils
from pathlib import Path

# typing
from os import PathLike
from typing import Tuple, List, Dict


def no_image_with_kw(
    json_d: List[dict],
    keywords: Tuple[str, str]
) -> Tuple[int, int, int]:
    only_a: int = 0
    only_b: int = 0
    a_b: int = 0
    not_a_not_b: int = 0

    for _, image in json_d.items():
        kw_0_found: bool = False
        kw_1_found: bool = False
        for rel in image["relationships"]:
            if keywords[0] in [rel["subject"], rel["object"]]:
                kw_0_found: bool = True
            if keywords[1] in [rel["subject"], rel["object"]]:
                kw_1_found: bool = True

        if kw_0_found and not kw_1_found:
            only_a += 1
        elif not kw_0_found and kw_1_found:
            only_b += 1
        elif kw_0_found and kw_1_found:
            a_b += 1
        else:
            not_a_not_b += 1

    assert len(json_d) == only_a + only_b + a_b + not_a_not_b
    return only_a, only_b, a_b


def venn_diagram(
    json_data: List[Dict],
    save_path: PathLike,
    colors: Tuple[str, str, str] = ("r", "g"),
) -> None:

    only_a, only_b, a_b = no_image_with_kw(
        json_data,
        keywords=("man", "woman")
    )

    venn2(
        subsets=(only_a, only_b, a_b),
        set_labels=(
            "man",
            "woman"
        ),
        set_colors=colors
    )
    plt.savefig(save_path)
    plt.clf()


ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT / "data" / "visual_genome" / "processed"
JSON_FILE: Path = DATA_DIR / "merged_data.json"
SAVE_DIR: Path = ROOT / "reports" / "figures"


def main():
    # open json file
    with open(JSON_FILE, "r") as f:
        json_data: Dict[Dict] = json.load(f)

    # Venn diagram
    os.makedirs(SAVE_DIR, exist_ok=True)
    venn_diagram(json_data, save_path=SAVE_DIR / "venn_vg.png")

    # distribution of objects

    # distirbution of relationships


if __name__ == "__main__":
    main()
