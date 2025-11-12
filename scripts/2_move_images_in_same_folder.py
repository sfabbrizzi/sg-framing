"""Visual Genome has images in two folders. This script moves all images to the same folder."""

# general imports
import os

# utils
from pathlib import Path

# typing
from typing import List


ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome" / "images"
FROM: Path = VISUAL_GENOME / "VG_100K_2"
TO: Path = VISUAL_GENOME / "VG_100K"


def main() -> None:
    IMAGES_FROM: List[str] = [f for f in os.listdir(FROM) if f[-4:] == ".jpg"]
    no_images: int = len(IMAGES_FROM) + len([f for f in os.listdir(TO) if f[-4:] == ".jpg"])

    for file_name in IMAGES_FROM:
        os.rename(FROM / file_name, TO / file_name)

    assert no_images == len([f for f in os.listdir(TO) if f[-4:] == ".jpg"])


if __name__ == "__main__":
    main()
