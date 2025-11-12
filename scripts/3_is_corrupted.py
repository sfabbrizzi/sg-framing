# general imports
import os

# PIL
from PIL import Image
from PIL import UnidentifiedImageError

# utils
from pathlib import Path
from tqdm import tqdm

# typings
from typing import List


ROOT: Path = Path(__file__).parent.parent
IMAGES: Path = ROOT / "data" / "visual_genome" / "images" / "VG_100K"
REPORTS: Path = ROOT / "reports"

RESIZE_FACTOR: int = 3


def main() -> None:
    unidentified: List[str] = list()

    for file_name in tqdm(os.listdir(IMAGES)):
        if file_name[-4:] != ".jpg":
            continue

        try:
            Image.open(IMAGES / file_name)
        except UnidentifiedImageError:
            unidentified.append(file_name)

    os.makedirs(REPORTS, exist_ok=True)
    if not os.path.isfile(REPORTS / "unidentified_images.txt"):
        with open(REPORTS / "unidentified_images.txt", "w") as f:
            f.write("Unidentified/corrupted images:\n")
            for img_id in unidentified:
                f.write(f"{img_id[:-4]}\n")


if __name__ == "__main__":
    main()
