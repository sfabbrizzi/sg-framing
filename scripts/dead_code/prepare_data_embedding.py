# general imports
import json
import os

# utils
from pathlib import Path
from tqdm import tqdm

# typing
from typing import List


ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome"
DATA_FILE: Path = ROOT / "data" / "visual_genome" / "processed" / "merged_data.json"
SAVE_DIR: Path = DATA_FILE.parent
SAVE_FILE: Path = SAVE_DIR / "relationships.csv"

SEP: str = "|"  # separator for the CSV file


def main() -> None:
    # load data
    with open(DATA_FILE, "r") as f_json:
        rel_data: List[dict] = json.load(f_json)

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_FILE, "w") as f_csv:
        handle: str = SEP.join([
            "from",  # subject
            "to",    # object
            "rel"    # relationship
        ])
        f_csv.write(f"{handle}\n")
        # iterate over json data
        for _, item in tqdm(rel_data.items()):
            # iterate over relationships
            for rel in item["relationships"]:
                line: str = SEP.join([
                    rel["subject"],
                    rel["object"],
                    rel["predicate"],
                ])

                f_csv.write(f"{line}\n")


if __name__ == "__main__":
    main()
