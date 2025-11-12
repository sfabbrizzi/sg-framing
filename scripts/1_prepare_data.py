# general imports
import os
import json

# utils
from pathlib import Path
from collections import defaultdict

# typing
from typing import List, Dict, Any

# src
from src.utils import load_json_data, extraxt_obj_info
from src.text import clean_text


# paths
ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome"
SAVE_DIR: Path = ROOT / "data" / "visual_genome" / "processed"
SAVE_FILE: Path = SAVE_DIR / "merged_data.json"


def main() -> None:
    json_files: Dict[str, Dict] = dict()

    for data_type in ["objects", "relationships", "attributes"]:
        json_files[data_type] = load_json_data(VISUAL_GENOME, data_type)

    assert len(json_files["objects"]) == len(json_files["relationships"])
    assert len(json_files["objects"]) == len(json_files["attributes"])

    objects_with_attr: Dict[int, List[str]] = defaultdict(list)
    for image in json_files["attributes"]:
        for obj in image["attributes"]:
            if "attributes" in obj:
                objects_with_attr[obj["object_id"]] += [
                    clean_text(" ".join(attr.split("_"))) for attr in obj["attributes"]
                ]

    merged_data: Dict[Dict[str, Any]] = dict()
    for image in json_files["relationships"]:
        image_dict: Dict[str, Any] = {
            "relationships": list()
        }
        for rel in image["relationships"]:
            rel_subj: Dict = rel["subject"]
            rel_obj: Dict = rel["object"]

            rel_dict: Dict[str, str] = {
                "subject": extraxt_obj_info(rel_subj),
                "predicate": extraxt_obj_info(rel),
                "object": extraxt_obj_info(rel_obj)
            }

            image_dict["relationships"].append(rel_dict)
        merged_data[image["image_id"]] = image_dict

    for image in json_files["objects"]:
        new_image: Dict[str, Any] = merged_data[image["image_id"]]
        for obj in image["objects"]:
            if obj["object_id"] in objects_with_attr:
                for attr in objects_with_attr[obj["object_id"]]:
                    new_image["relationships"].append(
                        {"subject": extraxt_obj_info(obj),
                         "predicate": "has_attribute",
                         "object": "_".join(attr.split())}
                    )
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_FILE, "w") as f:
        json.dump(merged_data, f)


if __name__ == "__main__":
    main()
