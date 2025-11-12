# general imports
import json

# utils
from pathlib import Path
from collections import defaultdict

# typing
from typing import List, Dict


ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome"
DATA_FILE: Path = ROOT / "data" / "visual_genome" / "processed" / "merged_data.json"
SAVE_DIR: Path = DATA_FILE.parent


def main() -> None:
    with open(DATA_FILE, "r") as f:
        json_data: Dict[Dict[str, List[Dict]]] = json.load(f)

    vocab_obj: Dict[str, int] = defaultdict(int)
    vocab_rel: Dict[str, int] = defaultdict(int)

    doc_freq_obj: Dict[str, int] = defaultdict(int)
    doc_freq_rel: Dict[str, int] = defaultdict(int)

    for _, image in json_data.items():
        appears_in_doc_obj: List = list()
        appears_in_doc_rel: List = list()
        for rel in image["relationships"]:
            vocab_rel[rel["predicate"]] += 1
            appears_in_doc_rel.append(rel["predicate"])

            vocab_obj[rel["subject"]] += 1
            vocab_obj[rel["object"]] += 1
            appears_in_doc_obj.append(rel["subject"])
            appears_in_doc_obj.append(rel["object"])

        for o in set(appears_in_doc_obj):
            doc_freq_obj[o] += 1
        for r in set(appears_in_doc_rel):
            doc_freq_rel[r] += 1

    with open(SAVE_DIR / "rel_vocab.json", "w") as f:
        json.dump(vocab_rel, f)
    with open(SAVE_DIR / "obj_vocab.json", "w") as f:
        json.dump(vocab_obj, f)
    with open(SAVE_DIR / "rel_doc_freq.json", "w") as f:
        json.dump(doc_freq_rel, f)
    with open(SAVE_DIR / "obj_doc_freq.json", "w") as f:
        json.dump(doc_freq_obj, f)


if __name__ == "__main__":
    main()
