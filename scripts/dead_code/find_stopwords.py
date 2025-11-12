# general imports
import json

# utils
from pathlib import Path
from collections import defaultdict

# typing
from typing import List, Dict


ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome"
DATA_FILE_OBJ: Path = ROOT / "data" / "visual_genome" / "processed" / "obj_doc_freq.json"
DATA_FILE_REL: Path = ROOT / "data" / "visual_genome" / "processed" / "rel_doc_freq.json"
DATA_FILE: Path = ROOT / "data" / "visual_genome" / "processed" / "merged_data.json"


def main() -> None:
    stop_words_obj: List[str] = list()
    stop_words_rel: List[str] = list()

    with open(DATA_FILE_OBJ, "r") as f:
        json_data: Dict[Dict[str, List[Dict]]] = json.load(f)

    for t, f in json_data.items():
        if f > 108_077 * 0.2:
            stop_words_obj.append(t)

    with open(DATA_FILE_REL, "r") as f:
        json_data: Dict[Dict[str, List[Dict]]] = json.load(f)

    for t, f in json_data.items():
        if f > 108_077 * 0.2:
            stop_words_rel.append(t)

    with open(DATA_FILE, "r") as f:
        json_data: Dict[Dict[str, List[Dict]]] = json.load(f)

    vocab_obj_men = defaultdict(int)
    vocab_obj_women = defaultdict(int)
    vocab_rel_men = defaultdict(int)
    vocab_rel_women = defaultdict(int)
    vocab_obj_both = defaultdict(int)
    vocab_rel_both = defaultdict(int)

    # KEYWORDS = ["man", "woman"]
    # NEG = []
    i = 0
    j = 0
    k = 0
    for _, image in json_data.items():
        objects: List[str] = list()
        predicates: List[str] = list()
        for rel in image["relationships"]:
            objects += [rel["subject"], rel["object"]]
            predicates.append(rel["predicate"])

        if "man" in objects and "woman" not in objects:  # and len({"baby", "child", "boy"}.intersection(set(objects))) != 0:
            i += 1
            for o in set(objects) - {"woman", "man"}:
                vocab_obj_men[o] += 1
            for p in set(predicates):
                vocab_rel_men[p] += 1
        elif "woman" in objects and "man" not in objects:  # and len({"baby", "child", "boy"}.intersection(set(objects))) != 0:
            j += 1
            for o in set(objects) - {"woman", "man"}:
                vocab_obj_women[o] += 1
            for p in set(predicates):
                vocab_rel_women[p] += 1
        elif "woman" in objects and "man" in objects:  # and len({"baby", "child", "boy"}.intersection(set(objects))) != 0:
            k += 1
            for o in set(objects) - {"woman", "man"}:
                vocab_obj_both[o] += 1
            for p in set(predicates):
                vocab_rel_both[p] += 1

    print(sorted([(t, round(100 * f / i, 1)) for t, f in vocab_obj_men.items() if t not in stop_words_obj], key=lambda x: -abs(x[-1]))[:30])
    print(sorted([(t, round(100 * f / i, 1)) for t, f in vocab_rel_men.items() if t not in stop_words_rel], key=lambda x: -abs(x[-1]))[:30])

    print()
    print(sorted([(t, round(100 * f / i - 100 * vocab_obj_women[t] / j, 1)) for t, f in vocab_obj_men.items()], key=lambda x: -abs(x[-1]))[:15])
    print(sorted([(t, round(100 * f / i - 100 * vocab_rel_women[t] / j, 1)) for t, f in vocab_rel_men.items()], key=lambda x: -abs(x[-1]))[:15])

    print(vocab_obj_women["girl"] / j, vocab_obj_men["girl"] / i, vocab_obj_both["girl"] / k)
    print(vocab_obj_women["boy"] / j, vocab_obj_men["boy"] / i, vocab_obj_both["boy"] / k)
    print(vocab_obj_women["child"] / j, vocab_obj_men["child"] / i, vocab_obj_both["child"] / i)
    print(vocab_obj_women["baby"] / j, vocab_obj_men["baby"] / i, vocab_obj_both["baby"] / i)

    print(vocab_obj_women["sky"] / j, vocab_obj_men["sky"] / i, vocab_obj_both["sky"] / k)


if __name__ == "__main__":
    main()
