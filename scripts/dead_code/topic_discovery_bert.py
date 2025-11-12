# general imports
import json

# bertopic
from bertopic import BERTopic

# utils
from pathlib import Path
from lightning import seed_everything
from collections import defaultdict

# typing
from typing import List, Dict


ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome"
DATA_FILE: Path = ROOT / "data" / "visual_genome" / "processed" / "merged_data.json"
SAVE_FILE: Path = ROOT / "reports" / "topics_vg_men_women.csv"

SEED: int = 1991


def main() -> None:
    seed_everything(SEED)

    with open(DATA_FILE, "r") as f:
        json_data: Dict[Dict[str, List[Dict]]] = json.load(f)

    docs: List[str] = list()
    indices = defaultdict(list)
    i: int = 0
    for _, image in json_data.items():
        objects: List[str] = list()
        doc: str = ""
        for rel in image["relationships"]:
            objects += [rel["subject"], rel["object"]]
            doc: str = " ".join(
                [doc] + rel["subject"].split("_") + rel["predicate"].split("_") + rel["object"].split("_")
            )

        if "man" in objects:
            indices["man"].append(i)

            i += 1
            docs.append(doc)

    bert = BERTopic().fit(docs)
    df = bert.get_topic_info()
    print(df.head(30))
    print()
    print(bert.get_document_info(docs))

    for k in range(32):
        print(k - 1, df.loc[k, "Representation"])
        print("man", df.loc[k, "Count"] / len(docs))
        # print("woman", topics_f_women[k - 1] / len(indices["woman"]))
        # print("both", topics_f_both[k - 1] / len(indices["both"]))


if __name__ == "__main__":
    main()
