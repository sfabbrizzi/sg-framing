# general imports
import torch
import json
import random
import os

# transformers
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    BatchFeature,
    set_seed
)

# utils
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login

# typing
from typing import Dict, List
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput

# src
from src.utils import read_token, load_corrupted_images


ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome"
DATA_FILE: Path = VISUAL_GENOME / "processed" / "merged_data.json"
IMAGES: Path = VISUAL_GENOME / "images" / "VG_100K"
SAVE_TO: Path = ROOT / "results" / "people_data.csv"

THRESHOLD: float = 0.97
SEED: int = 1798

HF_TOKEN: Path = ROOT.parent / "restricted" / "hf_token.txt"
login(read_token(HF_TOKEN))


def main():
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    random.seed(SEED)
    set_seed(SEED)

    # Initialize the processor and model
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm")
    model.to("mps")

    # Placeholder for further code (e.g., loading images, running inference)
    print("The model loaded and ready.")

    # load json data
    with open(DATA_FILE, "r") as f_json:
        json_data: Dict[Dict[str, List[Dict]]] = json.load(f_json)

    # load corrupted images
    corrupted: List[str] = load_corrupted_images(root=ROOT)

    os.makedirs(SAVE_TO.parent, exist_ok=True)
    with open(SAVE_TO, "w") as f_csv:
        # write header
        f_csv.write("image_id,is_corrupted,contains_person,used_people_detector\n")

        for image_id, image in tqdm(json_data.items()):
            # manage corrupted images
            if image_id in corrupted:
                f_csv.write(f"{image_id},1,0,0\n")
                continue

            # check for presence of person-related terms
            list_of_people_related_words: List[str] = [
                "man",
                "woman",
                "person",
                "people",
                "child",
                "boy",
                "girl"
            ]
            found: bool = False
            for rel in image["relationships"]:
                if rel["subject"] in list_of_people_related_words:
                    f_csv.write(f"{image_id},0,1,0\n")
                    found: bool = True
                    break
                if rel["object"] in list_of_people_related_words:
                    f_csv.write(f"{image_id},0,1,0\n")
                    found: bool = True
                    break

            if not found:
                with torch.no_grad():
                    # if not present, run inference
                    image_path: Path = IMAGES / f"{image_id}.jpg"
                    image = Image.open(image_path).convert("RGB")

                    # 1 is the label corresponding to "person"
                    inputs: BatchFeature = processor(
                        images=image,
                        return_tensors="pt"
                    )
                    outputs: DetrObjectDetectionOutput = model(
                        **inputs.to("mps")
                    )

                    target_sizes: torch.Tensor = torch.tensor([image.size[::-1]])
                    results: Dict = processor.post_process_object_detection(
                        outputs,
                        target_sizes=target_sizes,
                        threshold=THRESHOLD
                    )[0]

                    labels: List[str] = [
                        model.config.id2label[label.item()]
                        for label in results["labels"]
                    ]

                    contains_person: int = int("person" in labels)
                    f_csv.write(f"{image_id},0,{contains_person},1\n")


if __name__ == "__main__":
    main()
