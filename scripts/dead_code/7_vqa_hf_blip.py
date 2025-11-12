# general imports
import torch
import pandas as pd
import random

# transformers
from transformers import (
    pipeline,
    Pipeline,
    set_seed
)

# utils
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login

# typing
from typing import Dict, Any

# src
from src.utils import read_token


ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome"
IMAGES: Path = VISUAL_GENOME / "images" / "VG_100K"
CSV_FILE: Path = VISUAL_GENOME / "processed" / "people_data.csv"

THRESHOLD: float = 0.97
SEED: int = 1798

# QUESTION: str = "Does the image depict people practising sport?"
QUESTION: str = "Does the image depict people at work?"
# ATTRIBUTE: str = "is_sport_blip"
ATTRIBUTE: str = "is_work_blip"

HF_TOKEN: Path = ROOT.parent / "restricted" / "hf_token.txt"
login(read_token(HF_TOKEN))


def main():
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    random.seed(SEED)
    set_seed(SEED)

    # Initialize the processor and model
    pipe: Pipeline = pipeline(
        task="visual-question-answering",
        model="Salesforce/blip-vqa-base",
        # model="Salesforce/blip2-opt-2.7b",
        device="mps",
    )

    df: pd.DataFrame = pd.read_csv(CSV_FILE)
    df[ATTRIBUTE] = ["0"] * len(df)

    split: pd.DataFrame = df[
        (df.contains_person == 1)
    ]
    for i in tqdm(split.index):
        image_id: int = split.loc[i, "image_id"]
        path: Path = IMAGES / f"{image_id}.jpg"

        image = Image.open(path).convert("RGB")
        with torch.no_grad():
            outputs: Dict[str, Any] = pipe(image, QUESTION)[0]

        df.loc[i, ATTRIBUTE] = outputs["answer"]

    df.to_csv(CSV_FILE, index=False)


if __name__ == "__main__":
    main()
