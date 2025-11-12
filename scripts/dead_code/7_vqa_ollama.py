# general imports
import os
import pandas as pd

# ollama
from ollama import chat
from pydantic import BaseModel

# utils
from pathlib import Path
from tqdm import tqdm

# typing
from ollama import ChatResponse
from typing import Literal
from os import PathLike


ROOT: Path = Path(__file__).parent.parent
DATA: Path = ROOT / "data" / "visual_genome"
CSV_FILE: Path = ROOT / "results" / "people_data_manual_labels_work.csv"
CSV_FILE_SAVE: Path = ROOT / "results" / "work_data_ollama_label.csv"
IMAGES: Path = DATA / "images" / "VG_resized_fact_2"

SEED: int = 1348

MODEL: Literal[
    "llava-phi3:latest",
    "qwen2.5vl:3b",
    "mistral:latest"
] = "llava-phi3:latest"

# SYSTEM_PROMPT: str = (
#     "Your task is to decide whether an image depicts a sport scene or not.\n\n"
#     "A sport scene is characterized by individuals engaged in physical activities, "
#     "such as playing sports, exercising, or participating in athletic events.\n\n"
#     "It does not have to be a professional sport,"
#     "but it should be clear that the individuals are involved in some form of sport activity "
#     "Answer: 'Sport' or 'Not Sport'."
# )
# QUESTION: str = "Is it Sport or Not Sport?"
SYSTEM_PROMPT: str = (
    "Your task is to decide whether an image depicts a work scene or not.\n\n"
    "A work scene is characterized by individuals engaged in work-related activities. "
    "Note that the worker must be part of the main subject of the image to be classified as work "
    "and t should be clear that the individuals are involved in some form of work activity "
    "Answer: 'Work' or 'Not Work'."
)
QUESTION: str = "Is it Work or Not Work?"
ATTRIBUTE: str = "is_work_ollama"

# class Response(BaseModel):
#     response: Literal["Sport", "Not Sport"]


class Response(BaseModel):
    response: Literal["Work", "Not Work"]


def main() -> None:
    df: pd.DataFrame = pd.read_csv(CSV_FILE)
    os.makedirs(CSV_FILE_SAVE.parent, exist_ok=True)

    # split: pd.DataFrame = df[
    #     (df.contains_person == 1)
    #     & (df[f"is_work_{MODEL}"] == "NotApplied")
    #     & ((df.prof_sport_hf != 1)
    #        | (df.confidence_prof_sport_hf < 0.98))
    # ]
    if not os.path.isfile(CSV_FILE_SAVE):
        with open(CSV_FILE_SAVE, "w") as f:
            f.write(f"image_id,{ATTRIBUTE}\n")

    with open(CSV_FILE_SAVE, "r") as f:
        done = f.read().splitlines()
    done = [int(line.split(",")[0]) for line in done[1:]]

    with open(CSV_FILE_SAVE, "a") as f:
        for i in tqdm(df[(df.contains_person == 1)].index):
            image_id: int = df.loc[i, "image_id"]
            if image_id in done:
                continue
            path: PathLike = str(IMAGES / f"{image_id}.jpg")

            response: ChatResponse = chat(
                model=MODEL,
                messages=[
                    {
                        'role': 'system',
                        'content': SYSTEM_PROMPT,
                    },
                    {
                        'role': 'user',
                        'content': QUESTION,
                        "images": [path]
                    },
                ],
                format=Response.model_json_schema(),
                options={"seed": SEED}
            )

            value = Response.model_validate_json(response.message.content)
            f.write(f"{image_id},{value.response}\n")
            f.flush()


if __name__ == "__main__":
    main()
