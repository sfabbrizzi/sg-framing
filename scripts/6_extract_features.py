# general imports
import torch
import numpy as np
import argparse
import os

# PIL
from PIL import Image, UnidentifiedImageError

# utils
from pathlib import Path
from lightning import seed_everything
from tqdm import tqdm

# typing
from os import PathLike
from typing import List
from argparse import Namespace

# our package
from src.feature_extraction import load_extractor, extract_features
from src.utils import load_corrupted_images


def main(args: Namespace) -> None:

    # set seeds
    seed_everything(args.seed)
    torch.mps.manual_seed(args.seed)

    # gpu acceleration
    if args.device == "mps" and torch.backends.mps.is_available():
        device: str = "mps"
    elif args.device == "cuda" and torch.cuda.is_available():
        device: str = "cuda"
    else:
        device: str = "cpu"

    # load feature extraction model
    model_dict: dict = load_extractor(
        features=args.features,
        device=device
    )

    # load corrupted images
    corrupted: List[str] = load_corrupted_images(root=Path("../"))

    for path, _, files in os.walk(args.input_path):
        if len([f for f in files if f[-4:] in [".png", ".jpg"]]) == 0:
            continue

        OUTPUT_FOLDER: PathLike = Path(
            os.path.join(
                args.output_path,
                "/".join(path.split("/")[2:-1]),
                args.features,
                path.split("/")[-1]
            )
        )

        if os.path.isdir(OUTPUT_FOLDER):
            print(f"Output folder {OUTPUT_FOLDER} already exists, skipping.")
            continue
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        print(f"extracting features from {path}")
        for f in tqdm(files):
            if f[:-4] in corrupted:
                continue
            image_path: PathLike = Path(path) / f

            try:
                image: Image = Image.open(image_path)
            except UnidentifiedImageError:
                print(f"Could not open {image_path}, skipping.")
                continue

            with torch.no_grad():
                features_tensor: torch.Tensor = extract_features(
                    image,
                    model_dict=model_dict
                )

                np.save(
                    OUTPUT_FOLDER / f"{f[:-4]}.npy",
                    features_tensor.cpu().numpy(),
                )


if __name__ == "__main__":
    # initialize parser
    parser = argparse.ArgumentParser(
        prog="ExtractCLIPFeaturesCelebA",
        description="This program extract features from a given folder")

    # output
    parser.add_argument(
        "--input_path",
        default="../data/visual_genome/images/VG_100K",
        help="path to the images."
    )

    # output
    parser.add_argument(
        "--output_path",
        default="../data/features",
        help="path where to save the features."
    )

    parser.add_argument(
        "--seed",
        default=98
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="device to use."
    )

    parser.add_argument(
        "--features",
        default="clip",
        help="feature extractors to use."
    )

    # args
    args: Namespace = parser.parse_args()

    main(args)
