# general imports
import torch

# torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import ToTensor, Compose, Resize

# tranformers
from transformers import (CLIPModel,
                          CLIPImageProcessor,
                          ViTModel,
                          AutoProcessor,
                          SwinModel)

# PIL
from PIL.Image import Image


def load_extractor(features: str, device: torch.DeviceObjType | str) -> dict:
    """This function loads the model and processor for the
    feature extraction.

    Parameters
    ----------
    features : str
        features in ["clip", "resnet18", "vit", "swin"].
    device : torch.DeviceObjType |str

    Returns
    -------
    dict
        dictionary containing the model and processor
        (if features != resnet18), and other necessary
        information (features and device).
    """
    to_return = {"features": features, "device": device}

    match features:
        case "clip":
            processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                return_tensors="pt"
            )
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(device)

            to_return["model"] = model
            to_return["processor"] = processor

        case "resnet18":
            model: nn.Module = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1,
            ).to(device).eval()
            model.fc = nn.Identity()

            to_return["model"] = model

        case "vit":
            processor = AutoProcessor.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                return_tensors="pt",
                use_fast=True
            )
            model = ViTModel.from_pretrained(
                "google/vit-base-patch16-224-in21k"
            ).to(device)

            to_return["model"] = model
            to_return["processor"] = processor

        case "swin":
            processor = AutoProcessor.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224",
                return_tensors="pt",
                use_fast=True
            )
            model = SwinModel.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224"
            ).to(device)

            to_return["model"] = model
            to_return["processor"] = processor

        case _:
            raise NotImplementedError(
                "features must be in ['vit', 'swin', 'resnet18', 'clip']"
            )

    return to_return


def extract_features(image: Image,
                     model_dict: dict) -> torch.Tensor:
    """Given a dictionary returned by the function
    load_extractor and an Image, it returns
    the feature tensor.

    Parameters
    ----------
    image : Image
        Image to extract the features of.
    model_dict : dict
        dictiotionary returned by the function load_extractor.

    Returns
    -------
    torch.Tensor
    """

    features = model_dict["features"]
    device = model_dict["device"]
    model = model_dict["model"]
    if "processor" in model_dict:
        processor = model_dict["processor"]

    match features:
        case "clip":
            inputs: torch. Tensor = processor(
                image
            )["pixel_values"][0].reshape(1, 3, 224, -1)
            outputs: torch.Tensor = torch.Tensor(inputs)

            features_tensor: torch.Tensor = model.get_image_features(
                outputs.to(device)
            ).cpu().reshape(-1)

        case "resnet18":
            transform = Compose([ToTensor(), Resize((224, 224))])
            inputs: torch.Tensor = transform(
                image
            ).to(device).reshape(1, 3, 224, -1)

            features_tensor: torch.Tensor = model(
                inputs
            ).cpu().reshape(-1)

        case "vit":
            inputs: torch.Tensor = processor(image)["pixel_values"]

            features_tensor: torch.Tensor = model(
                inputs.to(device)
            ).pooler_output.cpu().reshape(-1)

        case "swin":
            inputs: torch.Tensor = processor(
                image)["pixel_values"][0].reshape(1, 3, 224, -1)

            features_tensor: torch.Tensor = model(
                inputs.to(device)
            ).pooler_output.cpu().reshape(-1)

    return features_tensor
