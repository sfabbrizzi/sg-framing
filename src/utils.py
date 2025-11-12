# general imports
import json

# utils
from pathlib import Path

# typing
from typing import List, Literal, Dict
from os import PathLike

from src.text import clean_text


def read_token(file_name: PathLike) -> str:
    """Reads a token from a file.

    Parameters
    ----------
    file_name : PathLike
        The path to the file containing the token.

    Returns
    -------
    str
        The token read from the file.
    """

    with open(file_name, "r") as f:
        token: str = f.readline().strip()

    return token


def load_json_data(
    root: PathLike,
    json_file: Literal["objects", "attributes", "relationships"]
) -> List[dict]:
    """Load JSON data from the Visual Genome dataset.

    Parameters
    ----------
    root : PathLike
        The root directory where the JSON files are located.
    json_file : Literal["objects", "attributes", "relationships"]
        The file to load.

    Returns
    -------
    List[dict]
        The loaded JSON data.
    """
    root = Path(root)

    file_map = {
        "objects": root / "objects.json",
        "attributes": root / "attributes.json",
        "relationships": root / "relationships.json"
    }

    if json_file not in file_map:
        raise ValueError(
            f"Invalid type: {json_file}. Must be one of {list(file_map.keys())}."
        )

    with open(file_map[json_file], "r") as f_json:
        data: List[dict] = json.load(f_json)

    return data


def extraxt_obj_info(json_d: Dict, synsets: bool = True) -> str:
    """Extracts the name or synset from a JSON dictionary.

    Parameters
    ----------
    json_d : Dict
        element of the item rel_data["relationships"],
        where rel_data is the relationship JSON from
        Visual Genome.
    synsets : bool, optional
        If True, the function will try to extract the synset
        from the "synsets" key in the json_d.

    Returns
    -------
    str
        synset or name for subject, object or relationship.

    Raises
    ------
    ValueError
        If no name or synset is found in the json_d.
        This can happen if the json_d is not well-formed
        or does not contain the expected keys.
    """

    if synsets and "synsets" in json_d and len(json_d["synsets"]) > 0:
        return json_d["synsets"][0].split(".")[0]
    elif "predicate" in json_d:
        return "_".join(clean_text(json_d["predicate"]).split())
    elif "names" in json_d and len(json_d["names"]) > 0:
        return "_".join(clean_text(json_d["names"][0]).split())
    elif "name" in json_d:
        return "_".join(clean_text(json_d["name"]).split())
    else:
        raise ValueError("No name found in json data")


def load_corrupted_images(root: Path) -> List[str]:
    """
    Load the list of corrupted images from the
    'unidentified_images.txt' file."""
    corrupted: List[str] = list()
    with open(root / "reports" / "unidentified_images.txt", "r") as f:
        lines: List[str] = f.readlines()[1:]
        corrupted = [line.strip() for line in lines]
    return corrupted


if __name__ == "__main__":
    print(load_corrupted_images(Path("../")))
