# sg-framing

We study framing bias in Visual Genome dataset. 

## Project Organization

```
├── LICENSE            <- MIT license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── visual_genome      <- Folder where to put VG data.
│       ├── processed          <- Folder for processed VG data.
│       └── images              <- Folder for images
│   
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short description, e.g.
│                         `1_Initial Manual Labelling`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as .txt, HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── src                <- Source code for use in this project.
│
└── scripts            <- Scripts to run the analysis.
    
```

--------

## Data preparation

As many other datasets annotated through crowd sourcing, Visual Genome presents a very noisy labelling. To reduce the impact of it, the authors provide WordNet's synsets to standardise the annotation as much as possible. Hence, when available we adopt the synsets provided by the authors, otherwise we adopt the annotation that is availble after applying minimal text cleaning (removing punctuation, applyinh lowercase and stripping the text).

Furthemore, we reduce object's attributes provided in Visual Genome 1.2 to triples of the type ```(obj, has_attribute, attr)```.

To clean the data run

```bash
cd scripts
python 1_prepare_data.py
```

Visual Genome's images can be downloaded from the website in two batches. Once those folders have been placed in ```data/visual_genome/images```, we run the following code to move everything in one folder.

```bash
cd scripts
python 2_move_images_in_the_same_folder.py
```

Some of the images are corrupted and cannot be opened.

To identify corrupted images run
```bash
cd scripts
python 3_is_corrupted.py
```
This saves a list of corrupted images in ```reports/unidentified_images.txt```.

Finally, since we are only interested in images containing people we filter others out. We keep the images that have been annotated with the words "person", "people", "man", "woman", "child", "boy", "girl" in the Visual Genome dataset. In addition to this, since the labelling is noisy and some people are labelled differently (e.g., as "worker") we apply an [object detection algorithm](https://huggingface.co/facebook/detr-resnet-50) and keep the images that contain at least one person.

Note that we apply a confidence threshold of 0.97 to limit false positives as much as possible.

```bash
cd scripts
python 4_classify_people.py
```
--------
## Topic discovery

We apply topic modelling to discover the main topics in the dataset.

Run the following script

```bash
cd scripts
python 5_topic_discovery.py
```
-------
## Linear Probing

After analysing the results of the topic modelling, we formulate hypothesis on the content of the data and apply linear probing model to quantify such content.

First we label a short portion of the data (1.000 images) to use as a test set. See ```./notebooks/Initial Labelling.ipynb```.

Then, to extract CLIP features and apply linear probing, run
```bash
cd scripts
python 6_extract_features.py
python 7_linear_probing.py
```

