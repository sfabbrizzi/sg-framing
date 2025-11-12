# general imports
import json
import pandas as pd

# sklearn
from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# plots
from matplotlib import pyplot as plt

# utils
from pathlib import Path
from lightning import seed_everything
from time import time

# typing
from typing import List, Dict

# src
from src.text import find_stopwords


def plot_top_words(
    model: LatentDirichletAllocation | NMF | MiniBatchNMF,
    feature_names: List[str],
    n_top_words: int,
    title: str,
    save_path: Path,
    topic_titles: List[str] | None = None
) -> None:
    """This function creates a bar plot for the top words in each topic.
    This function is from:
    https://scikit-learn.org/stable/auto_examples/applications/\\
        plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples\\
        -applications-plot-topics-extraction-with-nmf-lda-py
    Parameters
    ----------
    model : LatentDirichletAllocation | NMF | MiniBatchNMF
        Topic modelling model.
    feature_names : List[str]
        List of feature names.
    n_top_words : int
        Number of top words to display.
    title : str
        Title of the plot.
    topics_titles: List[str] | None
        List of titles for the topics. Default None.
    """
    n_comp: int = len(model.components_)
    n_rows: int = n_comp // 3 if n_comp % 3 == 0 else n_comp // 3 + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(100, 150), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        if topic_titles is None:
            ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 120})
        else:
            ax.set_title(f"{topic_idx + 1}. {topic_titles[topic_idx]}", fontdict={"fontsize": 120})
        # FIXME - labels
        ax.tick_params(axis="both", which="major", labelsize=110)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=150)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.savefig(save_path)
    plt.clf()


ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome"
DATA_FILE: Path = ROOT / "data" / "visual_genome" / "processed" / "merged_data.json"
CSV_FILE: Path = ROOT / "results" / "people_data.csv"
SAVE_FOLDER: Path = ROOT / "reports" / "figures"

SEED: int = 1991

N_FEATURES: int = 1000
N_COMPONENTS: int = 15
N_TOP_WORDS: int = 15
BATCH_SIZE: int = 128
INIT: str = "nndsvda"

# this list was derived after repeating the experiment with the same SEED
LDA_TOPIC_TITLES: List[str] = [
    "Riding",
    "Food",
    "Traffic",
    "Sport",
    "Sport/Outdoor",
    "Sport",
    "Indoor",
    "Traffic",
    "Sport",
    "Sport",
    "Unclear",
    "Outdoor",
    "Unclear",
    "Unclear",
    "Outdoor"


]


def main() -> None:
    """Code adpted from
    https://scikit-learn.org/stable/auto_examples/applications/\\
        plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples\\
        -applications-plot-topics-extraction-with-nmf-lda-py
    """
    seed_everything(SEED)

    with open(DATA_FILE, "r") as f:
        json_data: Dict[Dict[str, List[Dict]]] = json.load(f)

    df: pd.DataFrame = pd.read_csv(CSV_FILE)

    docs: List[str] = list()
    for image_id, image in json_data.items():
        if int(image_id) not in df[df.contains_person == 1].image_id.values:
            continue
        objects: List[str] = list()
        doc: str = ""
        for rel in image["relationships"]:
            objects += [rel["subject"], rel["object"]]
            doc: str = " ".join(
                [doc]
                + rel["subject"].split("_")
                + rel["predicate"].split("_")
                + rel["object"].split("_")
            )

        docs.append(doc)

    stopwords: List[str] = find_stopwords(docs)

    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=N_FEATURES, stop_words=stopwords,
    )
    tfidf = tfidf_vectorizer.fit_transform(docs)

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(
        max_df=0.95, min_df=2, max_features=N_FEATURES, stop_words=stopwords,
    )

    tf = tf_vectorizer.fit_transform(docs)

    print()
    # Fit the NMF model
    print(
        "Fitting the NMF model (Frobenius norm) with tf-idf features, "
        "n_samples=%d and n_features=%d..." % (len(docs), N_FEATURES)
    )
    t0 = time()
    nmf = NMF(
        n_components=N_COMPONENTS,
        random_state=SEED,
        init=INIT,
        beta_loss="frobenius",
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=1,
    ).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        nmf,
        tfidf_feature_names,
        N_TOP_WORDS,
        "Topics in NMF model (Frobenius norm)",
        SAVE_FOLDER / "nmf_frobenius.pdf"
    )

    # Fit the NMF model
    print(
        "\n" * 2,
        "Fitting the NMF model (generalized Kullback-Leibler "
        "divergence) with tf-idf features, n_samples=%d and n_features=%d..."
        % (len(docs), N_FEATURES),
    )
    t0 = time()
    nmf = NMF(
        n_components=N_COMPONENTS,
        random_state=SEED,
        init=INIT,
        beta_loss="kullback-leibler",
        solver="mu",
        max_iter=1000,
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    ).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        nmf,
        tfidf_feature_names,
        N_TOP_WORDS,
        "Topics in NMF model (generalized Kullback-Leibler divergence)",
        SAVE_FOLDER / "nmf_kl.pdf"
    )

    # Fit the MiniBatchNMF model
    print(
        "\n" * 2,
        "Fitting the MiniBatchNMF model (Frobenius norm) with tf-idf "
        "features, n_samples=%d and n_features=%d, batch_size=%d..."
        % (len(docs), N_FEATURES, BATCH_SIZE),
    )
    t0 = time()
    mbnmf = MiniBatchNMF(
        n_components=N_COMPONENTS,
        random_state=SEED,
        batch_size=BATCH_SIZE,
        init=INIT,
        beta_loss="frobenius",
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    ).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        mbnmf,
        tfidf_feature_names,
        N_TOP_WORDS,
        "Topics in MiniBatchNMF model (Frobenius norm)",
        SAVE_FOLDER / "minibatch_nmf_frobenius.pdf"
    )

    # Fit the MiniBatchNMF model
    print(
        "\n" * 2,
        "Fitting the MiniBatchNMF model (generalized Kullback-Leibler "
        "divergence) with tf-idf features, n_samples=%d and n_features=%d, "
        "batch_size=%d..." % (len(docs), N_FEATURES, BATCH_SIZE),
    )
    t0 = time()
    mbnmf = MiniBatchNMF(
        n_components=N_COMPONENTS,
        random_state=SEED,
        batch_size=BATCH_SIZE,
        init=INIT,
        beta_loss="kullback-leibler",
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    ).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        mbnmf,
        tfidf_feature_names,
        N_TOP_WORDS,
        "Topics in MiniBatchNMF model (generalized Kullback-Leibler divergence)",
        SAVE_FOLDER / "minibatch_nmf_kl.pdf"
    )

    print(
        "\n" * 2,
        "Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
        % (len(docs), N_FEATURES),
    )
    lda = LatentDirichletAllocation(
        n_components=N_COMPONENTS,
        max_iter=5,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    )
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    tf_feature_names = tf_vectorizer.get_feature_names_out()
    plot_top_words(
        lda,
        tf_feature_names,
        N_TOP_WORDS,
        "Topics in LDA model",
        SAVE_FOLDER / "lda.pdf",
        topic_titles=LDA_TOPIC_TITLES
    )


if __name__ == "__main__":
    main()
