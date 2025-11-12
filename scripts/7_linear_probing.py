# general imports
import numpy as np
import pandas as pd
import joblib
import random

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler

# utils
from pathlib import Path

# typing
from typing import List, Literal

# our package
from src.stats import perf_measure


ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome"
CSV_FILE: Path = ROOT / "results" / "people_data.csv"
FEATURES: Path = ROOT / "data" / "features" / "visual_genome" / "images" / "clip" / "VG_100K"
MANUAL_LABELS: Path = ROOT / "results" / "people_data_manual_labels_sports.csv"
# MANUAL_LABELS: Path = ROOT / "results" / "people_data_manual_labels_work.csv"

ATTRIBUTE: str = "is_sport_manual_label"
# ATTRIBUTE: str = "is_work_manual_label"

# SAVE_TO: str = "linear_probing_work.pkl"
SAVE_TO: str = "linear_probing_sport.pkl"

# MODEL: Literal["logistic_regression", "knn"] = "knn"
MODEL: Literal["logistic_regression", "knn"] = "logistic_regression"
OVERSAMPLE: bool = False

SEED: int = 892892


def load_space(df: pd.DataFrame) -> np.array:
    embeddings: List[np.array] = []
    for i in df.index:
        emb_path: Path = FEATURES / f"{i}.npy"
        emb: np.array = np.load(emb_path)
        embeddings.append(emb)
    return np.stack(embeddings)


def print_confusion_mtx(TP: int, FP: int, TN: int, FN: int) -> None:
    print("\tP\tN\t")
    print(f"P\t{TP}\t{FP}\t")
    print(f"N\t{FN}\t{TN}\t")
    print()


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    # load data
    df_labels: pd.DataFrame = pd.read_csv(MANUAL_LABELS).set_index("image_id")
    df: pd.DataFrame = pd.read_csv(CSV_FILE).set_index("image_id")
    df: pd.DataFrame = df[df.contains_person == 1]
    df.drop(df_labels.index, inplace=True)

    # split data
    train, test, y_train, y_test = train_test_split(
        df_labels,
        df_labels[ATTRIBUTE],
        test_size=0.5,
        random_state=SEED,
        stratify=df_labels[ATTRIBUTE]
    )

    # laod space
    train_space: np.array = load_space(train)
    test_space: np.array = load_space(test)
    unlabelled_space: np.array = load_space(df)

    if OVERSAMPLE:
        # oversample minority class
        train_space, y_train = RandomOverSampler(random_state=SEED).fit_resample(
            train_space,
            y_train
        )

    # train model
    match MODEL:
        case "logistic_regression":
            model = LogisticRegression(random_state=SEED, C=0.1, max_iter=1000)
        case "knn":
            model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        case _:
            raise ValueError(f"Unknown model {MODEL}")

    model.fit(train_space, y_train)
    y_pred: np.array = model.predict(test_space)

    TP, FP, TN, FN = perf_measure(y_test.values, y_pred)

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR: float = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR: float = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV: float = TP / (TP + FP)
    # # Negative predictive value
    # NPV: float = TN / (TN + FN)
    # Fall out or false positive rate
    FPR: float = FP / (FP + TN)
    # False negative rate
    FNR: float = FN / (TP + FN)
    # False discovery rate
    FDR: float = FP / (TP + FP)
    # False omission rate
    FOR: float = FN / (TN + FN)

    # Overall accuracy
    accuracy: float = (TP + TN) / (TP + FP + FN + TN)

    print_confusion_mtx(TP, FP, TN, FN)

    print(f"Accuracy: {accuracy:2f}, FDR: {FDR:2f}, FNR: {FNR:2f}, FOR: {FOR:2f}")
    print(f"FPR: {FPR:2f}")
    print(f"Confusion matrix:\nTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

    # predict on unlabelled data
    y_unlabelled_pred: np.array = model.predict(unlabelled_space)

    n_pos: int = (y_unlabelled_pred == 1).sum() + (y_train == 1).sum() + (y_test == 1).sum()
    tot: int = len(y_unlabelled_pred) + len(y_train) + len(y_test)
    print(f"Predicted positive: {n_pos} out of {tot} ({round(100 * n_pos / tot, 2)}%)")

    test["y_pred"] = y_pred
    print("List of FP:",
          list(test[(test.y_pred == 1) & (test[ATTRIBUTE] == 0)].index)
          )
    print("List of FN:",
          list(test[(test.y_pred == 0) & (test[ATTRIBUTE] == 1)].index)
          )

    # save model
    with open(ROOT / "models" / SAVE_TO, "wb") as f:
        joblib.dump(model, f)


if __name__ == "__main__":
    main()
