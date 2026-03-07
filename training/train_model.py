# training/train_model.py
"""
Train gesture classifier from dataset/gestures.csv.

Usage:
    python -m training.train_model

Output:
    models/model.pkl  — best SVC model + label_map + best_params bundled together
"""

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config
from core.gestures import GESTURE_LABELS


def load_dataset():
    data = np.genfromtxt(config.DATASET_PATH, delimiter=",", dtype=str, skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    labels_str = data[:, 0]
    features = data[:, 1:].astype(np.float32)
    label_to_int = {label: idx for idx, label in enumerate(GESTURE_LABELS)}
    y = np.array([label_to_int[l] for l in labels_str], dtype=np.int32)
    return features, y


def augment(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Data augmentation:
      1. Mirror: flip x-coords of all 21 landmarks (simulates left hand)
      2. Jitter: add Gaussian noise × JITTER_COPIES copies

    Feature layout: indices 0,3,6,...,60 are x-coords of landmarks 0-20.
    The palm-normal (indices 63-65) is left as-is for jitter, zeroed for mirror.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    aug_X, aug_y = [X], [y]

    # Mirror: negate x-coords (every 3rd value starting at 0, for first 63 features)
    mirrored = X.copy()
    mirrored[:, 0:63:3] *= -1      # flip x of 21 landmarks
    mirrored[:, 63:66] = 0.0       # palm normal undefined after flip
    aug_X.append(mirrored)
    aug_y.append(y)

    # Jitter
    for _ in range(config.JITTER_COPIES):
        noise = rng.normal(0, config.JITTER_SIGMA, X.shape).astype(np.float32)
        aug_X.append(X + noise)
        aug_y.append(y)

    return np.concatenate(aug_X), np.concatenate(aug_y)


def main():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")

    unique = np.unique(y)
    print(f"  Classes found: {[GESTURE_LABELS[i] for i in unique]}")
    if len(unique) < 2:
        print("ERROR: Need at least 2 gesture classes to train.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )
    unique_test = np.unique(y_test)

    print(f"\nAugmenting training data...")
    X_train_aug, y_train_aug = augment(X_train, y_train)
    print(f"  Before: {len(X_train)}  After: {len(X_train_aug)}")

    print("\nRunning GridSearchCV (SVC, rbf kernel)...")
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    grid = GridSearchCV(
        SVC(kernel="rbf", probability=True),
        config.SVC_PARAM_GRID,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_aug, y_train_aug)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV score: {grid.best_score_:.4f}")

    best_svc = grid.best_estimator_
    y_pred = best_svc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=[GESTURE_LABELS[i] for i in unique_test],
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    label_map = {idx: label for idx, label in enumerate(GESTURE_LABELS)}
    payload = {
        "model": best_svc,
        "label_map": label_map,
        "best_params": grid.best_params_,
        "feature_dim": config.FEATURE_DIM,
    }
    joblib.dump(payload, config.MODEL_PATH)
    print(f"\nSaved model to '{config.MODEL_PATH}'")
    print(f"Label map: {label_map}")


if __name__ == "__main__":
    main()
