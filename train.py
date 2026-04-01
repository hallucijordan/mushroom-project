"""
train.py — Train and save the base learner.

Run once after downloading the data:
    conda activate mushroom-project
    python download_data.py
    python train.py

Artefacts written to models/:
    models/preprocessor.joblib
    models/base_learner.joblib
"""

from sklearn.model_selection import train_test_split

from src.data.preprocessor import MushroomPreprocessor, load_secondary
from src.models.base_learner import BaseLearner


def main():
    print("Loading secondary data...")
    X_raw, y = load_secondary()
    print(f"  {len(y)} samples | class balance: "
          f"{(y==0).sum()} edible / {(y==1).sum()} poisonous")

    print("Preprocessing...")
    prep = MushroomPreprocessor()
    X = prep.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training BaseLearner on {len(y_train)} samples...")
    learner = BaseLearner()
    learner.fit(X_train, y_train)

    metrics = learner.evaluate(X_test, y_test)
    print(f"\nTest metrics ({len(y_test)} held-out samples):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    prep.save()
    learner.save()
    print("\nDone.  Artefacts written to models/")


if __name__ == "__main__":
    main()
