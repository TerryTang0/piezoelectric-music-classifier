import warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ignore numerical warnings from scikit-learn
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from dataset import assemble_windowed_dataset


def main():
    BASE_DIR = CODE_DIR.parent
    DATA_PATH = BASE_DIR / "Data"

    print(f"Project root: {BASE_DIR}")
    print(f"Loading files from: {DATA_PATH}\n")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data directory not found at: {DATA_PATH}")

    X, y, groups = assemble_windowed_dataset(DATA_PATH, window_sec=5.0, overlap=0.5)

    if len(X) == 0:
        raise ValueError("No data samples were loaded. Check Data folder path.")

    # Clean up any remaining NaN or infinite values
    X = X.clip(lower=-1e5, upper=1e5).fillna(0.0)

    print(f"\nLoaded {len(X)} windows across {len(np.unique(groups))} unique tracks.")
    print("Class distribution:\n", pd.Series(y).value_counts())

    # Setup 5 candidate models for benchmark tournament
    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=42
        ),
        "Support Vector Machine (RBF)": Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                class_weight='balanced',
                random_state=42
            ))
        ]),
        "k-Nearest Neighbors": Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='manhattan'
            ))
        ]),
        "Logistic Regression": Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', LogisticRegression(
                C=0.01,
                solver='liblinear',
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ))
        ])
    }

    # Stratified grouped cross-validation by song track
    sgkf = StratifiedGroupKFold(n_splits=5)
    tournament_summary = []

    best_macro_f1 = -1.0
    best_model_name = None
    best_predictions = None
    best_ground_truth = None

    print("\n" + "=" * 65)
    print("5-MODEL TOURNAMENT BENCHMARK (Scoring: Macro F1 across Unseen Songs)")
    print("=" * 65)

    for model_name, model in candidate_models.items():
        fold_f1_scores = []
        all_y_true = []
        all_y_pred = []

        for train_idx, test_idx in sgkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            fold_macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
            fold_f1_scores.append(fold_macro_f1)

            all_y_true.extend(y_test)
            all_y_pred.extend(preds)

        mean_macro_f1 = float(np.mean(fold_f1_scores))
        std_macro_f1 = float(np.std(fold_f1_scores))

        tournament_summary.append({
            "Model": model_name,
            "Mean Macro F1": round(mean_macro_f1, 4),
            "Std Dev": round(std_macro_f1, 4)
        })

        print(f"Evaluated: {model_name:<30} | Macro F1: {mean_macro_f1:.4f} (± {std_macro_f1:.4f})")

        if mean_macro_f1 > best_macro_f1:
            best_macro_f1 = mean_macro_f1
            best_model_name = model_name
            best_predictions = all_y_pred
            best_ground_truth = all_y_true

    # Tournament leaderboard
    leaderboard_df = pd.DataFrame(tournament_summary).sort_values(by="Mean Macro F1", ascending=False)
    print("\n" + "=" * 65)
    print("FINAL LEADERBOARD")
    print("=" * 65)
    print(leaderboard_df.to_string(index=False))

    # Diagnose for the champion model
    print("\n" + "=" * 65)
    print(f"CROWNED BEST MODEL: {best_model_name}")
    print("=" * 65)

    print("\nDetailed Per-Genre Classification Report:")
    print(classification_report(best_ground_truth, best_predictions, zero_division=0))

    print("Confusion Matrix:")
    labels = sorted(list(np.unique(y)))
    cm = pd.DataFrame(
        confusion_matrix(best_ground_truth, best_predictions, labels=labels),
        index=[f"True_{c}" for c in labels],
        columns=[f"Pred_{c}" for c in labels]
    )
    print(cm)

    # Feature importance breakdown for the tree-based model
    winning_model = candidate_models[best_model_name]
    if hasattr(winning_model, "feature_importances_"):
        winning_model.fit(X, y)
        importances = pd.Series(winning_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\n" + "=" * 65)
        print("RANDOM FOREST FEATURE IMPORTANCES (Gini Gain)")
        print("=" * 65)
        for rank, (feat, score) in enumerate(importances.items(), start=1):
            print(f"{rank:2d}. {feat:<28} : {score:.4f}")


if __name__ == "__main__":
    main()