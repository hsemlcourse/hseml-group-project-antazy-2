#!/usr/bin/env python3
"""Plot feature importances from the saved model."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
from petfinder.constants import DEFAULT_MODEL_PATH, PROJECT_ROOT

OUTPUT_PATH = PROJECT_ROOT / "report" / "assets" / "feature_importance.png"
TOP_N = 15


def main() -> None:
    pipeline = joblib.load(DEFAULT_MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    order = np.argsort(importances)[::-1][:TOP_N]
    top_names = [feature_names[i] for i in order]
    top_values = importances[order]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_names)), top_values[::-1], color="#4C72B0")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature importance (ExtraTrees)")
    ax.set_title("Топ признаков финальной модели")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
