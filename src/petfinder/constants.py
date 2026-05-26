"""Constants for PetFinder inference."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_DEFAULTS_PATH = PROJECT_ROOT / "models" / "feature_defaults.json"

TARGET_COLUMN = "AdoptionSpeed"

# Columns expected at inference (same as train.csv without target).
FEATURE_COLUMNS = [
    "Type",
    "Name",
    "Age",
    "Breed1",
    "Breed2",
    "Gender",
    "Color1",
    "Color2",
    "Color3",
    "MaturitySize",
    "FurLength",
    "Vaccinated",
    "Dewormed",
    "Sterilized",
    "Health",
    "Quantity",
    "Fee",
    "State",
    "RescuerID",
    "VideoAmt",
    "Description",
    "PetID",
    "PhotoAmt",
]

# Subset asked in Telegram wizard; rest filled from defaults.
WIZARD_FIELDS = [
    "Type",
    "Age",
    "Breed1",
    "Gender",
    "Color1",
    "Sterilized",
    "Health",
    "Vaccinated",
    "Dewormed",
    "Fee",
    "PhotoAmt",
    "Name",
]

CLASS_LABELS_RU = {
    0: "Пристройство в день публикации",
    1: "Пристройство за 1–7 дней",
    2: "Пристройство за 8–30 дней",
    3: "Пристройство после 100 дней",
    4: "Долго не пристроилось",
}
