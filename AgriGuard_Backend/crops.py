"""Crop vocabulary shared by plot validation and the AI request.

This mirrors `agriguard.irrigation` in the AI service:

* TRAINED_CROPS  - crops the irrigation model has indicator columns for
  (tomato, rice, pepper). These are the only crops scored by the model.
* RULE_ONLY_CROPS - crops the rule-based threshold table knows but the model
  was not trained on (maize, wheat, potato, soybean, onion, chili). The AI
  answers these from the rule and reports decision_source 'rule_fallback'.

The two lists are duplicated here on purpose. The backend and the AI service
run as separate processes, and importing the AI package would couple the
backend's startup to the AI's dependencies. `tests/test_crops.py` pins this
split against the shipped model, so a retrain that changes the trained crops
fails the backend tests rather than drifting silently.
"""

TRAINED_CROPS = ("tomato", "rice", "pepper")

RULE_ONLY_CROPS = ("chili", "maize", "onion", "potato", "soybean", "wheat")

KNOWN_CROPS = TRAINED_CROPS + RULE_ONLY_CROPS


def normalise_crop(crop_type):
    """Lower-cased, stripped crop name, or None when absent/blank.

    The AI matches crop names case-insensitively, so validation does too;
    rejecting "Tomato" here while the AI accepts it would be a false negative.
    """
    if crop_type is None:
        return None
    name = str(crop_type).strip().lower()
    return name or None


def is_known_crop(crop_type):
    return normalise_crop(crop_type) in KNOWN_CROPS
