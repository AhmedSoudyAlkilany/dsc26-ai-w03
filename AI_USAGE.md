
# AI Usage

## Tools I used, and for what

- **ChatGPT:** Used to understand the task requirements, review the implementation of `add_history_features` and `time_split`, inspect potential edge cases, and review the scikit-learn pipeline and evaluation logic.
- **Claude:** Used to review the implementation and discuss possible improvements to the baseline comparison.
- **Python / pytest:** Used to run the public tests and verify the model's behavior locally.

## AI-suggested code I rejected or fixed in add_history_features or time_split

During the code review, I checked the history feature implementation carefully, especially how it handles shuffled rows, non-default indexes, and parcels with identical timestamps.

I verified that:

- Parcels with the same customer and timestamp do not use each other's outcomes as historical information.
- Historical features are calculated using only strictly earlier timestamps.
- The original row order and index are preserved.
- The time-based split keeps all rows at or after the test boundary in the test set, including timestamp ties.

I also updated `compare_to_baseline` so the DummyClassifier is trained using only the allowed feature columns and history features.

## How I checked that my model has no leak

I checked that the model uses only information available when a parcel is created.

- Excluded `failed`, `delivered_at`, `courier_notes`, and `parcel_id` from the model features.
- Built customer history features using only outcomes from strictly earlier timestamps.
- Used a chronological train/test split instead of a random split.
- Trained both the model and DummyClassifier on the training data only.
- Evaluated performance on the held-out test set.

I ran `uv run pytest` locally, and all 5 public tests passed.

I also tested multiple random seeds and checked that ROC-AUC remained below 0.9, while PR-AUC lift exceeded the required threshold. For seed 0, the model achieved a PR-AUC of 0.3203 compared with 0.2200 for the DummyClassifier, with a Brier score of 0.1659 compared with 0.1717 for the baseline.

<!-- One sentence. -->
