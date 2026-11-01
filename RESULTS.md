
# Results: Tawseel failed-delivery baseline

## Baseline table

Evaluation on the last 20% of parcels by time, using `make_parcels(4000, seed=0)`
(model trained only on the earlier 80%).

| Model                                      | ROC AUC | PR-AUC | Brier | Base rate |
| ------------------------------------------ | ------- | ------ | ----- | --------- |
| Dummy (prior)                              | 0.500   | 0.220  | 0.172 | 0.220     |
| Rule: address has no number -> "will fail" | 0.550   | 0.242  | n/a   | 0.220     |
| My model                                   | 0.636   | 0.320  | 0.166 | 0.220     |

## Three sentences for Sherif

1. On parcels the model never saw (the most recent 20%), it finds failing deliveries clearly better than guessing: where about 22% of parcels fail overall, the model's top-ranked parcels are right noticeably more often (precision score 0.32 vs 0.22 for a model that only knows the overall failure rate).
2. This is a modest improvement, not magic: it will still miss many failures and flag some good parcels, and its probabilities are only slightly more accurate than the baseline's.
3. I only used information available when the parcel is created (area, time slot, COD amount, address quality, the customer's earlier parcels) and never the courier notes or delivery time, so I expect similar results in production rather than a number that disappears.
