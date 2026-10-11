from data import generate
from model import train, fairness_audit, save, feature_importances
from engine.fairness_extra import disparity_summary

df = generate()
model, metrics, test_df = train(df)
save(model)
print("Metrics:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})
print("\nFairness audit:\n", fairness_audit(test_df).to_string(index=False))
print("\nDisparity:\n", disparity_summary(test_df).to_string(index=False))
print("\nImportances:\n", feature_importances(model).to_string(index=False))
