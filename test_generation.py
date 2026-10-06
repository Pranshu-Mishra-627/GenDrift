from core.data_handler import load_dataset, split_dataset
from core.trainer import run_experiment


df = load_dataset("data/iris.csv")

train_df, test_df = split_dataset(
    df,
    "target"
)

results = run_experiment(
    train_df,
    test_df,
    "target",
    generations=10,
    real_ratio=0.7
)

for result in results:
    print(result)