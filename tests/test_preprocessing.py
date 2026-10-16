import pandas as pd
from core.preprocessing import TabularPreprocessor


def main():
    train = pd.DataFrame({
        "age": [20, 30, None, 40],
        "job": ["tech", "sales", "tech", None],
        "owns_home": [True, False, True, False],
    })
    prep = TabularPreprocessor().fit(train)
    encoded = prep.transform(train)
    assert list(encoded.columns) == list(train.columns)
    assert encoded.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(encoded[c]) for c in encoded.columns)

    unseen = pd.DataFrame({"age": [25], "job": ["never_seen"], "owns_home": [True]})
    unseen_encoded = prep.transform(unseen)
    assert unseen_encoded.loc[0, "job"] == -1
    assert unseen_encoded.shape == (1, 3)
    print("Preprocessing tests passed.")


if __name__ == "__main__":
    main()
