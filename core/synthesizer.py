from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata


def generate_synthetic_data(train_df, num_rows):
    metadata = Metadata.detect_from_dataframe(
        data=train_df
    )

    synthesizer = GaussianCopulaSynthesizer(metadata)

    synthesizer.fit(train_df)

    synthetic_df = synthesizer.sample(
        num_rows=num_rows
    )

    return synthetic_df