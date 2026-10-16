"""SDV-based synthetic data generation with controlled random state."""
import random
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata


def generate_synthetic_data(train_df, num_rows, random_state=42):
    """Fit SDV on the current training dataframe and sample synthetic rows.

    SDV's GaussianCopulaSynthesizer does not expose a consistent public
    random_state argument across supported versions. Seed Python and NumPy
    immediately before fitting/sampling to control the randomness used by
    common dependencies. Reproducibility must still be verified against the
    installed SDV version and recorded package versions.
    """
    if len(train_df) == 0:
        raise ValueError("train_df cannot be empty.")
    if num_rows < 1:
        raise ValueError("num_rows must be at least 1.")

    metadata = Metadata.detect_from_dataframe(data=train_df)
    random.seed(random_state)
    np.random.seed(random_state)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(train_df)

    # Set the random state of SDV's underlying model.
    synthesizer._set_random_state(random_state)

    return synthesizer.sample(num_rows=num_rows)
