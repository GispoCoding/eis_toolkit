import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence


@beartype
def _ILR_transform(df: pd.DataFrame, columns: Optional[Sequence[str]]) -> pd.DataFrame:
    """TODO: docstring."""
    return np.log(df)
