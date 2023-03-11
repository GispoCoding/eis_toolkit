import pandas as pd

from eis_toolkit.prediction.weights_of_evidence.weights_generalizations import reclass_gen, gen_weights_finalization
from eis_toolkit.prediction.weights_of_evidence.save_weights import save_weights

def _weights_generalization(
    df_wgts: pd.DataFrame, w_type: int, stud_cont: float = 2
) -> pd.DataFrame:

    rcls_df = reclass_gen(df_wgts, stud_cont)
    wgts_gen = gen_weights_finalization(rcls_df)
    wgts_fnl = save_weights(wgts_gen, w_type)
    return wgts_fnl

def weights_generalization(
    df_wgts: pd.DataFrame,  w_type: int, stud_cont: float = 2


) -> pd.DataFrame:
    """Identifies the favourable and unfavorable classes based on the weights for ascending and descending weights and recalculates the generalized weitghts
    Args:
        df_wgts (pandas.DataFrame): dataframe with the weights
        stud_cont (float, def = 2): studentized contrast value to be used for genralization of classes
    Returns:
        wgts_fnl (pandas.DataFrame): dataframe with generalized weights and generalized classes
    Raises:
    """
    wgts_fnl=_weights_generalization(df_wgts, w_type, stud_cont)
    return wgts_fnl
