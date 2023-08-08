import pandas as pd
import numpy as np
import functools

from eis_toolkit.exceptions import UnFavorableClassDoesntExistException, FavorableClassDoesntExistException


def _reclass_gen(
    df: pd.DataFrame, stud_cont: float = 2
) -> pd.DataFrame:

    df['Rcls'] = df.Stud_Cont.ge(stud_cont)[::-1].cummax()+1

    df_stud_cont = df[['Class', 'contrast', 'Stud_Cont', 'Rcls']].round(4)

    if 1 not in df['Rcls'].values:
        print(df_stud_cont)
        raise UnFavorableClassDoesntExistException("""Exception error: Class doesn't exist. 
    For the given studentized contrast value, none of the classes were classified to the 'Unfavorable' class, i.e., class 1. 
    Check the displayed studentized contrast values ('Stud_Cont') and run weights calculations again using a suitable threshold value.""")

    elif 2 not in df['Rcls'].values:
        print(df_stud_cont)
        raise FavorableClassDoesntExistException("""Exception error: Class doesn't exist. 
        For the given studentized contrast value none of the classes were classified to the 'Favorable' class, i.e., class 2. 
        Check the displayed studentized contrast values ('Stud_Cont') and run weights calculations again using a suitable threshold value.""")

    else:
        df_ = df.round(4).sort_values(by="Class", ascending=True)
        return df_


def reclass_gen(
        df: pd.DataFrame, stud_cont: float = 2
) -> pd.DataFrame:
    """Performs reclassification of classes into favourable and unfavourable categories for ordianl data, based on studentized contrast values provided by the user.
    Args:
        df (pandas.DataFrame): The dataframe with all the weights calculations; obtained from the contrast function
        stud_cont (float, def = 2): The threshold for studentized contrast for reclassification
    Returns:
        df_rcls (pandas.DataFrame): The dataframe with data classes categorized as favourable and unfavourable
    Raises: 
        UnFavorableClassDoesntExistException: Failure to generalize classes using the given studentised contrast threshold value. Class 1 (unfavorable class) doesn't exist
        FavorableClassDoesntExistException: Failure to generalize classes using the given studentised contrast threshold value. Class 2 (favorable class) doesn't exist
    """
    df_rcls = _reclass_gen(df, stud_cont)
    return df_rcls


def gen_weights(
    df: pd.DataFrame, gen_cls: int
) -> pd.DataFrame:
    """_summary_
    Args:
        df (pd.DataFrame): 
        gen_cls (int): List of generalized class values
    Returns:
        df_gen_wgts (pd.DataFrame): Dataframe with positives weights of associations for generalized classes
    Raises:

    """
    df_rc = df.groupby('Rcls')
    df_rc_ = df_rc.get_group(gen_cls)
    df_gen_wgts = (df_rc_
                   .assign(cmltv_dep_cnt=lambda df_rc_: df_rc_.Point_Count.cumsum())
                   .assign(cmltv_cnt=lambda df_rc_: df_rc_.Count.cumsum())
                   .assign(cmltv_no_dep_cnt=lambda df_rc_: df_rc_.No_Dep_Cnt.cumsum())
                   .iloc[[-1]]
                   )

    return (df_gen_wgts
            .assign(Num_wpls=lambda df_gen_wgts: df_gen_wgts.cmltv_dep_cnt/df_gen_wgts.Total_Deposits)
            .assign(Deno_wpls=lambda df_gen_wgts: df_gen_wgts.cmltv_no_dep_cnt/df_gen_wgts.Tot_No_Dep_Cnt)
            .assign(W_Gen=lambda df_gen_wgts: np.log(df_gen_wgts.Num_wpls)-np.log(df_gen_wgts.Deno_wpls))
            .assign(var_wpls_gen=lambda df_gen_wgts: (1/df_gen_wgts.cmltv_dep_cnt)+(1/df_gen_wgts.cmltv_no_dep_cnt))
            .assign(s_wpls_gen=lambda df_gen_wgts: np.sqrt(df_gen_wgts.var_wpls_gen))
            )


def gen_weights_finalization(
        df: pd.DataFrame
) -> pd.DataFrame:
    """ Calls the gen_weights function and calculates positives weights of associations for each generalized class

    Args:
        df (pd.DataFrame): Dataframe with weights of associations
    Returns:
        df (pd.DataFrame): Dataframe with positives weights of associations for generalized classes
    """
    gen_cls = [1, 2]
    gw_1, gw_2 = map(functools.partial(gen_weights, df), gen_cls)
    gw = pd.concat([gw_1, gw_2])
    for i, clss in enumerate(gen_cls):
        w = gw.iloc[i, 31]
        s_w = gw.iloc[i, 33]
        v_w = gw.iloc[i, 32]
        df.loc[df.Rcls == clss, 'W_Gen'] = w
        df.loc[df.Rcls == clss, 's_wpls_gen'] = s_w
        df.loc[df.Rcls == clss, 'var_wpls_gen'] = v_w
        df.round(4)
    return df
