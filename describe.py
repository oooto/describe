from typing import (
    Hashable,
    List,
    Optional,
)
import numpy as np
import pandas as pd
from pandas.util._validators import validate_percentile
from pandas.io.formats.format import format_percentiles
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)


def describe(data, percentiles=None, include=None, exclude=None):
    if data.ndim == 2 and data.columns.size == 0:
        raise ValueError("Cannot describe a DataFrame without columns")

    if percentiles is not None:
        # explicit conversion of `percentiles` to list
        percentiles = list(percentiles)

        # get them all to be in [0, 1]
        validate_percentile(percentiles)

        # median should always be included
        if 0.5 not in percentiles:
            percentiles.append(0.5)
        percentiles = np.asarray(percentiles)
    else:
        percentiles = np.array([0.25, 0.5, 0.75])

    # sort and check for duplicates
    unique_pcts = np.unique(percentiles)
    if len(unique_pcts) < len(percentiles):
        raise ValueError("percentiles cannot contain duplicates")
    percentiles = unique_pcts

    formatted_percentiles = format_percentiles(percentiles)

    def describe_numeric_1d(series):
        stat_index = (
            ["count", "mean", "std", "min"] + formatted_percentiles + ["max"]
        )
        d = (
            [series.count(), series.mean(), series.std(), series.min()]
            + series.quantile(percentiles).tolist()
            + [series.max()]
        )
        return pd.Series(d, index=stat_index, name=series.name)

    def describe_categorical_1d(data):
        names = ["count", "unique"]
        objcounts = data.value_counts()
        count_unique = len(objcounts[objcounts != 0])
        result = [data.count(), count_unique]
        dtype = None
        if result[1] > 0:
            top, freq = objcounts.index[0], objcounts.iloc[0]
            names += ["top", "freq"]
            result += [top, freq]

        # If the DataFrame is empty, set 'top' and 'freq' to None
        # to maintain output shape consistency
        else:
            names += ["top", "freq"]
            result += [np.nan, np.nan]
            dtype = "object"

        return pd.Series(result, index=names, name=data.name, dtype=dtype)

    def describe_timestamp_1d(data):
        # GH-30164
        stat_index = ["count", "mean", "min"] + formatted_percentiles + ["max"]
        d = (
            [data.count(), data.mean(), data.min()]
            + data.quantile(percentiles).tolist()
            + [data.max()]
        )
        return pd.Series(d, index=stat_index, name=data.name)

    def describe_1d(data):
        if is_bool_dtype(data):
            return describe_categorical_1d(data)
        elif is_numeric_dtype(data):
            return describe_numeric_1d(data)
        elif is_datetime64_any_dtype(data):
            return describe_timestamp_1d(data)
        elif is_timedelta64_dtype(data):
            return describe_numeric_1d(data)
        else:
            return describe_categorical_1d(data)

    if data.ndim == 1:
        return describe_1d(data)
    elif (include is None) and (exclude is None):
        # when some numerics are found, keep only numerics
        data = data.select_dtypes(include=[np.number])
        if len(data.columns) == 0:
            data = data
    elif include == "all":
        if exclude is not None:
            msg = "exclude must be None when include is 'all'"
            raise ValueError(msg)
        data = data
    else:
        data = data.select_dtypes(include=include, exclude=exclude)

    ldesc = [describe_1d(s) for _, s in data.items()]
    # set a convenient order for rows
    names: List[Optional[Hashable]] = []
    ldesc_indexes = sorted((x.index for x in ldesc), key=len)
    for idxnames in ldesc_indexes:
        for name in idxnames:
            if name not in names:
                names.append(name)

    d = pd.concat([x.reindex(names, copy=False) for x in ldesc], axis=1, sort=False)
    d.columns = data.columns.copy()
    return d

if __name__ == '__main__':
    s = pd.Series([1, 2, 3])
    print(describe(s))