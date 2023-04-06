# read and write files for manual coregistration
import numpy
import pandas


def read_landmarks_from_coreg_csv(csv_fn, column_names, src_prefix, dst_prefix,
                                  column_bool_filters=[],
                                  additional_return_columns=[]):
    # TODO separate out pandas read to allow easier manipulation
    df = pandas.read_csv(
            csv_fn,
            header=None,
            names=column_names)

    if column_bool_filters:
        f = df[column_bool_filters].values.sum(axis=1, dtype=bool)
        filtered_df = df[f]
    else:
        filtered_df = df

    src_pts = filtered_df[
        [f"{src_prefix}x", f"{src_prefix}y", f"{src_prefix}z"]
    ].values
    dst_pts = filtered_df[
        [f"{dst_prefix}x", f"{dst_prefix}y", f"{dst_prefix}z"]
    ].values

    additional_return_columns = [
        filtered_df[c].values for c in additional_return_columns
    ]
    
    return (src_pts, dst_pts, *additional_return_columns)


def write_landmarks_to_coreg_df(column_names, src_prefix, dst_prefix,
                                src_arr, dst_arr,
                                additional_columns_map=None):
    additional_columns_map = (
        additional_columns_map if additional_columns_map is not None else {}
    )

    src_x, src_y, src_z = src_arr.T
    dst_x, dst_y, dst_z = dst_arr.T

    d = dict(additional_columns_map, **{
        f"{src_prefix}x": src_x,
        f"{src_prefix}y": src_y,
        f"{src_prefix}z": src_z,
        f"{dst_prefix}x": dst_x,
        f"{dst_prefix}y": dst_y,
        f"{dst_prefix}z": dst_z,
    })
    
    df = pandas.DataFrame(d, columns=column_names)
    return df


def _format_df(df):
    d = {True: 'true', False: 'false'}
    return df.applymap(lambda x: (d[x] if isinstance(x, bool) else x))


def write_landmarks_to_coreg_csv(csv_fn, *args, **kwargs):
    df = write_landmarks_to_coreg_df(*args, **kwargs)    
    _format_df(df).to_csv(csv_fn, index=False, header=False)
