from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted


def get_params_from_path(res_path):
    res_path = Path(res_path)
    p_str = res_path.parent.name
    splits = p_str.split(';')
    params = {}

    for s in splits:
        ss = s.split('=')
        k = ss[0]
        v = '='.join(ss[1:])

        try:
            v = float(v)
        except ValueError:
            v = str(v)

        params[k] = v

    return params


def collate_results(output_path, metric='top1', value='max', filename='summary.csv', times_100=False,
                    precision=2, apply_style=False, cmap='Greens', ignore_params=(), filter_dict=None,
                    filter_if_not_present=(), res_parts=2, filter_mode='include', sort_by=None,
                    extra_metrics=(), extra_metrics_val=(), rename_p=None, drop_cols=None, drop_paths=True,
                    extra_metrics_from_same_model=True):
    files = Path(output_path).rglob(f'**/{filename}')
    rows = []
    output_path_parts = set(Path(output_path).parts)

    if not extra_metrics_from_same_model:
        print('Warning! You are going to get extra metrics from potentially different epochs/models!')
        assert len(extra_metrics) == len(extra_metrics_val), 'Provide min/max val for each extra metric'

    filter_dict = {} if filter_dict is None else filter_dict

    for res_path in files:
        res_path = Path(res_path)
        run_params = get_params_from_path(res_path)
        df = pd.read_csv(res_path)
        path_diff = set(res_path.parts[:-res_parts]) - output_path_parts
        run_params.update({f'p_{i}': p for i, p in enumerate(sorted(path_diff))})

        if isinstance(filter_if_not_present, dict):
            keep = all([run_params.get(k, None) == v for k, v in filter_if_not_present.items()])

            if not keep:
                continue
        else:
            if filter_if_not_present and not set(filter_if_not_present).intersection(set(run_params.keys())):
                continue

        add_row = True if len(filter_dict) == 0 or filter_mode == 'exclude' else False
        filter_inc_count = 0

        for k, v in run_params.items():
            if k in ignore_params:
                continue

            if filter_dict.get(k, None) == v:
                if filter_mode == 'exclude':
                    add_row = False
                    break
                else:
                    filter_inc_count += 1

        if filter_mode == 'include' and len(filter_dict) > 0:
            add_row = filter_inc_count == len(filter_dict)

        if add_row:
            row = run_params.copy()
            val = df[metric].max() if value == 'max' else df[metric].min()
            row[metric] = val * 100 if times_100 else val
            epoch_idx = np.where(df[metric] == val)[0][-1]  # take the last epoch with the best value

            for iem, em in enumerate(extra_metrics):
                if em not in df:
                    ev = None
                else:
                    if extra_metrics_from_same_model:
                        ev = df[em].values[epoch_idx]
                    else:
                        emv = extra_metrics_val[iem]
                        ev = df[em].max() if emv == 'max' else df[em].min()

                row[em] = ev * 100 if times_100 else ev

            row['path'] = res_path
            rows.append(row)

    df = pd.DataFrame(rows)

    if ignore_params:
        df = df.drop(columns=list(ignore_params))

    fixed_params = {}
    varying_params = []

    for c in df.columns:
        if c == metric:
            continue

        u_p = df[c].unique()

        if len(u_p) == 1:
            fixed_params[c] = u_p[0]
        else:
            varying_params.append(c)

    if len(df) > 1:
        df = df[varying_params + [metric]]  # + list(extra_metrics)]

    if len(varying_params) == 1:
        df = df.set_index(varying_params[0]).sort_index()
    elif len(varying_params) == 2 and len(extra_metrics) == 0:
        index, new_col = varying_params[0], varying_params[1]
        new_rows = []
        indices = df[index].unique()

        for i in indices:
            a = df[df[index] == i][[new_col, metric]]
            d = {'{}={}'.format(new_col, getattr(t[1], new_col)): getattr(t[1], metric) for t in a.iterrows()}
            d[index] = i
            new_rows.append(d)

        df = pd.DataFrame(new_rows)
        df = df.set_index(index)
        df = df.reindex(index=natsorted(df.index))
        df = df.reindex(columns=natsorted(df.columns))

    if sort_by is not None and len(varying_params) != 2:
        df = df.sort_values(by=sort_by, ignore_index=True)

    if rename_p is not None:
        df = df.rename(columns=rename_p)

    if apply_style:
        styled_df = df.style.background_gradient(cmap=cmap, axis=None).set_precision(precision)

        if value == 'max':
            styled_df = styled_df.highlight_max(axis=None, color='darkorange')
        else:
            styled_df = styled_df.highlight_min(axis=None, color='darkorange')
    else:
        styled_df = None

    paths = {t.Index: t.path for t in df.itertuples()}

    if drop_paths:
        df = df.drop(columns='path')

    if drop_cols is not None:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df, fixed_params, styled_df, paths



