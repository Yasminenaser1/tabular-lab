import numpy as np
import pandas as pd

RANGES = [
    ((390, 460), "circulatory"), ((785, 786), "circulatory"),
    ((460, 520), "respiratory"), ((786, 787), "respiratory"),
    ((520, 580), "digestive"),   ((787, 788), "digestive"),
    ((800, 1000), "injury"),
    ((710, 740), "musculoskeletal"),
    ((580, 630), "genitourinary"), ((788, 789), "genitourinary"),
    ((140, 240), "neoplasms"),
]


def group_code(code):
    if pd.isna(code):
        return "missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "other"
    try:
        num = float(code)
    except ValueError:
        return "other"
    if 250 <= num < 251:
        return "diabetes"
    for (lo, hi), name in RANGES:
        if lo <= num < hi:
            return name
    return "other"


def add_diag_groups(df):
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_group"] = df[col].map(group_code)
    return df
