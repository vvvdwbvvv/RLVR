import numpy as np
import pandas as pd


ROOT_DIR = '~'
fact = pd.read_csv("map.csv", dtype=str, encoding="utf-8-sig", low_memory=False)
inspection  = pd.read_csv(f"{ROOT_DIR}/inspection.csv", dtype=str, encoding="utf-8-sig", low_memory=False)

# --- existence check ---
for dfname, df, cols in [
    ("map.csv", fact, {"病歷編號", "就診日期"}),
    ("inspection.csv", inspection,  {"病歷編號", "檢驗日期"}),
]:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{dfname} miss: {missing}")

_FULL2HALF = str.maketrans("０１２３４５６７８９", "0123456789")
def normalize_digits_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip().apply(lambda x: x.translate(_FULL2HALF))

# yyyMMdd to yyyyMMdd
def parse_roc7_to_datetime(s: pd.Series) -> pd.Series:
    s0 = normalize_digits_str(s)
    n1 = pd.to_numeric(s0, errors="coerce")
    digits_only = s0.str.replace(r"\D", "", regex=True)
    n2 = pd.to_numeric(digits_only, errors="coerce")
    n  = n1.where(~n1.isna(), n2)

    mask7 = (n >= 1_000_000) & (n <= 9_999_999)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if mask7.any():
        n7 = n[mask7].astype("Int64")
        y_roc = (n7 // 10_000).astype("Int64")
        m     = ((n7 // 100) % 100).astype("Int64")
        d     = (n7 % 100).astype("Int64")
        y_ad  = (y_roc + 1911).astype("Int64")
        iso = (
            y_ad.astype(str).str.zfill(4) + "-" +
            m.astype(str).str.zfill(2) + "-" +
            d.astype(str).str.zfill(2)
        )
        out.loc[mask7] = pd.to_datetime(iso, errors="coerce", format="%Y-%m-%d")
    return out.dt.normalize()

def parse_mixed_to_datetime(s: pd.Series) -> pd.Series:
    s1 = normalize_digits_str(s).str.replace("/", "-", regex=False)
    return pd.to_datetime(s1, errors="coerce", format="mixed").dt.normalize()

fact["就診日期"]     = parse_roc7_to_datetime(fact["就診日期"])
inspection["檢驗日期"] = parse_mixed_to_datetime(inspection["檢驗日期"])
fact = fact.rename(columns={"就診日期": "檢驗日期"})

# nomalize
for col in ["病歷編號"]:
    fact[col] = normalize_digits_str(fact[col])
    inspection[col] = normalize_digits_str(inspection[col])


KEYS = ["病歷編號", "檢驗日期"]
inspection_non_keys = [c for c in inspection.columns if c not in KEYS]

grp = inspection.groupby(KEYS, dropna=False)


row_count = grp.size().rename("row_count")
if inspection_non_keys:
    def _rows_struct(g: pd.DataFrame):
        return g[inspection_non_keys].to_dict("records")
    rows_struct = grp.apply(_rows_struct).rename("rows_struct")
else:
    rows_struct = pd.Series([], dtype=object, name="rows_struct")

uniq_series_list = []
for c in inspection_non_keys:
    s = grp[c].agg(lambda x: sorted(pd.unique(x.dropna()))).rename(f"{c}_uniq")
    uniq_series_list.append(s)

parts = [row_count]
if inspection_non_keys:
    parts.append(rows_struct)
    parts.extend(uniq_series_list)

agg_right = pd.concat(parts, axis=1).reset_index()

out = fact.merge(agg_right, on=KEYS, how="left")
out.to_csv("map_with_inspection_exact_cpu.csv", index=False)


fact_keys = fact[KEYS].drop_duplicates()
insp_keys = inspection[KEYS].drop_duplicates()
miss = fact_keys.merge(insp_keys, on=KEYS, how="left", indicator=True)\
                .query('_merge == "left_only"')\
                .drop(columns=["_merge"])
print("miss:", len(miss))
print(miss.head(10))
