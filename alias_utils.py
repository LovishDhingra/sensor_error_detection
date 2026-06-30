import re
from difflib import get_close_matches

def normalize_col(col):
    if col is None:
        return ""
    s = str(col).lower().strip()
    s = re.sub(r'[^0-9a-z]', '', s)  # remove punctuation
    return s

SYNONYMS = {
    "mq2": "mq2_value",
    "mq2value": "mq2_value",
    "mq2_value": "mq2_value",

    "temp": "temperature",
    "temperature": "temperature",

    "humidity": "humidity",
    "hum": "humidity",

    "wafer": "wafer_id",
    "waferid": "wafer_id",
    "wafer_id": "wafer_id",

    "sensorvalue": "sensor_value",
    "sensor_value": "sensor_value",

    "ldr": "ldr_value",
    "ldrvalue": "ldr_value",
    "ldr_value": "ldr_value",

    "ambientlight": "ambient_light",
    "ambient_light": "ambient_light",

    # Soil timestamps (underscore version only)
    "timestampmsunderscore": "timestamp_ms",   # reserved internal tag
    "timestamp_ms": "timestamp_ms",

    # Temperature timestamp special (DO NOT MAP TO SOIL)
    "timestampmsparen": "timestamp(ms)",
}

EXPECTED_FEATURES = {
    "wafer": ["wafer_id"] + [f"sensor_{i}" for i in range(1, 30 + 1)],
    "gas": ["mq2_value", "temperature", "humidity", "hour", "dayofweek"],
    "soil": ["sensor_value", "rolling_mean", "rolling_std"],

    # Temperature model expects EXACT:
    "temperature": ["timestamp(ms)", "sensor_value"],

    "light": ["ldr_value", "voltage", "resistance", "ambient_light"],
}

def fuzzy_match(candidate, list_vals, cutoff=0.78):
    matches = get_close_matches(candidate, list_vals, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def build_normalized_map(cols):
    return {c: normalize_col(c) for c in cols}

def map_columns_with_aliases(df_cols, expected_cols, synonyms=SYNONYMS, fuzzy_cutoff=0.78):
    norm_map = build_normalized_map(df_cols)
    inv_expected = {normalize_col(c): c for c in expected_cols}

    rename_map = {}
    notes = []

    expected_norms = list(inv_expected.keys())
    syn_keys = list(synonyms.keys())

    for orig, norm in norm_map.items():

        # Exact synonym
        if norm in synonyms:
            canonical = synonyms[norm]
            rename_map[orig] = canonical
            notes.append(f"alias_exact:{orig}->{canonical}")
            continue

        # Exact expected
        if norm in inv_expected:
            canonical = inv_expected[norm]
            rename_map[orig] = canonical
            notes.append(f"match_expected:{orig}->{canonical}")
            continue

        # sensor1 → sensor_1
        m = re.match(r"sensor(\d+)$", norm)
        if m:
            canonical = f"sensor_{int(m.group(1))}"
            if canonical in expected_cols:
                rename_map[orig] = canonical
                notes.append(f"sensor_index:{orig}->{canonical}")
                continue

        # fuzzy expected
        fm = fuzzy_match(norm, expected_norms, fuzzy_cutoff)
        if fm:
            canonical = inv_expected[fm]
            rename_map[orig] = canonical
            notes.append(f"fuzzy_expected:{orig}->{canonical}")
            continue

        # fuzzy synonyms
        fm2 = fuzzy_match(norm, syn_keys, fuzzy_cutoff)
        if fm2:
            canonical = synonyms[fm2]
            rename_map[orig] = canonical
            notes.append(f"fuzzy_syn:{orig}->{canonical}")
            continue

        notes.append(f"no_map:{orig}")

    return rename_map, notes
