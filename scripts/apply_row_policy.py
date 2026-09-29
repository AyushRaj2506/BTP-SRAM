from pathlib import Path
import re
import ntpath
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "sram_fault_dataset.csv"
FLAGS = ROOT / "logs" / "convergence_flags.csv"

OUT_DIR = ROOT / "reports" / "audit"
MAIN_OUT = OUT_DIR / "dataset_clean_main.csv"
STRICT_OUT = OUT_DIR / "dataset_clean_strict.csv"
EXCLUDED_OUT = OUT_DIR / "excluded_rows.csv"


# ------------------------------------------------------------
# Convergence-flag sample ID parser
# ------------------------------------------------------------

OPS = sorted(
    ["hold", "write_0", "write", "read_0", "read", "snm_hold", "snm_read"],
    key=len,
    reverse=True,
)


def sample_id_from_net_path(p):
    if pd.isna(p):
        return None

    b = ntpath.basename(str(p))

    if b.endswith(".net"):
        b = b[:-4]

    b = re.sub(r"^tb_", "", b)

    for op in OPS:
        if b.endswith("_" + op):
            return b[: -(len(op) + 1)]

    return None


def normalize_reason(value):
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()

    if "snm" in text and "nonmonotonic" in text:
        return "snm_nonmonotonic"

    if "missing" in text or "empty" in text or ".raw" in text:
        return "missing_or_empty_raw"

    return text


# ------------------------------------------------------------
# Corner extraction
# ------------------------------------------------------------

def extract_corner(sample_id):
    """
    Extract corner such as:
        M40C_0P9V
        0C_1P0V
        27C_1P1V
        75C_0P9V
        125C_1P1V
    """

    if pd.isna(sample_id):
        return None

    match = re.search(
        r"_(M40C|0C|27C|75C|125C)_(0P9V|1P0V|1P1V)_",
        str(sample_id),
    )

    if not match:
        return None

    return f"{match.group(1)}_{match.group(2)}"


def find_column(df, candidates, description):
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("Loading dataset...")
    df = pd.read_csv(DATASET)

    print("Loading convergence flags...")
    flags = pd.read_csv(FLAGS)

    sample_col = find_column(
        df,
        ["sample_id", "sample", "id"],
        "dataset sample ID",
    )

    net_path_col = find_column(
        flags,
        ["net_path", "path", "netlist_path", "file_path"],
        "net_path",
    )

    reason_col = find_column(
        flags,
        ["reason", "warning_reason", "warning", "message", "flag"],
        "warning/reason",
    )

    if "converged" not in df.columns:
        raise ValueError(
            "Dataset does not contain 'converged'."
        )

    # --------------------------------------------------------
    # Build convergence warning map
    # --------------------------------------------------------

    flags["sample_id"] = flags[net_path_col].apply(
        sample_id_from_net_path
    )

    flags["warning_reason"] = flags[reason_col].apply(
        normalize_reason
    )

    flag_reasons = (
        flags[
            ["sample_id", "warning_reason"]
        ]
        .dropna(subset=["sample_id"])
        .drop_duplicates()
    )

    reason_map = (
        flag_reasons
        .groupby("sample_id")["warning_reason"]
        .apply(lambda x: "+".join(sorted(set(x))))
        .rename("warning_reasons")
    )

    # --------------------------------------------------------
    # Add warning information to dataset
    # --------------------------------------------------------

    work = df.copy()
    work["sample_id"] = work[sample_col].astype(str)

    work = work.merge(
        reason_map,
        left_on="sample_id",
        right_index=True,
        how="left",
    )

    work["warning_reasons"] = work["warning_reasons"].fillna("")

    # --------------------------------------------------------
    # Determine exclusions
    # --------------------------------------------------------

    # Main policy:
    # EXCLUDE every row having a missing/empty raw waveform warning.
    #
    # Keep snm_nonmonotonic-only rows.
    #
    # Do NOT exclude 20000-ps bridge sentinel values merely because
    # they equal 20000.

    has_missing_raw = work["warning_reasons"].str.contains(
        "missing_or_empty_raw",
        na=False,
    )

    main_exclude = has_missing_raw

    work["main_exclusion"] = main_exclude

    work["exclusion_reason"] = ""

    work.loc[
        main_exclude,
        "exclusion_reason"
    ] = "missing_or_empty_raw"

    # --------------------------------------------------------
    # Create main cleaned dataset
    # --------------------------------------------------------

    clean_main = work.loc[
        ~main_exclude
    ].copy()

    # --------------------------------------------------------
    # Strict sensitivity dataset
    # --------------------------------------------------------

    clean_strict = work.loc[
        work["converged"] == 1
    ].copy()

    # --------------------------------------------------------
    # Excluded rows
    # --------------------------------------------------------

    excluded = work.loc[
        main_exclude,
        [
            "sample_id",
            "fault_class",
            "converged",
            "warning_reasons",
            "exclusion_reason",
        ],
    ].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_main.to_csv(
        MAIN_OUT,
        index=False,
    )

    clean_strict.to_csv(
        STRICT_OUT,
        index=False,
    )

    excluded.to_csv(
        EXCLUDED_OUT,
        index=False,
    )

    # --------------------------------------------------------
    # Add corners
    # --------------------------------------------------------

    work["corner"] = work["sample_id"].apply(
        extract_corner
    )

    clean_main["corner"] = clean_main["sample_id"].apply(
        extract_corner
    )

    clean_strict["corner"] = clean_strict["sample_id"].apply(
        extract_corner
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROW POLICY SUMMARY")
    print("=" * 70)

    print(f"Original rows:       {len(work)}")
    print(f"Main rows remaining: {len(clean_main)}")
    print(f"Main rows removed:   {len(excluded)}")
    print(f"Strict rows:         {len(clean_strict)}")

    print()
    print("=" * 70)
    print("MAIN DATASET — ROWS REMOVED BY CLASS")
    print("=" * 70)

    removed_class = (
        excluded["fault_class"]
        .value_counts()
        .sort_index()
        .rename_axis("fault_class")
        .reset_index(name="removed")
    )

    print(removed_class.to_string(index=False))

    print()
    print("=" * 70)
    print("MAIN DATASET — ROWS REMAINING BY CLASS")
    print("=" * 70)

    remaining_class = (
        clean_main["fault_class"]
        .value_counts()
        .sort_index()
        .rename_axis("fault_class")
        .reset_index(name="remaining")
    )

    print(remaining_class.to_string(index=False))

    print()
    print("=" * 70)
    print("STRICT DATASET — ROWS REMAINING BY CLASS")
    print("=" * 70)

    strict_class = (
        clean_strict["fault_class"]
        .value_counts()
        .sort_index()
        .rename_axis("fault_class")
        .reset_index(name="remaining")
    )

    print(strict_class.to_string(index=False))

    print()
    print("=" * 70)
    print("MAIN DATASET — ROWS REMAINING BY CORNER")
    print("=" * 70)

    corner_counts = (
        clean_main["corner"]
        .value_counts()
        .sort_index()
        .rename_axis("corner")
        .reset_index(name="remaining")
    )

    print(corner_counts.to_string(index=False))

    print()
    print("=" * 70)
    print("EXCLUSION REASONS")
    print("=" * 70)

    print(
        excluded["exclusion_reason"]
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="count")
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("EXPECTED COUNT CHECK")
    print("=" * 70)

    expected_main = {
        0: 13,
        1: 171,
        2: 73,
        3: 84,
        4: 87,
    }

    actual_main = (
        clean_main["fault_class"]
        .value_counts()
        .to_dict()
    )

    expected_total = 428
    actual_total = len(clean_main)

    print(f"Expected main total: {expected_total}")
    print(f"Actual main total:   {actual_total}")

    for cls in sorted(expected_main):
        actual = actual_main.get(cls, 0)
        expected = expected_main[cls]

        status = "PASS" if actual == expected else "MISMATCH"

        print(
            f"Class {cls}: expected {expected}, "
            f"actual {actual} -> {status}"
        )

    print()
    print(f"Expected strict total: 302")
    print(f"Actual strict total:   {len(clean_strict)}")

    print()
    print("Saved:")
    print(MAIN_OUT)
    print(STRICT_OUT)
    print(EXCLUDED_OUT)


if __name__ == "__main__":
    main()