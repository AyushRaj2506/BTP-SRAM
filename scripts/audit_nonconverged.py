from pathlib import Path
import re
import ntpath
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "sram_fault_dataset.csv"
FLAGS = ROOT / "logs" / "convergence_flags.csv"
OUT_DIR = ROOT / "reports" / "audit"
OUT_FILE = OUT_DIR / "nonconverged_by_reason.csv"


def find_column(df, candidates, description):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )



    """
    Extract sample_id from a netlist/raw path.

    First try to find a substring beginning with C_.
    Then remove common file extensions.
    """
    if pd.isna(value):
        return None

    text = str(value)

    # Prefer a path component containing the sample ID.
    match = re.search(r"(C_[^\\/]+)", text)
    if not match:
        return None

    sample_id = match.group(1)

    # Remove extensions if present.
    sample_id = re.sub(
        r"\.(cir|sp|net|raw|log|txt|csv)$",
        "",
        sample_id,
        flags=re.IGNORECASE,
    )

    return sample_id
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
            return b[:-(len(op) + 1)]

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


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET)

    print("Loading convergence flags...")
    flags = pd.read_csv(FLAGS)

    # ------------------------------------------------------------
    # Identify important columns
    # ------------------------------------------------------------

    dataset_sample_col = find_column(
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

    # ------------------------------------------------------------
    # Build sample_id in convergence flags
    # ------------------------------------------------------------

    flags = flags.copy()

    flags["sample_id"] = flags[net_path_col].apply(
    sample_id_from_net_path
)

    flags["warning_reason"] = flags[reason_col].apply(normalize_reason)

    # ------------------------------------------------------------
    # Select non-converged dataset rows
    # ------------------------------------------------------------

    if "converged" not in df.columns:
        raise ValueError(
            "Dataset does not contain the required 'converged' column."
        )

    nonconv = df[df["converged"] == 0].copy()

    print()
    print(f"Total dataset rows: {len(df)}")
    print(f"Non-converged rows: {len(nonconv)}")

    # ------------------------------------------------------------
    # Join convergence warnings to dataset
    # ------------------------------------------------------------

    # Keep one row per sample/reason pair.
    flag_reasons = (
        flags[["sample_id", "warning_reason"]]
        .dropna(subset=["sample_id"])
        .drop_duplicates()
    )

    # Combine multiple warnings for the same sample.
    reason_map = (
        flag_reasons.groupby("sample_id")["warning_reason"]
        .apply(lambda x: "; ".join(sorted(set(x))))
        .rename("warning_reasons")
    )

    nonconv["sample_id"] = nonconv[dataset_sample_col].astype(str)

    audit = nonconv.merge(
        reason_map,
        left_on="sample_id",
        right_index=True,
        how="left",
    )

    audit["warning_reasons"] = audit["warning_reasons"].fillna(
        "no_matching_convergence_flag"
    )

    # ------------------------------------------------------------
    # Identify placeholder/fallback values
    # ------------------------------------------------------------

    audit["placeholder_iddq_norm_10"] = (
        pd.to_numeric(audit.get("iddq_norm"), errors="coerce") == 10.0
    )

    audit["placeholder_t_write_20000"] = (
        pd.to_numeric(audit.get("t_write_ps"), errors="coerce") == 20000.0
    )

    audit["placeholder_t_sense_20000"] = (
        pd.to_numeric(audit.get("t_sense_ps"), errors="coerce") == 20000.0
    )

    audit["has_placeholder"] = (
        audit["placeholder_iddq_norm_10"]
        | audit["placeholder_t_write_20000"]
        | audit["placeholder_t_sense_20000"]
    )

    # ------------------------------------------------------------
    # Save audit output
    # ------------------------------------------------------------

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        dataset_sample_col,
        "sample_id",
        "fault_class",
        "converged",
        "warning_reasons",
        "placeholder_iddq_norm_10",
        "placeholder_t_write_20000",
        "placeholder_t_sense_20000",
        "has_placeholder",
    ]

    # Add useful measurements if they exist.
    for col in [
        "iddq_norm",
        "t_write_ps",
        "t_sense_ps",
    ]:
        if col in audit.columns:
            output_columns.append(col)

    output_columns = [
        col for col in output_columns if col in audit.columns
    ]

    audit[output_columns].to_csv(OUT_FILE, index=False)

    # ------------------------------------------------------------
    # Print requested tables
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("NON-CONVERGED ROWS BY FAULT CLASS")
    print("=" * 70)

    class_counts = (
        audit["fault_class"]
        .value_counts()
        .sort_index()
        .rename_axis("fault_class")
        .reset_index(name="count")
    )

    print(class_counts.to_string(index=False))

    # Expand reasons so one sample contributes once to each reason.
    reason_rows = []

    for _, row in audit.iterrows():
        reasons = str(row["warning_reasons"]).split("; ")

        for reason in reasons:
            reason_rows.append(
                {
                    "fault_class": row["fault_class"],
                    "warning_reason": reason,
                    "sample_id": row["sample_id"],
                }
            )

    reason_df = pd.DataFrame(reason_rows).drop_duplicates()

    print()
    print("=" * 70)
    print("NON-CONVERGED ROWS BY WARNING REASON")
    print("=" * 70)

    reason_counts = (
        reason_df.groupby(["fault_class", "warning_reason"])
        .size()
        .reset_index(name="count")
        .sort_values(["fault_class", "warning_reason"])
    )

    print(reason_counts.to_string(index=False))

    print()
    print("=" * 70)
    print("PLACEHOLDER VALUES")
    print("=" * 70)

    placeholder_cols = [
        "placeholder_iddq_norm_10",
        "placeholder_t_write_20000",
        "placeholder_t_sense_20000",
    ]

    for col in placeholder_cols:
        print(f"{col}: {int(audit[col].sum())}")

    print(f"Any placeholder: {int(audit['has_placeholder'].sum())}")

    print()
    print("=" * 70)
    print("ROWS WITH PLACEHOLDER VALUES")
    print("=" * 70)

    placeholder_view = audit[
        audit["has_placeholder"]
    ][
        [
            "sample_id",
            "fault_class",
            "warning_reasons",
            "placeholder_iddq_norm_10",
            "placeholder_t_write_20000",
            "placeholder_t_sense_20000",
        ]
    ]

    if len(placeholder_view) == 0:
        print("None")
    else:
        print(placeholder_view.to_string(index=False))

    print()
    print(f"Saved audit file:")
    print(OUT_FILE)


if __name__ == "__main__":
    main()