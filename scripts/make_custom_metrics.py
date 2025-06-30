import os
import string

import pandas as pd


def safe_as_csv(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("data_name,value\n")
        for key, value in data.items():
            f.write(f"{key},{value}\n")


def load_csvs(folderpath):
    """
    This function loads a csv file as a pandas dataframe.
    """
    csv_files = [f for f in os.listdir(folderpath) if f.endswith(".csv")]
    dataframes = {}
    for file in csv_files:
        try:
            file_path = os.path.join(folderpath, file)
            if "custom_metrics" in file or "cumulative_costs" in file:
                df = pd.read_csv(file_path)
            elif "metrics" in file:
                df = pd.read_csv(file_path).drop(index=range(2))
            else:
                df = pd.read_csv(file_path).drop(index=range(3))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise e
        df.columns = list(string.ascii_uppercase[: len(df.columns)])
        key = os.path.splitext(file)[0]
        dataframes[key] = df.reset_index(drop=True)
    return dataframes


def calculate_custom_metric( #note that this works with the old logic of contains rather than equals of strings
    dataframes,
    key,
    fields_list,
    merge_fields,
    data_name_column,
    value_column,
    multiplier=1,
    filter_positive=True,
    remove_list=[],
):
    # Read the data
    data_df = dataframes[key]
    result_data = {}
    # Build condition for filtering
    condition = pd.Series([False] * len(data_df))
    for fields in fields_list:
        field_condition = pd.Series([True] * len(data_df))
        for i, field in enumerate(fields):
            field_condition &= data_df.iloc[:, i].str.contains(
                field, case=False, na=False
            )
        condition |= field_condition

    data_df["condition"] = condition.fillna(False)
    # Filter data based on condition
    data = data_df[data_df["condition"]].copy()
    data.drop(columns=["condition"], inplace=True)

    # Apply remove_list filter
    if remove_list:
        data = data[
            ~data[data_name_column].str.contains("|".join(remove_list), na=False)
        ]

    # Ensure value_column is numeric
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")

    # Apply filter_positive
    if filter_positive is None:
        data = data
    elif filter_positive:
        data = data[data[value_column] > 0]
    else:
        data = data[data[value_column] < 0]

    # Merge the values using the merge_fields
    for merge_conditions, new_name, is_cc in merge_fields:
        for merge_condition in merge_conditions:
            added_data = data[
                data[data_name_column].str.contains(merge_condition, na=False)
            ]
            if is_cc:  # CC case sensitive must be in the data_name
                added_data = added_data[
                    added_data[data_name_column].str.contains("CC", case=True, na=False)
                ]
            elif is_cc == False:  # CC case sensitive must not be in the data_name
                added_data = added_data[
                    ~added_data[data_name_column].str.contains(
                        "CC", case=True, na=False
                    )
                ]
            else:
                added_data = added_data
            result_data[new_name] = added_data[value_column].sum() * multiplier
    return result_data


if __name__ == "__main__":
    folder   = snakemake.params.folder
    out_file = snakemake.output[0]

    dfs = load_csvs(folder)

    metrics = calculate_custom_metric(
        dfs,
        "energy_balance",
        [["Link", "", "solid biomass"], ["Link", "", "biogas"]],
        [[[""], "Biomass supply", None]],
        "B",
        "D",
        filter_positive=True,
        remove_list=["solid biomass transport"],
    )
    metrics_renwable = calculate_custom_metric(
        dfs,
        "energy_balance",
        [["Generator", "solar rooftop", "low voltage"],["Generator", "onwind", "AC"],["Generator", "offwind-ac", "AC"],["Generator", "offwind-dc", "AC"],["Generator", "offwind-float", "AC"],["Generator", "solar", "AC"],["Generator", "solar-hsat", "AC"]],
        [
        [["solar"], "Solar electricity", None],
        [["wind"], "Wind electricity", None]
        ],
        "B",
        "D",
        filter_positive=True,
    )
    metrics.update(metrics_renwable)
    safe_as_csv(metrics, out_file)