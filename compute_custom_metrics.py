# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import os
import string

import pandas as pd


def safe_as_csv(data, path):
    """
    This function saves a dictionary as a csv file.
    """
    with open(path, "w") as f:
        f.write("data_name,value\n")
        for key, value in data.items():
            f.write(f"{key},{value}\n")
    return


def load_csvs(folderpath):
    """
    This function loads a csv file as a pandas dataframe.
    """
    csv_files = [f for f in os.listdir(folderpath) if f.endswith(".csv")]
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(folderpath, file)
        if "custom_metrics" not in file:
            df = pd.read_csv(file_path).drop(index=range(3))
            df.columns = list(string.ascii_uppercase[: len(df.columns)])
            key = os.path.splitext(file)[0]
            dataframes[key] = df.reset_index(drop=True)
    return dataframes


def calculate_custom_metric(
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
    # iterate through all the result folders
    results_dir = "results"
    folders = os.listdir(results_dir)
    for folder in folders:
        folder_path = os.path.join(results_dir, folder, "csvs")
        results = load_csvs(folder_path)
        # compute the metrics needed
        metrics = calculate_custom_metric(
            results,
            "supply_energy",
            [["solid biomass", "links"], ["biogas", "links"]],
            [[[""], "Biomass supply", None]],
            "C",
            "D",
            filter_positive=True,
            remove_list=["biomass transport"],
        )
        safe_as_csv(metrics, os.path.join(folder_path, "custom_metrics.csv"))
