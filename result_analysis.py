# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import os
import string

import pandas as pd


def load_results(results_dir, folders="all"):
    """
    Load results from CSV files in the specified directory.

    Parameters
    ----------
    results_dir (str): Path to the directory containing the results.

    Returns
    -------
    dict: Dictionary containing the loaded dataframes.
    """
    results = {}
    if folders == "all":
        folders = os.listdir(results_dir)
    else:
        folders = [f for f in folders if f in os.listdir(results_dir)]
    for folder in folders:
        folder_path = os.path.join(results_dir, folder, "csvs")
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        dataframes = {}
        print(f"Loading data from {folder}...")
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path).drop(index=range(3))
            df.columns = list(string.ascii_uppercase[: len(df.columns)])
            key = os.path.splitext(file)[0]
            dataframes[key] = df
        results[folder] = dataframes
    return results


def get_data(
    results,
    folders,
    dataframe,
    fields,
    value_column,
    data_name,
    year,
    multiplier=1,
    value="all",
    round_digits=1,
):
    """
    Process the results and calculate the division of supply energy by costs.

    Parameters
    ----------
    results (dict): Dictionary containing the dataframes.
    folders (list): List of folders to process.
    dataframe (str): The key to access the specific dataframe in the dictionary.
    fields (list): List of fields to filter the dataframe.
    value_column (str): The column in which the value is located.
    data_name (str): The name of the data to retrieve.
    year (str): The year of the data.
    multiplier (float): The multiplier to apply to the values in the value column.
    value (int): The index of the value to retrieve from the list of values.

    Returns
    -------
    dict: Dictionary containing the processed data.
    """
    result_data = {}
    if folders == "all":
        folders = results.keys()

    for folder, dataframes in results.items():
        if folder in folders:
            data_df = dataframes[dataframe]
            condition = True
            for i, field in enumerate(fields):
                condition &= data_df.iloc[:, i].str.contains(
                    field, case=False, na=False
                )
            data = data_df[condition].reset_index(drop=True)
            if len(data_name) == 1:
                text = (
                    data[data_name].values
                    if data_name in data.columns
                    else f"Warning: '{data_name}' does not match any column name. Use capital letter like A,B ..."
                )
            else:
                text = data_name
            # Convert the value column to numeric, forcing errors to NaN
            data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
            values = (data[value_column] * multiplier).values.tolist()
            for i in range(len(values)):
                values[i] = round(values[i], round_digits)
            if values == []:
                values = 0
            if value != "all":
                values = values[value]
            result_data[f"{folder}_{year}_{data_name}"] = {
                "folder": folder,
                "year": year,
                "data_name": text,
                "values": values,
            }
    return result_data


def print_data(data):
    """
    Print the data in a readable format.

    Parameters
    ----------
    data (dict): Dictionary containing the processed data.
    """
    for key, content in data.items():
        print(f"Folder: {content['folder']}")
        print(f"Year: {content['year']}")
        print(f"Data Name: {content['data_name']}")
        print(f"Values: {content['values']}")
        print()


def calculate_difference(
    results,
    scenario1,
    scenario2,
    dataframe,
    data_name_columns,
    value_column,
    year,
    merge_list,
    remove_list,
    multiplier=1,
    round_digits=1,
):
    """
    Calculate the difference between the value column of two scenarios, output the original values and the difference, and sort the output by the difference.

    Parameters
    ----------
    results (dict): Dictionary containing the dataframes.
    scenario1 (str): The key to access the first scenario in the dictionary.
    scenario2 (str): The key to access the second scenario in the dictionary.
    dataframe (str): The key to access the specific dataframe in the dictionary.
    data_name_columns (list): List of columns in which the data name is located.
    value_column (str): The column in which the value is located.
    year (str): The year to filter the data.
    merge_list (list): List of lists containing merge conditions, new name, and case sensitivity.
    remove_list (list): List of strings to remove from the data.
    multiplier (int, optional): Multiplier to apply to the values. Default is 1.
    round_digits (int, optional): Number of digits to round the values. Default is 1.

    Returns
    -------
    dict: Dictionary containing the processed data.
    """
    # Get the dataframes
    df1 = results[scenario1][dataframe]
    df2 = results[scenario2][dataframe]

    merged_df1 = pd.DataFrame()
    merged_df2 = pd.DataFrame()
    for df, merged_df, scenario in [
        (df1, merged_df1, scenario1),
        (df2, merged_df2, scenario2),
    ]:
        # remove the data that should be removed, all fields must meet the corresponding remove condition (a list of fields)
        for remove in remove_list:
            condition = pd.Series([True] * len(df))
            for field in remove:
                condition &= ~df.iloc[:, 0].str.contains(field, case=False, na=False)
            df = df[condition]

        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        for merge_conditions, new_name, is_cc in merge_list:
            merged_data = pd.DataFrame()
            df = df.reset_index(drop=True)
            condition = pd.Series([False] * len(df))

            for merge_condition in merge_conditions:
                field_condition = pd.Series([True] * len(df), index=df.index)
                for i, field in enumerate(merge_condition):
                    field_condition &= df.iloc[:, i].str.contains(
                        field, case=True, na=False
                    )
                condition |= field_condition
                condition = condition.reindex(df.index, fill_value=False)
                added_data = df[condition]
                # if is_cc is True, CC case sensitive must be in the data_name
                if is_cc:
                    added_data = added_data[
                        added_data[data_name_columns[-1]].str.contains(
                            "CC", case=True, na=False
                        )
                    ]
                # if is_cc is False, CC case sensitive must not be in the data_name
                elif is_cc == False:
                    added_data = added_data[
                        ~added_data[data_name_columns[-1]].str.contains(
                            "CC", case=True, na=False
                        )
                    ]
                else:
                    added_data = added_data
                merged_data = pd.concat([merged_data, added_data])
                # remove the added data from the original data
                if not merged_data.empty:
                    if scenario == scenario1:
                        df1 = df1[~df1.index.isin(merged_data.index)]
                        df1 = df1.reset_index(drop=True)
                    else:
                        df2 = df2[~df2.index.isin(merged_data.index)]
                        df2 = df2.reset_index(drop=True)
                    df = df[~df.index.isin(merged_data.index)]
                    df = df.reset_index(drop=True)
            # create a new row using the new_name and the sum of the values of merged rows
            if not merged_data.empty:
                new_row = {
                    "folder": scenario1,
                    "year": year,
                    "data_name": new_name,
                    "values": merged_data[value_column].sum() * multiplier,
                }
                merged_df = pd.concat([merged_df, pd.DataFrame([new_row])])
        if scenario == scenario1:
            merged_df1 = merged_df
        else:
            merged_df2 = merged_df

    # Add remaining data, data_name will be a combination of the data_name_columns
    for df, scenario, merged_df in [
        (df1, scenario1, merged_df1),
        (df2, scenario2, merged_df2),
    ]:
        for _, row in df.iterrows():
            key = "_".join([row[column] for column in data_name_columns])
            key = key.replace(" ", "_")
            new_row = {
                "year": year,
                "data_name": key,
                "values": row[value_column] * multiplier,
            }
            merged_df = pd.concat([merged_df, pd.DataFrame([new_row])])
        if scenario == scenario1:
            merged_df1 = merged_df
        else:
            merged_df2 = merged_df

    # Merge the dataframes and calculate the difference
    merged_df = merged_df1.merge(
        merged_df2,
        on=["data_name", "year"],
        how="outer",
        suffixes=(f"_{scenario1}", f"_{scenario2}"),
    )
    # add a column for the difference
    merged_df["difference"] = (
        merged_df[f"values_{scenario2}"] - merged_df[f"values_{scenario1}"]
    ) * multiplier
    # round the values
    merged_df = merged_df.round(
        {
            f"values_{scenario1}": round_digits,
            f"values_{scenario2}": round_digits,
            "difference": round_digits,
        }
    )
    # sort by the difference
    merged_df = merged_df.sort_values(by="difference", ascending=False)
    dict_data = {}
    for _, row in merged_df.iterrows():
        key = f"{scenario1}_{scenario2}_{year}_{row['data_name']}"
        dict_data[key] = {
            "year": year,
            "data_name": row["data_name"],
            f"values_{scenario1}": row[f"values_{scenario1}"],
            f"values_{scenario2}": row[f"values_{scenario2}"],
            "difference": row["difference"],
        }
    return dict_data


def export_results(
    data,
    filename,
    include_share=False,
    include_difference=False,
    scenario1="default",
    scenario2="biomass_emissions",
    add_costs=False,
):
    """
    Export the results to a CSV file.

    Parameters
    ----------
    data (dict): Dictionary containing the processed data.
    filename (str): Name of the CSV file to save.
    """
    export_dir = "export"
    os.makedirs(export_dir, exist_ok=True)
    rows = []
    for key, content in data.items():
        # Flatten the values if they are lists with a single entry
        if include_difference:
            values1 = content[f"values_{scenario1}"]
            if isinstance(values1, list) and len(values1) == 1:
                values1 = values1[0]
            values2 = content[f"values_{scenario2}"]
            if isinstance(values2, list) and len(values2) == 1:
                values2 = values2[0]
            rows.append(
                {
                    "Year": content["year"],
                    "Data Name": content["data_name"],
                    f"Values_{scenario1}": values1,
                    f"Values_{scenario2}": values2,
                    "Difference": content["difference"],
                }
            )
        else:
            values = content["values"]
            if isinstance(values, list) and len(values) == 1:
                values = values[0]
            rows.append(
                {
                    "Folder": content["folder"],
                    "Year": content["year"],
                    "Data Name": content["data_name"],
                    "Values": values,
                }
            )
            if include_share:
                rows[-1]["Share"] = content["share"]
        if add_costs:
            if content["costs"] is not None and isinstance(
                content["costs"], (int, float)
            ):
                rows[-1]["Costs"] = round(content["costs"], 2)
            else:
                rows[-1]["Costs"] = content["costs"]
    df = pd.DataFrame(rows)
    file_path = os.path.join(export_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Results exported to {file_path}")


def aggregate_and_calculate_share(
    results,
    folders,
    dataframe,
    fields_list,
    merge_fields,
    value_column,
    data_name_column,
    year,
    multiplier=1,
    filter_positive=True,
    remove_list=[],
    calculate_share=True,
):
    """
    Aggregate data and optionally calculate the share of each unique data name of the overall sum.

    Parameters
    ----------
    results (dict): Dictionary containing the dataframes.
    folders (list): List of folders to process.
    dataframe (str): The key to access the specific dataframe in the dictionary.
    fields_list (list): List of lists of fields to filter the dataframe.
    merge_fields (list): List of lists containing merge conditions, new name, and case sensitivity.
    value_column (str): The column in which the value is located.
    data_name_column (str): The column in which the data name is located.
    year (str): The year to filter the data.
    multiplier (int, optional): Multiplier to apply to the values. Default is 1.
    filter_positive (bool, optional): If True, only take rows with positive values. Default is True.
    remove_list (list, optional): List of strings to remove from the data. Default is [].
    calculate_share (bool, optional): If True, calculate the share of each unique data name of the overall sum. Default is False.

    Returns
    -------
    dict: Dictionary containing the aggregated data and optionally their share of the overall sum.
    """
    result_data = {}
    if folders == "all":
        folders = results.keys()

    for folder, dataframes in results.copy().items():
        if folder in folders:
            data_df = dataframes[dataframe].reset_index(drop=True)

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
                    ~data[data_name_column].str.contains(
                        "|".join(remove_list), na=False
                    )
                ]

            # Ensure value_column is numeric
            data[value_column] = pd.to_numeric(data[value_column], errors="coerce")

            # Apply filter_positive
            if filter_positive:
                data = data[data[value_column] > 0]
            else:
                data = data[data[value_column] < 0]

            # Calculate total sum for share calculation
            total_sum = data[value_column].sum()

            # Merge the values using the merge_fields
            for merge_conditions, new_name, is_cc in merge_fields:
                merged_data = pd.DataFrame()
                for merge_condition in merge_conditions:
                    added_data = data[
                        data[data_name_column].str.contains(merge_condition, na=False)
                    ]
                    if is_cc:  # CC case sensitive must be in the data_name
                        added_data = added_data[
                            added_data[data_name_column].str.contains(
                                "CC", case=True, na=False
                            )
                        ]
                    elif (
                        is_cc == False
                    ):  # CC case sensitive must not be in the data_name
                        added_data = added_data[
                            ~added_data[data_name_column].str.contains(
                                "CC", case=True, na=False
                            )
                        ]
                    else:
                        added_data = added_data
                    merged_data = pd.concat([merged_data, added_data])

                    # Remove the added data from the original data
                    if not added_data.empty:
                        data = data[~data.index.isin(added_data.index)]

                # Create a new row using the new_name and the sum of the values of merged rows
                if not merged_data.empty:
                    new_row = {
                        "folder": folder,
                        "year": year,
                        "data_name": new_name,
                        "values": round(
                            merged_data[value_column].sum() * multiplier, 1
                        ),
                    }
                    if calculate_share:
                        share = merged_data[value_column].sum() / total_sum
                        new_row["share"] = round(share, 4)
                    result_data[f"{folder}_{year}_{new_name}"] = new_row

            # Add remaining data and calculate shares if calculate_share is True
            if not data.empty:
                for _, row in data.iterrows():
                    key = f"{folder}_{year}_{row[data_name_column]}"
                    new_row = {
                        "folder": folder,
                        "year": year,
                        "data_name": row[data_name_column],
                        "values": round(row[value_column] * multiplier, 1),
                    }
                    if calculate_share:
                        share = row[value_column] / total_sum
                        new_row["share"] = round(share, 4)
                    result_data[key] = new_row

    return result_data


def add_costs(data):
    costs = {  # Euro/MWh_LHV
        "agricultural waste": 12.8786,
        "fuelwood residues": 15.3932,
        "fuelwoodRW": 12.6498,
        "manure": 22.1119,
        "residues from landscape care": 10.5085,
        "secondary forestry residues": 8.1876,
        "coal": 9.5542,
        "fuelwood": 14.5224,
        "gas": 24.568,
        "oil primary": 52.9111,
        "woody crops": 44.4,
        "grasses": 18.9983,
        "sludge": 22.0995,
        "solid biomass import": 54,
        "sawdust": 6.4791,
        "C&P_RW": 25.4661,
    }
    if add_costs:
        for key, content in data.items():
            if content["data_name"] in costs:
                content["costs"] = costs[content["data_name"]]
            else:
                content["costs"] = None
    return data


def add_co2_price(data, co2_prices, column="values"):
    """
    Add the co2 price to the data

    Parameters
    ----------
    data (dict): Dictionary containing the processed data.
    co2_prices (dict): Dictionary of dictionaries containing the co2 prices for each scenario.
    """
    for key, content in data.items():
        if content["data_name"] in co2_prices[content["folder"]]:
            if content[column] is not None:
                content[column] += co2_prices[content["folder"]][content["data_name"]]
    return data


def calculate_carbon_removal(
    dict,
    carbon_intensity,
    carbon_removal,
    add_to_total=True,
    capture_rate=0.9,
    existing_dict=None,
    scenarios=["default", "biomass_emissions"],
    is_removed=False,
    gas_shares=None,
):
    """
    Returns dict with carbon removed and total carbon content
    """
    results = {}
    for folder in scenarios:
        if gas_shares is not None:
            capture_rate = capture_rate * (1 - gas_shares[folder]["share"])
        if existing_dict is None:
            carbon_removed = 0
            total_carbon = 0
        else:
            carbon_removed = existing_dict[folder]["carbon_removed"]
            total_carbon = existing_dict[folder]["total_carbon"]
        for key, content in dict.items():
            if content["folder"] == folder:
                if content["values"] is not None:
                    if is_removed:
                        carbon_removed += (
                            content["values"] * carbon_removal * capture_rate
                        )
                    if add_to_total:
                        total_carbon += content["values"] * carbon_intensity
            results[folder] = {
                "carbon_removed": carbon_removed,
                "total_carbon": total_carbon,
            }
    return results


def calculate_removal_share(dict):
    results = {}
    for key, content in dict.items():
        carbon_removed = content["carbon_removed"]
        total_carbon = content["total_carbon"]
        share_removed = carbon_removed / total_carbon
        results[key] = {
            "carbon_removed": carbon_removed,
            "total_carbon": total_carbon,
            "share_removed": share_removed,
        }
    return results


def export_carbon_removal(data, filename):
    export_dir = "export"
    os.makedirs(export_dir, exist_ok=True)
    # Simply make the dict into a csv
    rows = []
    for key, content in data.items():
        rows.append(
            {
                "Folder": key,
                "Carbon Removed": content["carbon_removed"],
                "Total Carbon": content["total_carbon"],
                "Share Removed": content["share_removed"],
            }
        )
    df = pd.DataFrame(rows)
    file_path = os.path.join(export_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Results exported to {file_path}")


def calc_gas_share(
    data,
    scenarios,
):
    results = {}
    for folder in scenarios:
        gas_values = 0
        total = 0
        for key, content in data.items():
            if content["folder"] == folder:
                if content["data_name"] == "gas":
                    gas_values += content["values"]
                total += content["values"]
        share_gas = gas_values / total
        results[folder] = {"gas": gas_values, "total": total, "share": share_gas}
    return results


def main():
    results_dir = "results"
    scenarios = ["default", "biomass_emissions"]
    results = load_results(results_dir, scenarios)
    # scenarios = "all"

    # years = ["2030", "2050"]

    # biomass_types = [
    #     "agricultural waste",
    #     "fuelwood residues",
    #     "secondary forestry residues",
    #     "sawdust",
    #     "residues from landscape care",
    #     "grasses",
    #     "woody crops",
    #     "fuelwoodRW",
    #     "manure",
    #     "sludge",
    # ]

    remove_list = ["agriculture machinery oil1"]
    oil_production_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["oil", "links"]],
        [],
        "D",
        "C",
        "2050",
        filter_positive=True,
        remove_list=remove_list,
        calculate_share=True,
    )
    export_results(oil_production_2050, "oil_production_2050.csv", include_share=True)

    electricity_generation_share_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [
            ["AC", "generators"],
            ["low voltage", "generators", "solar rooftop"],
            ["AC", "links", "CHP"],
            ["AC", "links", "OCGT"],
            ["AC", "storage_unit", "hydro"],
        ],
        [
            [["wind"], "wind", False],
            [["solar"], "solar", False],
            [["hydro", "ror"], "hydro", False],
            [["biomass CHP"], "biomass CHP", None],
            [["waste CHP"], "waste CHP", None],
            [["urban central CHP"], "gas CHP", None],
        ],  # merge_fields
        "D",
        "C",
        "2050",
        filter_positive=True,
        calculate_share=True,
    )
    export_results(
        electricity_generation_share_2050,
        "electricity_generation_share_2050.csv",
        include_share=True,
    )

    beccs = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [
            ["solid biomass", "links", "CC"],
            ["solid biomass", "links", "electrobiofuel"],
        ],
        [],
        "D",
        "C",
        "2050",
        multiplier=-1,
        filter_positive=False,
        calculate_share=True,
    )
    export_results(beccs, "beccs.csv")

    fields_list = [
        ["highT industry", "links"],
        ["mediumT industry", "links"],
        ["lowT industry", "links"],
    ]  # all industries
    merge_fields = [
        [["biomass"], "biomass", False],
        [["biomass"], "biomass CC", True],
        [["gas", "methane"], "gas", False],
        [["gas", "methane"], "gas CC", True],
        [["hydrogen"], "hydrogen", False],
        [["heat pump", "electricity"], "electricity/heat pump", False],
    ]
    industrial_energy_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        fields_list,
        merge_fields,
        "D",
        "C",
        "2050",
        filter_positive=True,
    )
    export_results(
        industrial_energy_2050, "industrial_energy_2050.csv", include_share=True
    )

    fields_list = [
        ["rural heat", "links"],
        ["urban central heat", "links"],
        ["urban decentral heat", "links"],
    ]  # all heating
    merge_fields = [
        [["biomass"], "biomass", False],
        [["biomass"], "biomass CC", True],
        [["waste"], "waste", False],
        [["waste"], "waste CC", True],
        [["gas", "CHP"], "gas", False],
        [["gas", "CHP"], "gas CC", True],
        [["H2"], "hydrogen", False],
        [["heat pump", "resistive"], "electricity/heat pump", False],
        [["water tanks discharger"], "water tank discharger", False],
    ]
    heating_energy_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        fields_list,
        merge_fields,
        "D",
        "C",
        "2050",
        filter_positive=True,
    )
    export_results(heating_energy_2050, "heating_energy_2050.csv", include_share=True)

    # primary energy
    fields_list = [
        ["bioliquids", "generators"],
        ["AC", "generators"],
        ["AC", "links", "waste"],
        ["AC", "storage_units", "hydro"],
        ["biogas", "generators"],
        ["biogas", "links"],
        ["coal", "generators"],
        ["gas", "generators"],
        ["low voltage", "generators", "solar rooftop"],
        ["waste", "generators"],
        ["oil primary", "generators"],
        ["solid biomass", "links"],
        ["solid biomass", "generators"],
    ]
    merge_fields = [
        [
            [
                "agricultural waste",
                "fuelwood residues",
                "secondary forestry residues",
                "sawdust",
                "residues from landscape care",
                "grasses",
                "woody crops",
                "fuelwoodRW",
                "biomass",
                "C&P_RW",
            ],
            "solid biomass",
            False,
        ],
        [["manure", "sludge"], "biogas", False],
        [["wind"], "wind", False],
        [["solar"], "solar", False],
        [["hydro", "ror"], "hydro", False],
    ]
    remove_list = ["biomass transport", "waste CHP"]
    primary_energy_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        fields_list,
        merge_fields,
        "D",
        "C",
        "2050",
        filter_positive=True,
        remove_list=remove_list,
    )
    export_results(primary_energy_2050, "primary_energy_2050.csv", include_share=True)

    merge_fields = [
        [["biogas to gas"], "biogas upgrading without CC", False],
        [["biogas to gas"], "biogas upgrading with CC", True],
        [["CC", "electrobiofuels"], "biomass use with CC", None],
        [[""], "biomass use without CC", False],
    ]
    biomass_use_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["solid biomass", "links", ""]],
        merge_fields,
        "D",
        "C",
        "2050",
        filter_positive=False,
        multiplier=-1,
        calculate_share=True,
        remove_list=["transport", "import"],
    )
    export_results(biomass_use_2050, "biomass_use_2050.csv", include_share=True)

    merge_fields = [
        [["SNG"], "Conversion to SNG", None],
        [["to liquid", "electrobiofuels"], "Conversion to liquid fuels", None],
        [["industry", "boiler"], "Use for heat", None],
        [["CHP"], "Use in CHP", None],
        [["hydrogen"], "Use for hydrogen production", None],
    ]
    biomass_use_by_sector_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["solid biomass", "links", ""]],
        merge_fields,
        "D",
        "C",
        "2050",
        filter_positive=False,
        multiplier=-1,
        calculate_share=True,
        remove_list=["transport", "import"],
    )
    export_results(
        biomass_use_by_sector_2050, "biomass_use_by_sector_2050.csv", include_share=True
    )

    # costs_2030 = get_data(
    #     results,
    #     scenarios,
    #     "metrics",
    #     ["total costs"],
    #     "B",
    #     "Total costs (Billion €)",
    #     "2030",
    #     1e-9,
    # )
    costs_2050 = get_data(
        results,
        scenarios,
        "metrics",
        ["total costs"],
        "B",
        "Total costs (Billion €)",
        "2050",
        1e-9,
    )
    # combined_costs = {**costs_2030, **costs_2050}
    export_results(costs_2050, "costs2050.csv")

    # Get the supply data for all biomass types
    biomass_supply_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["solid biomass", "links"], ["biogas", "links"]],
        [],
        "D",
        "C",
        "2050",
        filter_positive=True,
        remove_list=["biomass transport"],
    )
    export_results(biomass_supply_2050, "biomass_supply.csv")

    fossil_fuel_supply_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["oil primary", "generators"], ["gas", "generators"], ["coal", "generators"]],
        [],
        "D",
        "C",
        "2050",
        filter_positive=True,
    )
    export_results(fossil_fuel_supply_2050, "fossil_fuel_supply.csv")

    merge_fields = [
        [[["", "", "wind"]], "wind", None],
        [[["", "", "waste CHP"]], "waste CHP CC", True],
        [[["", "", "biomass import"]], "biomass import", None],
        [[["", "", "heat pump"]], "heat pumps", None],
        [[["", "", "electrobiofuels"]], "electrobiofuels", None],
        [[["", "", "biomass to liquid"]], "biomass to liquid", None],
        [[["", "", "solar"]], "solar", None],
        [[["", "", "H2"]], "hydrogen", None],
        [[["", "", "gas boiler"]], "(bio)gas boiler", None],
        [[["", "", "oil primary"]], "oil primary", None],
        [[["", "", "biomass boiler"]], "biomass boiler", None],
        [[["", "", "biomass CHP"]], "biomass CHP", None],
        [[["", "", "nuclear"]], "nuclear", None],
        [[["generators", "", "gas"]], "natural gas", None],
        [[["", "", "water pits"]], "water pits (storage)", None],
        [[["", "", "biogas"]], "biogas CC", True],
        [[["", "", "biogas"]], "biogas without CC", False],
        [
            [
                ["", "", "agricultural waste"],
                ["", "", "fuelwood residues"],
                ["", "", "secondary forestry residues"],
                ["", "", "sawdust"],
                ["", "", "residues from landscape care"],
                ["", "", "grasses"],
                ["", "", "woody crops"],
                ["", "", "fuelwoodRW"],
                ["", "", "manure"],
                ["", "", "sludge"],
                ["", "", "C&P_RW"],
            ],
            "biomass extraction",
            None,
        ],
        [
            [
                ["", "", "DC"],
                ["lines", "", "AC"],
                ["", "", "electricity distribution grid"],
            ],
            "transmission",
            None,
        ],
    ]
    # Calculate the difference between two scenarios in costs
    cost_difference = calculate_difference(
        results,
        "default",
        "biomass_emissions",
        "costs",
        ["A", "B", "C"],
        "D",
        "2050",
        merge_fields,
        remove_list=[],
        round_digits=0,
    )
    export_results(cost_difference, "cost_difference.csv", include_difference=True)

    shadow_price_2050 = get_data(
        results,
        scenarios,
        "metrics",
        ["co2_shadow"],
        "B",
        "CO2 shadow price",
        "2050",
        multiplier=-1,
    )
    export_results(shadow_price_2050, "shadow_price_2050.csv")

    hydrogen_production_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["H2", "links"]],
        [],
        "D",
        "C",
        "2050",
        filter_positive=True,
        remove_list=["pipeline"],
    )
    export_results(hydrogen_production_2050, "hydrogen_production_2050.csv")

    heat_pumps_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["low voltage", "links", "heat pump"]],
        [],
        "D",
        "C",
        "2050",
        filter_positive=False,
        multiplier=-1,
    )
    export_results(heat_pumps_2050, "heat_pumps_2050.csv")

    gas_use_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["gas", "links"]],
        [
            [["biogas"], "biogas upgrading with CC", True],
            [["biogas"], "biogas upgrading without CC", False],
            [[""], "gas use with CC", True],
            [[""], "gas use without CC", False],
        ],
        "D",
        "C",
        "2050",
        filter_positive=False,
        multiplier=-1,
        remove_list=["pipeline"],
    )
    export_results(gas_use_2050, "gas_use_2050.csv")

    weighted_prices_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "weighted_prices",
        [
            ["agricultural waste"],
            ["fuelwood residues"],
            ["secondary forestry residues"],
            ["sawdust"],
            ["residues from landscape care"],
            ["grasses"],
            ["woody crops"],
            ["fuelwoodRW"],
            ["manure"],
            ["sludge"],
            ["C&P_RW"],
            ["oil"],
            ["gas"],
            ["coal"],
            ["biomass import"],
            ["solid biomass"],
            ["biogas"],
        ],
        [],
        "B",
        "A",
        "2050",
        filter_positive=True,
        remove_list=["agriculture machinery oil"],
    )
    co2_prices_fossil = {
        "default": {
            "oil": 192.6 * 0.2571,
            "oil primary": 192.6 * 0.2571,
            "gas": 192.6 * 0.198,
        },
        "biomass_emissions": {
            "oil": 467.8 * 0.2571,
            "oil primary": 467.8 * 0.2571,
            "gas": 467.8 * 0.198,
        },
    }
    weighted_prices_2050 = add_costs(weighted_prices_2050)
    weighted_prices_2050 = add_co2_price(
        weighted_prices_2050, co2_prices_fossil, column="values"
    )
    weighted_prices_2050 = add_co2_price(
        weighted_prices_2050, co2_prices_fossil, column="costs"
    )
    # weighted_prices_2050 = add_co2_price(weighted_prices_2050, co2_prices_biomass, column="costs")
    export_results(weighted_prices_2050, "weighted_prices_2050.csv", add_costs=True)

    solid_biomass_supply = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["solid biomass", "links"]],
        [],
        "D",
        "C",
        "2050",
        filter_positive=True,
        remove_list=["biomass transport"],
    )
    digestable_biomass_supply = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["biogas", "links"]],
        [],
        "D",
        "C",
        "2050",
        filter_positive=True,
    )
    all_gas_generation_2050 = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["gas", "generators"], ["gas", "links"]],
        [],
        "D",
        "C",
        "2050",
        filter_positive=True,
        remove_list=["gas pipeline"],
    )

    co2_solid_biomass = 0.3667
    co2_digestable_biomass = 0.2848

    carbon_from_solid_biomass = calculate_carbon_removal(
        solid_biomass_supply, co2_solid_biomass, co2_solid_biomass
    )
    total_biomass_carbon = calculate_carbon_removal(
        digestable_biomass_supply,
        co2_digestable_biomass,
        co2_digestable_biomass,
        existing_dict=carbon_from_solid_biomass,
    )

    all_biomass_co2_capture = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [
            ["CO2 stored", "links", "urban central solid biomass CHP CC"],
            ["CO2 stored", "links", "BioSNG CC"],
            ["CO2 stored", "links", "biomass to liquid CC"],
            ["CO2 stored", "links", "biogas to gas CC"],
            ["CO2 stored", "links", "lowT industry solid biomass CC"],
            ["CO2 stored", "links", "solid biomass for mediumT industry CC"],
            ["CO2 stored", "links", "solid biomass to hydrogen"],
            ["CO2 stored", "links", "urban central solid biomass CHP CC"],
        ],
        [],
        "D",
        "C",
        "2050",
        filter_positive=True,
    )
    all_gas_capture = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [
            ["CO2 stored", "links", "SMR CC"],
            ["CO2 stored", "links", "gas for highT industry CC"],
            ["CO2 stored", "links", "gas for mediumT industry CC"],
            ["CO2 stored", "links", "lowT industry methane CC"],
            ["CO2 stored", "links", "urban central CHP CC"],
        ],
        [],
        "D",
        "C",
        "2050",
        filter_positive=True,
        remove_list=["urban central solid biomass CHP CC4"],
    )
    gas_shares = calc_gas_share(all_gas_generation_2050, scenarios)

    seq_biomass = calculate_carbon_removal(
        all_biomass_co2_capture,
        1,
        1,
        capture_rate=1,
        existing_dict=total_biomass_carbon,
        is_removed=True,
        add_to_total=False,
    )
    seq_biomass2 = calculate_carbon_removal(
        all_gas_capture,
        1,
        1,
        capture_rate=1,
        existing_dict=seq_biomass,
        is_removed=True,
        add_to_total=False,
        gas_shares=gas_shares,
    )
    with_share = calculate_removal_share(seq_biomass2)
    export_carbon_removal(with_share, "carbon_removal.csv")

    all_biomass_supply = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["solid biomass", "links"], ["biogas", "links"]],
        [
            [[""], "biomass", None],
        ],
        "D",
        "C",
        "2050",
        filter_positive=True,
        remove_list=["biomass transport"],
    )
    export_results(all_biomass_supply, "all_biomass_supply.csv")

    co2_use = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["CO2 stored", "links"]],
        [],
        "D",
        "C",
        "2050",
        filter_positive=False,
        multiplier=-1,
    )
    export_results(co2_use, "co2_use.csv", include_share=True)

    co2_capture = aggregate_and_calculate_share(
        results,
        scenarios,
        "supply_energy",
        [["CO2 stored", "links"]],
        [],
        "D",
        "C",
        "2050",
        filter_positive=True,
    )
    export_results(co2_capture, "co2_capture.csv", include_share=True)


if __name__ == "__main__":
    main()

# Other areas of the energy sector that are significantly affected? (wind and solar, fossil fuels, CCS)
