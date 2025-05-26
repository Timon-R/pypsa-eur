# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import os
import string

import pandas as pd
import pypsa
import yaml

def get_emission_factors(config_file_path = "config/config.yaml", new_names=False, add_imported_biomass=False):

    with open(config_file_path) as file:
        config = yaml.safe_load(file)

    emission_factors = config["biomass"]["emission_factors"]

    if new_names:
        new_emission_factors = {}
        new_names_dict = {
            "woody crops": "woody crops",
            "grasses": "grasses",
            "fuelwoodRW": "stemwood",
            "C&P_RW": "chips and pellets",
            "secondary forestry residues": "secondary forestry residues",
            "sawdust": "sawdust",
            "fuelwood residues": "logging residues",
            "agricultural waste": "crop residues",
            "residues from landscape care": "residues from landscape care",
            "sludge": "sludge",
            "manure": "manure",
            "solid biomass import": "imported biomass",
        }  
        for key, value in emission_factors.items():
            if key in new_names_dict:
                new_emission_factors[new_names_dict[key]] = value
            else:
                new_emission_factors[key] = value

        if add_imported_biomass:
            new_emission_factors["imported biomass"] = round(config["sector"]["solid_biomass_import"]["upstream_emissions_factor"]*0.3667,4)

        return new_emission_factors
    
    if add_imported_biomass:
            emission_factors["solid biomass import"] = round(config["sector"]["solid_biomass_import"]["upstream_emissions_factor"]*0.3667,4)

    return emission_factors

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
            if "custom_metrics" not in file and "metrics" not in file and "weighted_prices" not in file:
                df = pd.read_csv(file_path).drop(index=range(3))
            elif "weighted_prices" in file:
                df = pd.read_csv(file_path).drop(index=range(1))
            else:
                df = pd.read_csv(file_path).drop(index=range(2))
            df.columns = list(string.ascii_uppercase[: len(df.columns)])
            key = os.path.splitext(file)[0]
            dataframes[key] = df.reset_index(drop=True)

        results[folder] = dataframes
    return results

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
    scenario2="carbon_costs",
    add_costs=False,
    simply_print=False,
    export_dir="export",
    round_digits = 2,
):
    """
    Export the results to a CSV file.

    Parameters
    ----------
    data (dict): Dictionary containing the processed data.
    filename (str): Name of the CSV file to save.
    """
    os.makedirs(export_dir, exist_ok=True)
    rows = []
    if simply_print:
        # for every key, add all the items of the content dictionary to the row
        for key, content in data.items():
            rows.append({**content})
    else:
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
                    rows[-1]["Costs"] = round(content["costs"], round_digits)
                else:
                    rows[-1]["Costs"] = content["costs"]
    df = pd.DataFrame(rows)
    file_path = os.path.join(export_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Results exported to {file_path}")


def get_data(
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
    optional_columns = None,
    round_digits = 1,
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
    optional_column (list of lists, optional): List of columns to add to the result. Default is None.

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
                    field_condition &= (data_df.iloc[:, i] == field) | (field == "")
                condition |= field_condition

            data_df["condition"] = condition.fillna(False)
            # Filter data based on condition
            data = data_df[data_df["condition"]].copy()
            data.drop(columns=["condition"], inplace=True)

            # Apply remove_list filter
            if remove_list:
                data = data[
                    ~data[data_name_column].isin(remove_list)  # Use exact match instead of contains
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

            # Calculate total sum for share calculation
            total_sum = data[value_column].sum()

            # Merge the values using the merge_fields
            for merge_conditions, new_name in merge_fields:
                merged_data = pd.DataFrame()
                for merge_condition in merge_conditions:
                    added_data = data[
                        data[data_name_column] == merge_condition
                    ]
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
                            merged_data[value_column].sum() * multiplier, round_digits
                        ),
                    }
                    if calculate_share:
                        share = merged_data[value_column].sum() / total_sum
                        new_row["share"] = round(share, 4)
                    if optional_columns is not None:
                        for i in range(len(optional_columns)):
                            new_row[optional_columns[i][0]] = merged_data[optional_columns[i][1]].sum()
                        
                    result_data[f"{folder}_{year}_{new_name}"] = new_row

            # Add remaining data and calculate shares if calculate_share is True
            if not data.empty:
                for _, row in data.iterrows():
                    key = f"{folder}_{year}_{row[data_name_column]}"
                    new_row = {
                        "folder": folder,
                        "year": year,
                        "data_name": row[data_name_column],
                        "values": round(row[value_column] * multiplier, round_digits),
                    }
                    if calculate_share:
                        share = row[value_column] / total_sum
                        new_row["share"] = round(share, 4)
                    if optional_columns is not None:
                        for i in range(len(optional_columns)):
                            new_row[optional_columns[i][0]] = row[optional_columns[i][1]]
                    #if key already exists, add i+1 to the end of the key
                    if key in result_data:
                        i = 1
                        while f"{key}_{i}" in result_data:
                            i += 1
                        key = f"{key}_{i}"
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
    scenarios=["default", "carbon_costs"],
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
            carbon_stored = 0
            total_carbon = 0
        else:
            carbon_stored = existing_dict[folder]["carbon_stored"]
            total_carbon = existing_dict[folder]["total_carbon"]
        for key, content in dict.items():
            if content["folder"] == folder:
                if content["values"] is not None:
                    if is_removed:
                        carbon_stored += (
                            content["values"] * carbon_removal * capture_rate
                        )
                    if add_to_total:
                        total_carbon += content["values"] * carbon_intensity
            results[folder] = {
                "carbon_stored": carbon_stored,
                "total_carbon": total_carbon,
            }
    return results


def calculate_share(dict):
    folders = list(set([content["folder"] for content in dict.values()]))
    totals = {}
    for folder in folders:
        total = 0
        for key, content in dict.items():
            if content["folder"] == folder:
                total += content["values"]
        totals[folder] = total
    results = {}
    for key, content in dict.items():
        share = content["values"] / totals[content["folder"]]
        results[key] = {
            "folder": content["folder"],
            "year": content["year"],
            "data_name": content["data_name"],
            "values": content["values"],
            "share": share,
        }
    return results


def calculate_removal_share(dict):
    results = {}
    for key, content in dict.items():
        carbon_stored = content["carbon_stored"]
        total_carbon = content["total_carbon"]
        share_removed = carbon_stored / total_carbon
        results[key] = {
            "carbon_stored": carbon_stored,
            "total_carbon": total_carbon,
            "share_stored": share_removed,
        }
    return results


def export_carbon_removal(data, filename, export_dir="export"):
    os.makedirs(export_dir, exist_ok=True)
    # Simply make the dict into a csv
    rows = []
    for key, content in data.items():
        rows.append(
            {
                "Folder": key,
                "Carbon Stored": content["carbon_stored"],
                "Carbon Utilised": content["carbon_utilised"],
                "Total Carbon": content["total_carbon"],
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


def split_CHP(data):
    electrictiy_share = 0.248
    heat_share = 0.752
    new_entries = []
    updated_entries = []
    deleted_entries = []

    for key, content in data.items():
        if "CHP" in content["data_name"]:
            # Check if an entry with "electricity production" already exists
            electricity_exists = False
            for key2, content2 in data.items():
                if (
                    content2["folder"] == content["folder"]
                    and content2["year"] == content["year"]
                    and content2["data_name"] == "electricity production"
                ):
                    updated_entries.append(
                        (
                            key2,
                            {
                                "folder": content2["folder"],
                                "year": content2["year"],
                                "data_name": "electricity production",
                                "values": content2["values"]
                                + content["values"] * electrictiy_share,
                                "share": content2["share"]
                                + content["share"] * electrictiy_share,
                            },
                        )
                    )
                    electricity_exists = True
                    break

            if not electricity_exists:
                new_key = key + "_electricity"
                new_entries.append(
                    (
                        new_key,
                        {
                            "folder": content["folder"],
                            "year": content["year"],
                            "data_name": "electricity production",
                            "values": content["values"] * electrictiy_share,
                            "share": content["share"] * electrictiy_share,
                        },
                    )
                )

            # Check if an entry with "heat production" already exists
            heat_exists = False
            for key2, content2 in data.items():
                if (
                    content2["folder"] == content["folder"]
                    and content2["year"] == content["year"]
                    and "heat production" in content2["data_name"]
                ):
                    updated_entries.append(
                        (
                            key2,
                            {
                                "folder": content2["folder"],
                                "year": content2["year"],
                                "data_name": "heat production",
                                "values": content2["values"]
                                + content["values"] * heat_share,
                                "share": content2["share"]
                                + content["share"] * heat_share,
                            },
                        )
                    )
                    heat_exists = True
                    break

            if not heat_exists:
                new_key = key + "_heat"
                new_entries.append(
                    (
                        new_key,
                        {
                            "folder": content["folder"],
                            "year": content["year"],
                            "data_name": "heat production",
                            "values": content["values"] * heat_share,
                            "share": content["share"] * heat_share,
                        },
                    )
                )

            deleted_entries.append(key)

    # Update the dictionary after iteration
    for new_key, new_content in new_entries:
        data[new_key] = new_content
    for key2, updated_content in updated_entries:
        data[key2] = updated_content
    for key in deleted_entries:
        del data[key]

    # Sort the dictionary
    data = dict(sorted(data.items()))

    return data


def calculate_supply_difference_and_emission_difference(
    data, scenario1, scenario2, year
):
    results = {}
    emission_factors = get_emission_factors(
        config_file_path="config/config.yaml",
        new_names=False,
        add_imported_biomass=True,
    )
    # find all rows that match year and data name and calculate the difference between folder1 and folder2
    for key, content in data.items():
        # remove number 1 from data name
        content["data_name"] = content["data_name"].replace("1", "")
        if content["year"] == year:
            if content["folder"] == scenario1:
                key2 = key.replace(scenario1, scenario2)
                if key2 in data:
                    difference = content["values"] - data[key2]["values"]
                    new_key = key.replace(scenario1, f"{scenario1}_{scenario2}")
                    results[new_key] = {
                        "year": year,
                        "data_name": content["data_name"],
                        f"values_{scenario1}": content["values"],
                        f"values_{scenario2}": data[key2]["values"],
                        "difference": difference,
                        "emission_difference": difference
                        * emission_factors[content["data_name"]],
                    }
    results["total"] = {
        "year": year,
        "data_name": "total",
        f"values_{scenario1}": sum(
            [
                content["values"]
                for content in results.values()
                if "values" in content and scenario1 in content
            ]
        ),
        f"values_{scenario2}": sum(
            [
                content["values"]
                for content in results.values()
                if "values" in content and scenario2 in content
            ]
        ),
        "difference": sum(
            [
                content["difference"]
                for content in results.values()
                if "difference" in content
            ]
        ),
        "emission_difference": sum(
            [
                content["emission_difference"]
                for content in results.values()
                if "emission_difference" in content
            ]
        ),
    }

    return results


def add_carbon_utilisation(dict_with_storage, data, share_sequestered):
    #not all of the carbon stored is sequestered, so we need to add that amount to the carbon utilised and substract it from the carbon stored
    for key, content in dict_with_storage.items():
        for key1, content1 in data.items():
            if content1["folder"] == key:
                shift_from_stored_to_utilised = content["carbon_stored"] * (1 - share_sequestered[key]["share"])
                dict_with_storage[key]["carbon_utilised"] = content1[
                    "values"
                ] + shift_from_stored_to_utilised
                dict_with_storage[key]["carbon_stored"] = (
                    dict_with_storage[key]["carbon_stored"]- shift_from_stored_to_utilised
                )
    # for key, content in dict_with_storage.items():
    #     content["share_utilised"] = content["carbon_utilised"] / content["total_carbon"]
    return dict_with_storage


def get_biomass_potentials(
    network_path="results/carbon_costs/networks/base_s_39___2050.nc", export_dir="export"
):
    config_file_path = "config/config.yaml"

    # Open and load the YAML file
    with open(config_file_path) as file:
        config = yaml.safe_load(file)

    # Extract biomass types
    biomass_types = list(config["biomass"]["classes"].keys())

    n = pypsa.Network(network_path)

    biomass_potentials = {}

    for biomass_type in biomass_types:
        biomass_stores = n.stores[n.stores.carrier == biomass_type]
        biomass_potentials[biomass_type] = biomass_stores.e_initial.sum()

    os.makedirs(export_dir, exist_ok=True)
    rows = []
    for key, content in biomass_potentials.items():
        rows.append(
            {
                "Biomass Type": key,
                "Potential": content,
            }
        )
    df = pd.DataFrame(rows)
    file_path = os.path.join(export_dir, "biomass_potentials.csv")
    df.to_csv(file_path, index=False)


def calculate_upstream_emissions(data, scenarios):
    emission_factors = get_emission_factors(
        config_file_path="config/config.yaml",
        new_names=False,
        add_imported_biomass=True,
    )
    results = {}
    for key, content in data.items():
        for scenario in scenarios:
            if content["folder"] == scenario:
                results[key] = {
                    "folder": content["folder"],
                    "data_name": content["data_name"],
                    "year": content["year"],
                    "upstream emissions": content["values"]
                    * emission_factors[content["data_name"]],
                }
    # calculate the total upstream emissions
    for scenario in scenarios:
        total_upstream_emissions = sum(
            [
                content["upstream emissions"]
                for content in results.values()
                if content["folder"] == scenario
            ]
        )
        results[f"{scenario}_total"] = {
            "folder": scenario,
            "data_name": "total",
            "year": content["year"],
            "upstream emissions": total_upstream_emissions,
        }
    return results


def calculate_share_of_sequestration(data, scenarios):
    results = {}
    for scenario in scenarios:
        total = 0
        for key, content in data.items():
            if content["folder"] == scenario:
                total += content["values"]
        for key, content in data.items():
            if content["folder"] == scenario:
                if "sequestered" in content["data_name"]:
                    share = content["values"] / total
                    results[content["folder"]] = {
                        "folder": content["folder"],
                        "share": share,
                    }
    return results

def calc_beccus(results, scenarios, solid_biomass_supply, digestable_biomass_supply, gas_shares, share_sequestered):   
    co2_solid_biomass = 0.3667
    co2_digestable_biomass = 0.2848 #correctly calculated using CO2 stored plus gas intensity

    carbon_from_solid_biomass = calculate_carbon_removal(
        solid_biomass_supply, co2_solid_biomass, co2_solid_biomass, scenarios=scenarios, capture_rate=1
    )
    total_biomass_carbon = calculate_carbon_removal(
        digestable_biomass_supply,
        co2_digestable_biomass,
        co2_digestable_biomass,
        existing_dict=carbon_from_solid_biomass,
        scenarios=scenarios,
        capture_rate=1,
    )

    all_biomass_co2_storage = get_data(
        results,
        scenarios,
        "energy_balance",
        [
            ["Link", "urban central solid biomass CHP CC", "co2 stored"],
            ["Link", "BioSNG CC", "co2 stored"],
            ["Link", "biomass to liquid CC", "co2 stored"],
            ["Link", "biogas to gas CC", "co2 stored"],
            ["Link", "lowT industry solid biomass CC", "co2 stored"],
            ["Link", "solid biomass for mediumT industry CC", "co2 stored"],
            ["Link", "solid biomass to hydrogen", "co2 stored"],
            ["Link", "urban central solid biomass CHP CC", "co2 stored"],
        ],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
    )
    all_gas_co2_storage = get_data(
        results,
        scenarios,
        "energy_balance",
        [
            ["Link", "SMR CC", "co2 stored"],
            ["Link", "gas for highT industry CC", "co2 stored"],
            ["Link", "gas for mediumT industry CC", "co2 stored"],
            ["Link", "lowT industry methane CC", "co2 stored"],
            ["Link", "urban central CHP CC", "co2 stored"],
        ],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
        remove_list=["urban central solid biomass CHP CC"],
    )

    seq_biomass = calculate_carbon_removal(
        all_biomass_co2_storage,
        1,
        1,
        capture_rate=1,
        existing_dict=total_biomass_carbon,
        is_removed=True,
        add_to_total=False,
        scenarios=scenarios,
    )
    seq_biomass2 = calculate_carbon_removal(
        all_gas_co2_storage,
        1,
        1,
        capture_rate=1,
        existing_dict=seq_biomass,
        is_removed=True,
        add_to_total=False,
        gas_shares=gas_shares, #subtracting the share of sequestered gas carbon stemming from natural gas
        scenarios=scenarios,
    )
    #until here total carbon is correctly calculated as well as total amount of biogenic carbon stored

    carbon_utilisation = get_data(
        results,
        scenarios,
        "energy_balance",
        [
            ["Link", "biomass to liquid", "co2"],
            ["Link", "electrobiofuels", "co2"],
            ["Link", "biogas to gas", "co2"],
            ["Link", "BioSNG", "co2"],
        ],
        [[[""], "All utilised", None]],
        "D",
        "B",
        "2050",
        filter_positive=False,
        multiplier=-1,
        calculate_share=False,
    )

    #substract the stored carbon in the biogas from the carbon utilisation
    for key, content in carbon_utilisation.items():
        for key1, content1 in all_gas_co2_storage.items():
            if content1["folder"] == content["folder"]:
                carbon_utilisation[key]["values"] -= (content1["values"]*(1-gas_shares[content["folder"]]["share"]))

    other_that_need_to_be_removed = get_data(
        results,
        scenarios,
        "energy_balance",
        [
            ["Link", "biomass to liquid CC", "co2 stored"],
            ["Link", "biogas to gas CC", "co2 stored"],
        ],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
    )
    #substract the stored carbon from the carbon utilisation from biomass to liquid CC and biogas to gas CC (otherwise counted for both stored and utilised)
    for key, content in carbon_utilisation.items():
        for key1, content1 in other_that_need_to_be_removed.items():
            if content1["folder"] == content["folder"]:
                carbon_utilisation[key]["values"] -= content1["values"]

    #not all of the carbon stored is sequestered, so we need to add that amount to the carbon utilised and substract it from the carbon stored
    carbon_utilisation = add_carbon_utilisation(
        seq_biomass2, carbon_utilisation, share_sequestered
    )
    
    return carbon_utilisation

def modify_co2_data(data, threshold=0): 
    new_data = {}
    sink_dict = {
        "HVC to air": "HVC",
        "OCGT": "gas",
        "SMR": "gas",
        "SMR CC": "gas",
        "agriculture machinery oil": "liquid fuels",
        "biogas to gas": "gas",
        "biogas to gas CC": "gas",
        "biomass to liquid": "liquid fuels",
        "biomass to liquid CC": "liquid fuels",
        "electrobiofuels": "liquid fuels",
        "gas for highT industry": "gas",
        "industry methanol": "gas",
        "kerosene for aviation": "liquid fuels",
        "methanolisation": "liquid fuels",
        "municipal solid waste": "municipal solid waste",
        "oil refining": "crude oil",
        "process emissions CC": "cement",
        "rural gas boiler": "gas",
        "shipping methanol": "liquid fuels",
        "solid biomass for mediumT industry CC": "solid biomass",
        "urban central gas CHP": "gas",
        "urban central gas boiler": "gas",
        "urban decentral gas boiler": "gas",
        "Fischer-Tropsch": "liquid fuels",
        "waste CHP CC": "municipal solid waste",
        "onwind landuse emission": "indirect land use emissions from renewables",
        "solar landuse emission": "indirect land use emissions from renewables",
        "solar-hsat landuse emission": "indirect land use emissions from renewables",
        "agricultural waste": "indirect emissions from biomass",
        "fuelwood residues": "indirect emissions from biomass",
        "fuelwoodRW": "indirect emissions from biomass",
        "manure": "indirect emissions from biomass",
        "sludge": "indirect emissions from biomass",
        "secondary forestry residues": "indirect emissions from biomass",
        "sawdust": "indirect emissions from biomass",
        "residues from landscape care": "indirect emissions from biomass", 
        "grasses": "indirect emissions from biomass",
        "woody crops": "indirect emissions from biomass",
        "C&P_RW": "indirect emissions from biomass",
        "gas for highT industry CC": "gas",
        "gas for mediumT industry CC": "gas",
        "lowT industry solid biomass CC": "solid biomass",
        "urban central solid biomass CHP CC": "solid biomass",
        "urban central gas CHP CC": "gas",
        "BioSNG CC": "gas",
        "gas for industry": "gas",
        "solid biomass for industry CC": "solid biomass",
        "solid biomass to hydrogen": "solid biomass",
        "Sabatier": "gas",
        "gas for industry CC": "gas",
        "solid biomass import": "solid biomass",
    }
    for key, content in data.items():
        if abs(content["values"]) < threshold:
            continue
        if content["data_name"] == "DAC" or content["data_name"] == "solid biomass for mediumT industry CC" :
            # Check if there is already a DAC entry with the same folder
            existing_dac = any(
                entry["data_name"] == content["data_name"] and entry["folder"] == content["folder"]
                for entry in new_data.values()
            )
            if not existing_dac:
                new_data[key] = content
                new_data[key]["values"] = abs(content["values"])
                new_data[key]["from_sink"] = "co2"
                new_data[key]["to_sink"] = "co2 stored"
        elif content["data_name"] == "co2 sequestered":
            # Check if there is already a co2 sequestered entry with the same folder
            existing_sequestered = any(
                entry["data_name"] == "co2 sequestered" and entry["folder"] == content["folder"]
                for entry in new_data.values()
            )
            if not existing_sequestered:
                new_data[key] = content
                new_data[key]["values"] = abs(content["values"])
                new_data[key]["from_sink"] = "co2 stored"
                new_data[key]["to_sink"] = "co2 sequestered"
        elif content["data_name"] == "biogas to gas CC" or content["data_name"] == "biomass to liquid CC":
            if content["emission_type"] == "co2 stored":
                new_data[key] = content
                new_data[key]["values"] = abs(content["values"])
                new_data[key]["from_sink"] = "co2"
                new_data[key]["to_sink"] = "co2 stored"
        else:
            new_data[key] = content
            if content["values"] >= 0:
                new_data[key]["from_sink"] = sink_dict[content["data_name"]]
                new_data[key]["to_sink"] = content["emission_type"]
            else:
                new_data[key]["from_sink"] = content["emission_type"]
                new_data[key]["to_sink"] = sink_dict[content["data_name"]]
                new_data[key]["values"] = abs(content["values"])

    biogas_difference_by_folder = {}
    biomass_difference_by_folder = {}
    for key, content in data.items():
        folder = content["folder"]
        if content["data_name"] == "biogas to gas CC":
            if folder not in biogas_difference_by_folder:
                biogas_difference_by_folder[folder] = 0
            biogas_difference_by_folder[folder] += content["values"]
        elif content["data_name"] == "biomass to liquid CC":
            if folder not in biomass_difference_by_folder:
                biomass_difference_by_folder[folder] = 0
            biomass_difference_by_folder[folder] += content["values"]

    # Create new entries for each folder
    for folder, difference in biogas_difference_by_folder.items():
        new_data[f"{folder}_biogas to gas CC - gas"] = {
            "folder": folder,
            "data_name": "biogas to gas CC",
            "values": abs(difference),
            "from_sink": "co2",
            "to_sink": "gas",
        }

    for folder, difference in biomass_difference_by_folder.items():
        new_data[f"{folder}_biomass to liquid CC - liquid fuels"] = {
            "folder": folder,
            "data_name": "biomass to liquid CC",
            "values": abs(difference),
            "from_sink": "co2",
            "to_sink": "liquid fuels",
        }
   
    for key, content in new_data.items():
        for key1, content1 in content.items():
            if content1 == "co2":
                new_data[key][key1] = "atmosphere"
            elif content1 == "co2 stored":
                new_data[key][key1] = "co2 captured"
    return new_data

def merge_data(data_dict, merge_fields, data_name_key, value_key, excepted_keys, rename_key=None, combine_key=None):
    """
    Merge data entries based on merge_fields where all non-excepted keys match.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing the data to merge
    merge_fields : list
        List of lists containing merge conditions, new name, and case sensitivity
    data_name_key : str
        Key used to identify entries for merging based on conditions
    value_key : str
        Key containing values to sum when merging
    excepted_keys : list
        Keys that do not need to match for entries to be merged
    rename_key : str, optional
        Key to update with the new name in merged entries. If None, uses data_name_key
    combine_key : str, optional
        If specified, combines values from all entries for this key instead of using the first entry
    
    Returns
    -------
    dict
        Dictionary with merged entries
    """
    if rename_key is None:
        rename_key = data_name_key
    new_data = {}
    
    # Copy the original data to ensure all keys are represented
    for key, content in data_dict.items():
        new_data[key] = content.copy()
    
    # Process each merge field
    for merge_conditions, new_name, is_cc in merge_fields:
        # Find all keys that match the merge conditions
        matching_keys = []
        for key, content in data_dict.items():
            # Check if entry matches the conditions exactly (not substring)
            if any(content[data_name_key] == field for field in merge_conditions):
                matching_keys.append(key)
        
        # Group matching keys by their non-excepted attributes
        groups = {}
        for key in matching_keys:
            # Create a tuple of all values except for excepted_keys
            group_key = tuple((k, data_dict[key][k]) for k in sorted(data_dict[key].keys()) 
                            if k != data_name_key and k != value_key and k not in excepted_keys)
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(key)
        
        # Process each group
        for group_keys in groups.values():
            if len(group_keys) <= 1:
                continue  # Skip if only one item in group
                
            # Create a new key for the merged entry
            new_key = "_".join(group_keys)
            
            # Create the merged entry using the first entry as template
            template_key = group_keys[0]
            new_data[new_key] = data_dict[template_key].copy()
            
            # If combine_key is specified, combine values from all entries for that key
            if combine_key is not None:
                combined_values = [data_dict[k][combine_key] for k in group_keys]
                new_data[new_key][combine_key] = "_".join(combined_values)
            
            # Update with new name and sum of values
            new_data[new_key][rename_key] = new_name  # Use rename_key instead of data_name_key
            new_data[new_key][value_key] = sum(data_dict[k][value_key] for k in group_keys)
            
            # Remove the original entries that were merged
            for k in group_keys:
                if k in new_data:
                    del new_data[k]
    
    return new_data

def main(results_dir="results", export_dir="export",scenarios=["default", "carbon_costs"]):

    results = load_results(results_dir, scenarios)

    electricity_generation_share = get_data(
        results,
        scenarios,
        "energy_balance",
        [
            ["Generator", "", "AC"],
            ["Generator", "solar rooftop", "low voltage"],
            ["Link", "", "AC"],
            ["StorageUnit", "hydro", "AC"],
        ],
        [
            [["wind"], "wind"],
            [["solar","solar-hsat","solar rooftop"], "solar"],
            [["hydro", "ror"], "hydro"],
            [["urban central solid biomass CHP","urban central solid biomass CHP CC"], "biomass CHP"],
            [["waste CHP","waste CHP CC"], "waste CHP"],
            [["urban central gas CHP","urban central gas CHP CC"], "gas CHP"],
        ],  # merge_fields
        "D",
        "B",
        "2050",
        filter_positive=True,
        calculate_share=True,
        remove_list=["H2 Fuel Cell","battery discharger"],
    )
    export_results(
        electricity_generation_share,
        "electricity_generation_share.csv",
        include_share=True,
        export_dir=export_dir
    )

    fields_list = [
        ["Generator", "", "bioliquids"],
        ["Generator", "", "AC"],
        ["Link", "waste", "AC"],
        ["StorageUnit", "hydro", "AC"],
        ["Generator", "", "biogas"],
        ["Link", "", "biogas"],
        ["Generator", "", "coal"],
        ["Generator", "", "gas"],
        ["Generator", "solar rooftop", "low voltage"],
        ["Generator", "", "waste"],
        ["Generator", "", "oil primary"],
        ["Link", "", "solid biomass"],
        ["Generator", "", "solid biomass"],
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
        ],
        [["manure", "sludge"], "biogas"],
        [["onwind","offwind-ac","offwind-dc","offwind-float"], "wind"],
        [["solar","solar-hsat","solar rooftop"], "solar"],
        [["hydro", "ror"], "hydro"],
    ]
    remove_list = ["biomass transport", "waste CHP"]
    primary_energy = get_data(
        results,
        scenarios,
        "energy_balance",
        fields_list,
        merge_fields,
        "D",
        "B",
        "2050",
        filter_positive=True,
        remove_list=remove_list,
    )
    export_results(primary_energy, "primary_energy.csv", include_share=True, export_dir=export_dir)

    costs = get_data(
        results,
        scenarios,
        "metrics",
        [["total costs"]],
        [[["total costs"], "Total costs (Billion €)"]],
        "B",
        "A",
        "2050",
        1e-9,
    )
    export_results(costs, "costs2050.csv", export_dir=export_dir)

    # Get the supply data for all biomass types
    biomass_supply = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "","solid biomass"], ["Link", "", "biogas"]],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
        remove_list=["biomass transport", "solid biomass for industry","solid biomass for industry CC"],
    )
    export_results(biomass_supply, "biomass_supply.csv", export_dir=export_dir)
    difference = calculate_supply_difference_and_emission_difference(
        biomass_supply, "default", "carbon_costs", "2050"
    )
    export_results(difference, "biomass_supply_difference.csv", simply_print=True, export_dir=export_dir)

    fossil_fuel_supply = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Generator", "", "oil primary"], ["Generator", "", "gas"], ["Generator", "", "coal"]],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
    )
    export_results(fossil_fuel_supply, "fossil_fuel_supply.csv", export_dir=export_dir)

    merge_fields = [
        [[["", "", "wind"]], "wind", None],
        [[["", "", "waste CHP"]], "waste CHP with CC", True],
        [[["", "", "biomass import"]], "biomass import", None],
        [[["", "", "heat pump"]], "heat pumps", None],
        [[["", "", "electrobiofuels"]], "electrobiofuels", None],
        [[["", "", "biomass to liquid"]], "biomass to liquid", None],
        [[["", "", "solar"]], "solar", None],
        [[["", "", "H2"]], "hydrogen", None],
        [[["", "", "gas boiler"]], "biogas boiler", None],
        [[["", "", "oil primary"]], "primary oil", None],
        [[["", "", "biomass boiler"]], "biomass boiler", None],
        [[["", "", "biomass CHP"]], "biomass CHP", None],
        [[["", "", "nuclear"]], "nuclear", None],
        [[["", "Generator", "gas"]], "natural gas", None],
        [[["", "", "water pits"]], "thermal energy storage", None],
        [[["", "", "biogas"]], "biogas production with CC", True],
        [[["", "", "biogas"]], "biogas production without CC", False],
        [[["", "", "DAC"]], "DAC", None],
        [[["","","battery"]], "batteries", None],
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
                ["", "Line", "AC"],
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
        "carbon_costs",
        "costs",
        ["A", "B", "C"],
        "D",
        "2050",
        merge_fields,
        remove_list=[],
        round_digits=0,
    )
    export_results(cost_difference, "cost_difference.csv", include_difference=True, export_dir=export_dir)

    # shadow_price = get_data(
    #     results,
    #     scenarios,
    #     "metrics",
    #     [["co2_shadow"]],
    #     [[["co2_shadow"], "CO2 shadow price", None]],
    #     "B",
    #     "A",
    #     "2050",
    #     multiplier=-1,
    #     filter_positive=False,
    # )
    # export_results(shadow_price, "shadow_price.csv", export_dir=export_dir)

    # hydrogen_production = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "", "H2"]],
    #     [],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    #     remove_list=["pipeline"],
    # )
    # export_results(hydrogen_production, "hydrogen_production.csv", export_dir=export_dir)

    # heat_pumps = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "heat pump", "low voltage"]],
    #     [],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=False,
    #     multiplier=-1,
    # )
    # export_results(heat_pumps, "heat_pumps.csv", export_dir=export_dir)

    # gas_use = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["link", "", "gas"]],
    #     [
    #         [["biogas"], "biogas upgrading with CC", True],
    #         [["biogas"], "biogas upgrading without CC", False],
    #         [[""], "gas use with CC", True],
    #         [[""], "gas use without CC", False],
    #     ],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=False,
    #     multiplier=-1,
    #     remove_list=["pipeline"],
    # )
    # export_results(gas_use, "gas_use.csv", export_dir=export_dir)

    # weighted_prices = get_data(
    #     results,
    #     scenarios,
    #     "weighted_prices",
    #     [
    #         ["agricultural waste"],
    #         ["fuelwood residues"],
    #         ["secondary forestry residues"],
    #         ["sawdust"],
    #         ["residues from landscape care"],
    #         ["grasses"],
    #         ["woody crops"],
    #         ["fuelwoodRW"],
    #         ["manure"],
    #         ["sludge"],
    #         ["C&P_RW"],
    #         ["oil"],
    #         ["gas"],
    #         ["coal"],
    #         ["biomass import"],
    #         ["solid biomass"],
    #         ["biogas"],
    #     ],
    #     [],
    #     "B",
    #     "A",
    #     "2050",
    #     filter_positive=None,
    #     remove_list=["agriculture machinery oil"],
    # )

    # weighted_prices = add_costs(weighted_prices)
    # export_results(weighted_prices, "weighted_prices.csv", add_costs=True, export_dir=export_dir)

    # solid_biomass_supply = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "", "solid biomass"]],
    #     [],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    #     remove_list=["biomass transport"],
    # )
    # digestable_biomass_supply = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "","biogas"]],
    #     [],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    # )
    # all_gas_generation = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Generator", "", "gas"], ["Link", "", "gas"]],
    #     [],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    #     remove_list=["gas pipeline"],
    # )

    # co2_use = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "","co2 stored"]],
    #     [],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=False,
    #     multiplier=-1,
    # )
    # export_results(co2_use, "co2_use.csv", include_share=True, export_dir=export_dir)



    # co2_capture = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "","co2 stored"]],
    #     [],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    # )
    # export_results(co2_capture, "co2_capture.csv", include_share=True, export_dir=export_dir)


    # gas_shares = calc_gas_share(all_gas_generation, scenarios)
    # share_sequestered = calculate_share_of_sequestration(co2_use, scenarios)

    # carbon_utilisation = calc_beccus(
    #     results,
    #     scenarios,
    #     solid_biomass_supply,
    #     digestable_biomass_supply,
    #     gas_shares,
    #     share_sequestered,
    # )
    
    # export_carbon_removal(carbon_utilisation, "CCUS.csv", export_dir=export_dir)

    # all_biomass_supply = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "", "solid biomass"], ["Link", "", "biogas"]],
    #     [
    #         [[""], "biomass", None],
    #     ],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    #     remove_list=["biomass transport", "solid biomass for industry","solid biomass for industry CC"],
    # )
    # export_results(all_biomass_supply, "all_biomass_supply.csv", export_dir=export_dir)

    # # use "gas_shares"

    # merge_fields = [
    #     [["to liquid", "electrobiofuels","methanol"], "conversion to liquid fuels", None],
    #     [["industry", "boiler"], "heat production", None],
    #     [["CHP"], "CHP", None],
    #     [["hydrogen", "SMR"], "hydrogen production", None],
    #     [["OCGT"], "electricity production", None],
    # ]
    # solid_biomass_use_by_sector = (
    #     get_data(  # this doesn't account for the gas share yet
    #         results,
    #         scenarios,
    #         "energy_balance",
    #         [["Link", "", "solid biomass"]],
    #         merge_fields,
    #         "D",
    #         "B",
    #         "2050",
    #         filter_positive=False,
    #         multiplier=-1,
    #         calculate_share=False,
    #         remove_list=["transport", "import", "pipeline", "biogas to gas", "BioSNG","BioSNG CC"],
    #     )
    # )
    # digestable_biomass_use_by_sector = (
    #     get_data(  # this doesn't account for the gas share yet
    #         results,
    #         scenarios,
    #         "energy_balance",
    #         [["Link", "", "gas"]],
    #         merge_fields,
    #         "D",
    #         "B",
    #         "2050",
    #         filter_positive=False,
    #         multiplier=-1,
    #         calculate_share=False,
    #         remove_list=["transport", "import", "pipeline", "biogas to gas"],
    #     )
    # )
    # for key, content in digestable_biomass_use_by_sector.items():
    #     # mulitply the values by the 1-gas share
    #     content["values"] = content["values"] * (
    #         1 - gas_shares[content["folder"]]["share"]
    #     )

    # biomass_use_by_sector = solid_biomass_use_by_sector.copy()
    # for key, value in digestable_biomass_use_by_sector.items():
    #     if key in biomass_use_by_sector:
    #         biomass_use_by_sector[key]["values"] += value["values"]
    #     else:
    #         biomass_use_by_sector[key] = value
    # biomass_use_by_sector = calculate_share(biomass_use_by_sector)
    # biomass_use_by_sector = split_CHP(biomass_use_by_sector)

    # export_results(
    #     biomass_use_by_sector, "biomass_use_by_sector.csv", include_share=True, export_dir=export_dir
    # )
    # solid_biomass_use = (
    #     get_data(  # this doesn't account for the gas share yet
    #         results,
    #         scenarios,
    #         "energy_balance",
    #         [["Link", "", "solid biomass"]],
    #         [],
    #         "D",
    #         "B",
    #         "2050",
    #         filter_positive=False,
    #         multiplier=-1,
    #         calculate_share=False,
    #         remove_list=["transport", "import", "pipeline", "biogas to gas", "SNG"],
    #     )
    # )
    # digestable_biomass_use = (
    #     get_data(  # this doesn't account for the gas share yet
    #         results,
    #         scenarios,
    #         "energy_balance",
    #         [["Link", "", "gas"]],
    #         [],
    #         "D",
    #         "B",
    #         "2050",
    #         filter_positive=False,
    #         multiplier=-1,
    #         calculate_share=False,
    #         remove_list=["transport", "import", "pipeline", "biogas to gas"],
    #     )
    # )
    # for key, content in digestable_biomass_use.items():
    #     # mulitply the values by the 1-gas share
    #     content["values"] = content["values"] * (
    #         1 - gas_shares[content["folder"]]["share"]
    #     )

    # #merge digestable and solid biomass use by sector and sort dict
    # biomass_use = calculate_share({**digestable_biomass_use, **solid_biomass_use})
    # biomass_use = dict(sorted(biomass_use.items()))

    # export_results(
    #     biomass_use, "biomass_use.csv", include_share=True, export_dir=export_dir
    # )

    # upstream_emissions = calculate_upstream_emissions(biomass_supply, scenarios)
    # export_results(upstream_emissions, "upstream_emissions.csv", simply_print=True, export_dir=export_dir)

    # #get_biomass_potentials(export_dir=export_dir)

    # remove_list = ["agriculture machinery oil1"]
    # oil_production = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "","oil"]],
    #     [],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    #     remove_list=remove_list,
    #     calculate_share=True,
    # )
    # export_results(oil_production, "oil_production.csv", include_share=True, export_dir=export_dir)

    # beccs = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [
    #         ["Link", "CC", "solid biomass"],
    #     ],
    #     [],
    #     "D",
    #     "B",
    #     "2050",
    #     multiplier=-1,
    #     filter_positive=False,
    #     calculate_share=True,
    # )
    # export_results(beccs, "beccs.csv", export_dir=export_dir)

    # fields_list = [
    #     ["Link", "", "highT industry"],
    #     ["Link", "", "mediumT industry"],
    #     ["Link", "", "lowT industry"],
    # ]  # all industries
    # merge_fields = [
    #     [["biomass"], "biomass", False],
    #     [["biomass"], "biomass CC", True],
    #     [["gas", "methane"], "gas", False],
    #     [["gas", "methane"], "gas CC", True],
    #     [["hydrogen"], "hydrogen", False],
    #     [["heat pump", "electricity"], "electricity/heat pump", False],
    # ]
    # industrial_energy = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     fields_list,
    #     merge_fields,
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    # )
    # export_results(
    #     industrial_energy, "industrial_energy.csv", include_share=True, export_dir=export_dir
    # )

    # fields_list = [
    #     ["Link", "", "rural heat"],
    #     ["Link", "", "urban central heat"],
    #     ["Link", "", "urban decentral heat"],
    # ]
    # merge_fields = [
    #     [["biomass"], "biomass", False],
    #     [["biomass"], "biomass CC", True],
    #     [["waste"], "waste", False],
    #     [["waste"], "waste CC", True],
    #     [["gas", "CHP"], "gas", False],
    #     [["gas", "CHP"], "gas CC", True],
    #     [["H2"], "hydrogen", False],
    #     [["heat pump", "resistive"], "electricity/heat pump", False],
    #     [["water tanks discharger"], "water tank discharger", False],
    # ]
    # heating_energy = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     fields_list,
    #     merge_fields,
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    # )
    # export_results(heating_energy, "heating_energy.csv", include_share=True, export_dir=export_dir)

    # co2 = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "", "co2"],["Generator", "", "co2"]],
    #     [],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=None,
    #     optional_columns=[["emission_type", "C"]],
    # )

    # #remove the share and the year keys
    # for key, content in co2.items():
    #     if "share" in content:
    #         del content["share"]
    #     if "year" in content:
    #         del content["year"]

    # export_results(co2, "co2.csv", include_share=False, export_dir=export_dir, simply_print=True)

    # co2_sankey = modify_co2_data(co2,100)
    # for key, content in co2.items():
    #     if "emission_type" in content:
    #         del content["emission_type"]

    # export_results(
    #     co2_sankey,
    #     "co2_sankey.csv",
    #     include_share=False,
    #     export_dir=export_dir,
    #     simply_print=True,
    # )

    # results = load_results(results_dir, scenarios)
    # capacity_factors = get_data(
    #     results,
    #     scenarios,
    #     "capacity_factors",
    #     [["Generator", "solar"],["Generator", "solar-hsat"],["Generator", "onwind"]],
    #     [],
    #     "C",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    #     remove_list=["urban central solar thermal","urban decentral solar thermal","solar rooftop","rural solar thermal","onwind landuse emission","solar landuse emission","solar-hsat landuse emission"],
    #     round_digits =3,
    # )
    # export_results(
    #     capacity_factors,
    #     "capacity_factors.csv",
    #     export_dir=export_dir,
    #     round_digits = 3,
    # )


if __name__ == "__main__":
    results_dir = "results"

    # scenarios = ["default_optimal", "optimal", default_710_optimal, "710_optimal"]
    # export_dir = "export/seq"
    # TODO add a function to rename the scenarios in export_results (give dict for renaming)

    scenarios = ["default", "carbon_costs", "default_710","carbon_costs_710"]
    export_dir = "export/seq"

    # scenarios = ["default", "carbon_costs"]
    # export_dir = "export/basic"

    main(results_dir=results_dir, export_dir=export_dir, scenarios=scenarios)



    #################### MGA ####################
    # results_dir = "results"
    # scenarios = ["optimal", "max_0.025", "max_0.05","max_0.1","max_0.15","min_0.025","min_0.05","min_0.1","min_0.15"]
    # export_dir = "export/mga"
    # results = load_results(results_dir, scenarios)

    # all_biomass_supply = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "", "solid biomass"], ["Link", "", "biogas"]],
    #     [
    #         [[""], "biomass", None],
    #     ],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    #     remove_list=["biomass transport"],
    # )
    # export_results(all_biomass_supply, "biomass_use_carbon_costs.csv", export_dir=export_dir)

    # scenarios = ["710_optimal", "710_max_0.025", "710_max_0.05","710_max_0.1","710_max_0.15","710_min_0.025","710_min_0.05","710_min_0.1","710_min_0.15"]
    # export_dir = "export/mga"
    # results = load_results(results_dir, scenarios)

    # all_biomass_supply = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "", "solid biomass"], ["Link", "", "biogas"]],
    #     [
    #         [[""], "biomass", None],
    #     ],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    #     remove_list=["biomass transport"],
    # )
    # export_results(all_biomass_supply, "biomass_use_carbon_costs_710.csv", export_dir=export_dir)

    # results_dir = "results"
    # scenarios = ["default_optimal", "default_max_0.025", "default_max_0.05","default_max_0.1","default_max_0.15","default_min_0.025","default_min_0.05","default_min_0.1","default_min_0.15"]
    # export_dir = "export/mga"
    # results = load_results(results_dir, scenarios)

    # all_biomass_supply = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "", "solid biomass"], ["Link", "", "biogas"]],
    #     [
    #         [[""], "biomass", None],
    #     ],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    #     remove_list=["biomass transport"],
    # )
    # export_results(all_biomass_supply, "biomass_use_default.csv", export_dir=export_dir)

    # scenarios = ["default_710_optimal", "default_710_max_0.025", "default_710_max_0.05","default_710_max_0.1","default_710_max_0.15","default_710_min_0.025","default_710_min_0.05","default_710_min_0.1","default_710_min_0.15"]
    # export_dir = "export/mga"
    # results = load_results(results_dir, scenarios)

    # all_biomass_supply = get_data(
    #     results,
    #     scenarios,
    #     "energy_balance",
    #     [["Link", "", "solid biomass"], ["Link", "", "biogas"]],
    #     [
    #         [[""], "biomass", None],
    #     ],
    #     "D",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    #     remove_list=["biomass transport"],
    # )
    # export_results(all_biomass_supply, "biomass_use_default_710.csv", export_dir=export_dir)