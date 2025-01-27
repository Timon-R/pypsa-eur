# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import os
import string

import pandas as pd


def load_results(results_dir):
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
    for folder in os.listdir(results_dir):
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


def export_results(data, filename, include_share=False):
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
                    else:  # CC case sensitive must not be in the data_name
                        added_data = added_data[
                            ~added_data[data_name_column].str.contains(
                                "CC", case=True, na=False
                            )
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


def main():
    results_dir = "results"
    results = load_results(results_dir)
    # scenarios = "all"
    scenarios = ["endogenous_0_ef", "endogenous_1_ef"]
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
        "E",
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
        ],  # merge_fields
        "E",
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
        [["solid biomass", "links", "CC"]],
        [],
        "E",
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
        "E",
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
        "E",
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
                "manure",
                "sludge",
                "biomass",
            ],
            "biomass",
            False,
        ],
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
        "E",
        "C",
        "2050",
        filter_positive=True,
        remove_list=remove_list,
    )
    export_results(primary_energy_2050, "primary_energy_2050.csv", include_share=True)

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
    # costs_2050 = get_data(
    #     results,
    #     scenarios,
    #     "metrics",
    #     ["total costs"],
    #     "C",
    #     "Total costs (Billion €)",
    #     "2050",
    #     1e-9,
    # )
    # combined_costs = {**costs_2030, **costs_2050}
    # export_results(combined_costs, "combined_costs.csv")

    # # Get the supply data for all biomass types
    # all_biomass_supply = {}
    # for biomass in biomass_types:
    #     supply_data_2030 = get_data(
    #         results,
    #         scenarios,
    #         "supply_energy",
    #         [biomass, "stores", ""],
    #         "D",
    #         f"{biomass}",
    #         "2030",
    #     )
    #     all_biomass_supply.update(supply_data_2030)
    # for biomass in biomass_types:
    #     supply_data_2050 = get_data(
    #         results,
    #         scenarios,
    #         "supply_energy",
    #         [biomass, "stores", ""],
    #         "E",
    #         f"{biomass}",
    #         "2050",
    #     )
    #     all_biomass_supply.update(supply_data_2050)
    # export_results(all_biomass_supply, "biomass_supply.csv")

    # # Get the costs data for all biomass types
    # all_biomass_costs = {}
    # for biomass in biomass_types:
    #     costs_data_2030 = get_data(
    #         results,
    #         scenarios,
    #         "costs",
    #         ["stores", "marginal", biomass],
    #         "D",
    #         f"Costs of {biomass}",
    #         "2030",
    #     )
    #     all_biomass_costs.update(costs_data_2030)
    # for biomass in biomass_types:
    #     costs_data_2050 = get_data(
    #         results,
    #         scenarios,
    #         "costs",
    #         ["stores", "marginal", biomass],
    #         "E",
    #         f"Costs of {biomass}",
    #         "2050",
    #     )
    #     all_biomass_costs.update(costs_data_2050)
    # export_results(all_biomass_costs, "biomass_costs.csv")

    # # Redirection of biomass use
    # biomass_use_2030 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["solid biomass", "links", ""],
    #     "D",
    #     "Biomass use",
    #     "2030",
    # )
    # biomass_use_2050 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["solid biomass", "links", ""],
    #     "E",
    #     "Biomass use",
    #     "2050",
    # )
    # biomass_use_merged = {**biomass_use_2030, **biomass_use_2050}
    # # Filter out positive values and corresponding data names
    # biomass_use_sum = {
    #     key: {
    #         "folder": data["folder"],
    #         "year": data["year"],
    #         "data_name": data["data_name"],
    #         "values": abs(sum([float(v) for v in data["values"] if float(v) < 0])),
    #     }
    #     for key, data in biomass_use_merged.items()
    # }
    # export_results(biomass_use_sum, "biomass_use_sum.csv")

    # biomass_use_2030 = calculate_share(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["solid biomass", "links", ""],
    #     "D",
    #     "C",
    #     "2030",
    #     filter_positive=False,
    #     multiplier=-1,
    # )
    # biomass_use_2050 = calculate_share(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["solid biomass", "links", ""],
    #     "E",
    #     "C",
    #     "2050",
    #     filter_positive=False,
    #     multiplier=-1,
    # )
    # biomass_use = {**biomass_use_2030, **biomass_use_2050}
    # export_results(biomass_use, "biomass_use.csv", include_share=True)

    # # Shadow price of CO2
    # shadow_price_2030 = get_data(
    #     results, scenarios, "metrics", ["co2_shadow"], "B", "CO2 shadow price", "2030",multiplier= -1
    # )
    # shadow_price_2050 = get_data(
    #     results, scenarios, "metrics", ["co2_shadow"], "C", "CO2 shadow price", "2050",multiplier= -1
    # )
    # combined_shadow_price = {**shadow_price_2030, **shadow_price_2050}
    # export_results(combined_shadow_price, "combined_shadow_price.csv")

    # # Gas supply
    # gas_supply_2030 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["gas", "generators", "gas"],
    #     "D",
    #     "Gas supply",
    #     "2030",
    # )
    # gas_supply_2050 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["gas", "generators", "gas"],
    #     "E",
    #     "Gas supply",
    #     "2050",
    # )
    # gas_supply = {**gas_supply_2030, **gas_supply_2050}
    # # remove first value of the values list (biogas)
    # gas_supply = {
    #     key: {
    #         "folder": data["folder"],
    #         "year": data["year"],
    #         "data_name": data["data_name"],
    #         "values": data["values"][1:],
    #     }
    #     for key, data in gas_supply.items()
    # }
    # export_results(gas_supply, "gas_supply.csv")

    # oil_supply_2030 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["oil", "generators"],
    #     "D",
    #     "Oil extraction",
    #     "2030",
    # )
    # oil_supply_2050 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["oil", "generators"],
    #     "E",
    #     "Oil extraction",
    #     "2050",
    # )
    # oil_supply = {**oil_supply_2030, **oil_supply_2050}
    # export_results(oil_supply, "oil_supply.csv")

    # electricity_generation_share_2030 = calculate_share(
    #     results, scenarios, "supply_energy", ["AC", "generators"], "D", "C", "2030"
    # )
    # electricity_generation_share_2050 = calculate_share(
    #     results, scenarios, "supply_energy", ["AC", "generators"], "E", "C", "2050"
    # )
    # combined_electricity_generation_share = {
    #     **electricity_generation_share_2030,
    #     **electricity_generation_share_2050,
    # }
    # export_results(
    #     combined_electricity_generation_share,
    #     "electricity_generation_share.csv",
    #     include_share=True,
    # )

    # solar_rooftop_2030 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["low voltage", "generators", "solar rooftop"],
    #     "D",
    #     "Solar rooftop generation",
    #     "2030",
    # )
    # solar_rooftop_2050 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["low voltage", "generators", "solar rooftop"],
    #     "E",
    #     "Solar rooftop generation",
    #     "2050",
    # )
    # solar_rooftop = {**solar_rooftop_2030, **solar_rooftop_2050}
    # export_results(solar_rooftop, "solar_rooftop.csv")

    # v2g_2030 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["EV battery", "links", "V2G"],
    #     "D",
    #     "V2G",
    #     "2030",
    #     multiplier=-1,
    # )
    # v2g_2050 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["EV battery", "links", "V2G"],
    #     "E",
    #     "V2G",
    #     "2050",
    #     multiplier=-1,
    # )
    # EV_charging_2030 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["EV battery", "links", "BEV charger"],
    #     "D",
    #     "EV charging",
    #     "2030",
    # )
    # EV_charging_2050 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["EV battery", "links", "BEV charger"],
    #     "E",
    #     "EV charging",
    #     "2050",
    # )
    # EV_data = {**v2g_2030, **v2g_2050, **EV_charging_2030, **EV_charging_2050}
    # export_results(EV_data, "EV_data.csv")

    # pumped_hydro_2030 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["AC", "storage_units", "PHS"],
    #     "D",
    #     "Pumped hydro storage energy",
    #     "2030",
    #     multiplier=-1,
    # )
    # pumped_hydro_2050 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["AC", "storage_units", "PHS"],
    #     "E",
    #     "Pumped hydro storage energy",
    #     "2050",
    #     multiplier=-1,
    # )
    # hydro_2030 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["AC", "storage_units", "hydro"],
    #     "D",
    #     "Hydro energy storage",
    #     "2030",
    # )
    # hydro_2050 = get_data(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["AC", "storage_units", "hydro"],
    #     "E",
    #     "Hydro energy storage",
    #     "2050",
    # )
    # storage_data = {
    #     **pumped_hydro_2030,
    #     **pumped_hydro_2050,
    #     **hydro_2030,
    #     **hydro_2050,
    # }
    # export_results(storage_data, "storage_data.csv")

    # biomass_prices_2030 = get_data(
    #     results,
    #     scenarios,
    #     "prices",
    #     ["solid biomass"],
    #     "B",
    #     "Biomass prices",
    #     "2030",
    #     value=0,
    # )
    # biomass_prices_2050 = get_data(
    #     results,
    #     scenarios,
    #     "prices",
    #     ["solid biomass"],
    #     "C",
    #     "Biomass prices",
    #     "2050",
    #     value=0,
    # )
    # combined_biomass_prices = {**biomass_prices_2030, **biomass_prices_2050}
    # export_results(combined_biomass_prices, "biomass_prices.csv")

    # low_voltage_output_share_2030 = calculate_share(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["low voltage", "n"],
    #     "D",
    #     "C",
    #     "2030",
    #     multiplier=-1,
    #     filter_positive=False,
    # )
    # low_voltage_output_share_2050 = calculate_share(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["low voltage", "n"],
    #     "E",
    #     "C",
    #     "2050",
    #     multiplier=-1,
    #     filter_positive=False,
    # )
    # combined_low_voltage_share = {
    #     **low_voltage_output_share_2030,
    #     **low_voltage_output_share_2050,
    # }
    # export_results(
    #     combined_low_voltage_share, "low_voltage_share_output.csv", include_share=True
    # )

    # low_voltage_input_share_2030 = calculate_share(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["low voltage", "n"],
    #     "D",
    #     "C",
    #     "2030",
    #     filter_positive=True,
    # )
    # low_voltage_input_share_2050 = calculate_share(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["low voltage", "n"],
    #     "E",
    #     "C",
    #     "2050",
    #     filter_positive=True,
    # )
    # combined_low_voltage_input_share = {
    #     **low_voltage_input_share_2030,
    #     **low_voltage_input_share_2050,
    # }
    # export_results(
    #     combined_low_voltage_input_share,
    #     "low_voltage_share_input.csv",
    #     include_share=True,
    # )

    # lowT_share_2030 = calculate_share(
    #     results, scenarios, "supply_energy", ["lowT industry", "links"], "D", "C", "2030"
    # )
    # lowT_share_2050 = calculate_share(
    #     results, scenarios, "supply_energy", ["lowT industry", "links"], "E", "C", "2050"
    # )
    # combined_lowT_share = {**lowT_share_2030, **lowT_share_2050}
    # export_results(combined_lowT_share, "lowT_share.csv", include_share=True)

    # mediumT_share_2030 = calculate_share(
    #     results, scenarios, "supply_energy", ["mediumT industry", "links"], "D", "C", "2030"
    # )
    # mediumT_share_2050 = calculate_share(
    #     results, scenarios, "supply_energy", ["mediumT industry", "links"], "E", "C", "2050"
    # )
    # combined_mediumT_share = {**mediumT_share_2030, **mediumT_share_2050}
    # export_results(combined_mediumT_share, "mediumT_share.csv", include_share=True)

    # highT_share_2030 = calculate_share(
    #     results, scenarios, "supply_energy", ["highT industry", "links"], "D", "C", "2030"
    # )
    # highT_share_2050 = calculate_share(
    #     results, scenarios, "supply_energy", ["highT industry", "links"], "E", "C", "2050"
    # )
    # combined_highT_share = {**highT_share_2030, **highT_share_2050}
    # export_results(combined_highT_share, "highT_share.csv", include_share=True)

    # rural_heating_share_2030 = calculate_share(
    #     results, scenarios, "supply_energy", ["rural heat", "links"], "D", "C", "2030"
    # )
    # rural_heating_share_2050 = calculate_share(
    #     results, scenarios, "supply_energy", ["rural heat", "links"], "E", "C", "2050"
    # )
    # combined_rural_heating_share = {
    #     **rural_heating_share_2030,
    #     **rural_heating_share_2050,
    # }
    # export_results(
    #     combined_rural_heating_share, "rural_heating_share.csv", include_share=True
    # )

    # urban_decentral_heat_share_2030 = calculate_share(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["urban decentral heat", "links"],
    #     "D",
    #     "C",
    #     "2030",
    # )
    # urban_decentral_heat_share_2050 = calculate_share(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["urban decentral heat", "links"],
    #     "E",
    #     "C",
    #     "2050",
    # )
    # combined_urban_decentral_heat_share = {
    #     **urban_decentral_heat_share_2030,
    #     **urban_decentral_heat_share_2050,
    # }
    # export_results(
    #     combined_urban_decentral_heat_share,
    #     "urban_decentral_heat_share.csv",
    #     include_share=True,
    # )

    # urban_central_heat_share_2030 = calculate_share(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["urban central heat", "links"],
    #     "D",
    #     "C",
    #     "2030",
    # )
    # urban_central_heat_share_2050 = calculate_share(
    #     results,
    #     scenarios,
    #     "supply_energy",
    #     ["urban central heat", "links"],
    #     "E",
    #     "C",
    #     "2050",
    # )
    # combined_urban_central_heat_share = {
    #     **urban_central_heat_share_2030,
    #     **urban_central_heat_share_2050,
    # }
    # export_results(
    #     combined_urban_central_heat_share,
    #     "urban_central_heat_share.csv",
    #     include_share=True,
    # )


if __name__ == "__main__":
    main()

# Other areas of the energy sector that are significantly affected? (wind and solar, fossil fuels, CCS)
