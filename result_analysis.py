# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import os
import string
from math import isfinite

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


def _load_country_transport_costs_EUR_per_MWh():
    """
    Load country-specific biomass transport cost rates in EUR/MWh/km.
    """
    sc1_path = "data/biomass_transport_costs_supplychain1.csv"
    sc2_path = "data/biomass_transport_costs_supplychain2.csv"
    if not (os.path.isfile(sc1_path) and os.path.isfile(sc2_path)):
        return None

    sc1 = pd.read_csv(sc1_path, index_col=0, skiprows=2)
    sc2 = pd.read_csv(sc2_path, index_col=0, skiprows=2)
    transport_costs = pd.concat([sc1["EUR/km/ton"], sc2["EUR/km/ton"]], axis=1).mean(
        axis=1
    )
    transport_costs /= 4.8  # MWh/t conversion
    transport_costs = transport_costs.rename(index={"UK": "GB", "EL": "GR"})
    if "SE" in transport_costs.index and "NO" not in transport_costs.index:
        transport_costs.loc["NO"] = transport_costs.loc["SE"]
    return transport_costs


def _collect_biomass_transport_cost_rows(
    results_dir,
    scenarios,
    weighting,
    average_distance_km=200.0,
):
    """
    Collect scenario/carrier weighted-average biomass transport adders [EUR/MWh].
    """
    if weighting not in {"use", "potential"}:
        raise ValueError("weighting must be 'use' or 'potential'")

    transport_costs = _load_country_transport_costs_EUR_per_MWh()
    if transport_costs is None:
        print(
            "Warning: Missing biomass transport supply-chain input files. "
            "Skipping biomass transport-cost exports."
        )
        return []

    biomass_carriers = sorted(get_emission_factors(add_imported_biomass=True).keys())
    no_transport_carriers = {"manure", "sludge", "solid biomass import"}

    if scenarios == "all":
        scenario_folders = [
            folder
            for folder in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, folder))
        ]
    else:
        scenario_folders = [
            folder
            for folder in scenarios
            if os.path.isdir(os.path.join(results_dir, folder))
        ]

    rows = []
    for scenario in scenario_folders:
        if weighting == "use":
            source_path = os.path.join(results_dir, scenario, "csvs", "nodal_energy_balance.csv")
            if not os.path.isfile(source_path):
                print(
                    f"Warning: Missing nodal_energy_balance.csv for '{scenario}'. "
                    "Skipping scenario in use-weighted transport export."
                )
                continue
            source_df = pd.read_csv(source_path, skiprows=3)
            required_columns = {"component", "carrier", "location", "bus_carrier"}
            if not required_columns.issubset(set(source_df.columns)):
                print(
                    f"Warning: nodal_energy_balance for '{scenario}' missing required columns "
                    f"{sorted(required_columns)}. Skipping scenario in use-weighted transport export."
                )
                continue
            value_column = source_df.columns[-1]
            source_df["value"] = pd.to_numeric(source_df[value_column], errors="coerce").fillna(0.0)
            weighted_rows = source_df[
                (source_df["component"] == "Link")
                & (source_df["carrier"].isin(biomass_carriers))
                & (source_df["bus_carrier"] == source_df["carrier"])
            ].copy()
            weighted_rows["weight_mwh"] = (-weighted_rows["value"]).clip(lower=0.0)
        else:
            source_path = os.path.join(results_dir, scenario, "csvs", "nodal_capacities.csv")
            if not os.path.isfile(source_path):
                print(
                    f"Warning: Missing nodal_capacities.csv for '{scenario}'. "
                    "Skipping scenario in potential-weighted transport export."
                )
                continue
            source_df = pd.read_csv(source_path, skiprows=3)
            required_columns = {"component", "carrier", "location"}
            if not required_columns.issubset(set(source_df.columns)):
                print(
                    f"Warning: nodal_capacities for '{scenario}' missing required columns "
                    f"{sorted(required_columns)}. Skipping scenario in potential-weighted transport export."
                )
                continue
            value_column = source_df.columns[-1]
            source_df["value"] = pd.to_numeric(source_df[value_column], errors="coerce").fillna(0.0)
            weighted_rows = source_df[
                (source_df["component"] == "Store")
                & (source_df["carrier"].isin(biomass_carriers))
            ].copy()
            weighted_rows["weight_mwh"] = weighted_rows["value"].clip(lower=0.0)

        weighted_rows["country"] = weighted_rows["location"].astype(str).str[:2]
        weighted_rows["transport_adder"] = weighted_rows["country"].map(transport_costs).fillna(0.0)
        weighted_rows["transport_adder"] *= average_distance_km
        weighted_rows.loc[
            weighted_rows["carrier"].isin(no_transport_carriers), "transport_adder"
        ] = 0.0

        for carrier in biomass_carriers:
            carrier_rows = weighted_rows[weighted_rows["carrier"] == carrier]
            total_weight_mwh = float(carrier_rows["weight_mwh"].sum())
            if total_weight_mwh > 0:
                avg_transport_cost = float(
                    (carrier_rows["weight_mwh"] * carrier_rows["transport_adder"]).sum()
                    / total_weight_mwh
                )
            else:
                avg_transport_cost = 0.0

            rows.append(
                {
                    "scenario": scenario,
                    "carrier": carrier,
                    "weight_basis": weighting,
                    "weight_TWh": total_weight_mwh * 1e-6,
                    "avg_transport_cost_EUR_per_MWh": avg_transport_cost,
                }
            )

    return rows


def _export_transport_rows(rows, export_dir, filename):
    if not rows:
        return False
    os.makedirs(export_dir, exist_ok=True)
    output_path = os.path.join(export_dir, filename)
    pd.DataFrame(rows).sort_values(["scenario", "carrier"]).to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")
    return True


def export_biomass_avg_transport_cost_by_type(
    results_dir,
    scenarios,
    export_dir="export",
    average_distance_km=200.0,
):
    """
    Export both use-weighted and potential-weighted biomass transport adders [EUR/MWh].

    Files written:
    - biomass_avg_transport_cost_by_type_use_weighted.csv
    - biomass_avg_transport_cost_by_type_potential_weighted.csv
    - biomass_avg_transport_cost_by_type.csv (legacy alias, potential-weighted)
    """
    try:
        use_rows = _collect_biomass_transport_cost_rows(
            results_dir=results_dir,
            scenarios=scenarios,
            weighting="use",
            average_distance_km=average_distance_km,
        )
        potential_rows = _collect_biomass_transport_cost_rows(
            results_dir=results_dir,
            scenarios=scenarios,
            weighting="potential",
            average_distance_km=average_distance_km,
        )
    except Exception as exc:
        print(
            "Warning: Failed to build biomass transport-cost outputs. "
            f"Skipping export. ({exc})"
        )
        return

    wrote_use = _export_transport_rows(
        use_rows,
        export_dir,
        "biomass_avg_transport_cost_by_type_use_weighted.csv",
    )
    wrote_potential = _export_transport_rows(
        potential_rows,
        export_dir,
        "biomass_avg_transport_cost_by_type_potential_weighted.csv",
    )

    if wrote_potential:
        _export_transport_rows(
            potential_rows,
            export_dir,
            "biomass_avg_transport_cost_by_type.csv",
        )
    elif not wrote_use:
        print(
            "Warning: No biomass transport-cost rows produced. "
            "Skipped biomass transport-cost exports."
        )


def calculate_renewable_lcoe(
    results,
    scenarios,
    year="2050",
    carriers=("solar", "solar-hsat", "onwind"),
):
    """
    Calculate model-implied renewable LCOE [EUR/MWh] from summary outputs.

    Uses:
    - costs.csv: capital + marginal costs by (component, carrier)
    - energy.csv: generated energy by (component, carrier)
    """
    rows = []
    if scenarios == "all":
        scenario_list = list(results.keys())
    else:
        scenario_list = list(scenarios)

    for scenario in scenario_list:
        scenario_data = results.get(scenario, {})
        costs_df = scenario_data.get("costs")
        energy_df = scenario_data.get("energy")

        if costs_df is None or energy_df is None:
            print(
                f"Warning: Missing costs/energy for '{scenario}'. "
                "Skipping renewable LCOE export for this scenario."
            )
            continue
        if not {"A", "B", "C", "D"}.issubset(set(costs_df.columns)):
            print(
                f"Warning: Unexpected costs format for '{scenario}'. "
                "Skipping renewable LCOE export for this scenario."
            )
            continue
        if not {"A", "B", "C"}.issubset(set(energy_df.columns)):
            print(
                f"Warning: Unexpected energy format for '{scenario}'. "
                "Skipping renewable LCOE export for this scenario."
            )
            continue

        costs_df = costs_df.copy()
        energy_df = energy_df.copy()
        costs_df["D"] = pd.to_numeric(costs_df["D"], errors="coerce").fillna(0.0)
        energy_df["C"] = pd.to_numeric(energy_df["C"], errors="coerce").fillna(0.0)

        generator_costs = costs_df[costs_df["B"] == "Generator"]
        capital = generator_costs[generator_costs["A"] == "capital"].set_index("C")["D"]
        marginal = generator_costs[generator_costs["A"] == "marginal"].set_index("C")["D"]
        generation = energy_df[energy_df["A"] == "Generator"].set_index("B")["C"]

        for carrier in carriers:
            generation_mwh = float(generation.get(carrier, 0.0))
            if generation_mwh <= 0:
                continue
            capital_eur = float(capital.get(carrier, 0.0))
            marginal_eur = float(marginal.get(carrier, 0.0))
            total_eur = capital_eur + marginal_eur
            lcoe = total_eur / generation_mwh
            rows.append(
                {
                    "Folder": scenario,
                    "Year": str(year),
                    "Data Name": carrier,
                    "Generation [MWh]": generation_mwh,
                    "Capital costs [EUR]": capital_eur,
                    "Marginal costs [EUR]": marginal_eur,
                    "Total costs [EUR]": total_eur,
                    "LCOE [EUR/MWh]": lcoe,
                }
            )
    return rows


def export_renewable_lcoe(
    results,
    scenarios,
    export_dir="export",
    year="2050",
):
    rows = calculate_renewable_lcoe(results, scenarios, year=year)
    if not rows:
        print("Warning: No renewable LCOE rows produced.")
        return
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, "renewable_lcoe.csv")
    (
        pd.DataFrame(rows)
        .sort_values(["Folder", "Data Name"])
        .to_csv(file_path, index=False)
    )
    print(f"Results exported to {file_path}")

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
        # Filter to only include directories, not files like .DS_Store
        folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
    else:
        # Check that the specified folders exist and are directories
        folders = [f for f in folders if f in os.listdir(results_dir) and os.path.isdir(os.path.join(results_dir, f))]
    for folder in folders:
        folder_path = os.path.join(results_dir, folder, "csvs")
        
        # Check if the csvs directory exists
        if not os.path.exists(folder_path):
            print(f"Warning: No 'csvs' directory found in {folder}, skipping...")
            continue
            
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        dataframes = {}
        print(f"Loading data from {folder}...")
        for file in csv_files:
            try:
                file_path = os.path.join(folder_path, file)
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
    df1 = results[scenario1][dataframe].copy()
    df2 = results[scenario2][dataframe].copy()

    merged_df1 = pd.DataFrame()
    merged_df2 = pd.DataFrame()
    remaining_data = {}

    for df_source, scenario in [(df1, scenario1), (df2, scenario2)]:
        df = df_source.reset_index(drop=True).copy()
        merged_df = pd.DataFrame()

        # Remove data where all fields in a remove-condition match.
        for remove in remove_list:
            condition = pd.Series([True] * len(df), index=df.index)
            for field in remove:
                condition &= ~df.iloc[:, 0].astype(str).str.contains(
                    field, case=False, na=False
                )
            df = df[condition].reset_index(drop=True)

        for merge_conditions, new_name, is_cc in merge_list:
            if df.empty:
                break

            # Build a fresh OR-mask for this merge group only.
            condition = pd.Series([False] * len(df), index=df.index)
            for merge_condition in merge_conditions:
                field_condition = pd.Series([True] * len(df), index=df.index)
                for i, field in enumerate(merge_condition):
                    field_condition &= df.iloc[:, i].astype(str).str.contains(
                        field, case=True, na=False
                    )
                condition |= field_condition

            added_data = df[condition].copy()

            # CC filtering inside the matched subset.
            if is_cc is True:
                added_data = added_data[
                    added_data[data_name_columns[-1]]
                    .astype(str)
                    .str.contains("CC", case=True, na=False)
                ]
            elif is_cc is False:
                added_data = added_data[
                    ~added_data[data_name_columns[-1]]
                    .astype(str)
                    .str.contains("CC", case=True, na=False)
                ]

            if added_data.empty:
                continue

            # Create merged row using the sum of matched values.
            new_row = {
                "folder": scenario,
                "year": year,
                "data_name": new_name,
                "values": added_data[value_column].sum() * multiplier,
            }
            merged_df = pd.concat([merged_df, pd.DataFrame([new_row])], ignore_index=True)

            # Remove only the rows actually used for this merge.
            df = df.drop(index=added_data.index).reset_index(drop=True)

        if scenario == scenario1:
            merged_df1 = merged_df
        else:
            merged_df2 = merged_df
        remaining_data[scenario] = df

    # Add remaining data, data_name will be a combination of the data_name_columns
    for scenario, merged_df in [(scenario1, merged_df1), (scenario2, merged_df2)]:
        df = remaining_data.get(scenario, pd.DataFrame())
        for _, row in df.iterrows():
            key = "_".join([str(row[column]) for column in data_name_columns])
            key = key.replace(" ", "_")
            new_row = {
                "year": year,
                "data_name": key,
                "values": row[value_column] * multiplier,
            }
            merged_df = pd.concat([merged_df, pd.DataFrame([new_row])], ignore_index=True)
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
    convert_fields=None
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

    def _safe_share(value, total):
        """
        Return a finite share value if possible, else NaN.
        Keeps original share logic for normal numeric inputs and avoids
        runtime warnings for cases like inf/inf or division by zero.
        """
        if pd.isna(value) or pd.isna(total):
            return float("nan")
        if not isfinite(value) or not isfinite(total) or total == 0:
            return float("nan")
        return value / total

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
                #make all convert fields absolute values before deleting all non negatives
                if convert_fields is not None:
                    for field in convert_fields:
                        # Check if any rows contain the field in the data_name_column
                        mask = data[data_name_column].str.contains(field, case=False, na=False)
                        if mask.any():
                            # Convert the value_column to absolute values for matching rows
                            data.loc[mask, value_column] = abs(data.loc[mask, value_column])
                data = data[data[value_column] > 0]
            else:
                data = data[data[value_column] < 0]

            # Calculate total sum for share calculation
            total_sum = data[value_column].sum()

            # Merge the values using the merge_fields
            for merge_conditions, new_name in merge_fields:
                merged_data = pd.DataFrame()
                for merge_condition in merge_conditions:
                    if merge_condition == "":  # If the merge condition is an empty string, select all data
                        added_data = data
                    else:
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
                        share = _safe_share(merged_data[value_column].sum(), total_sum)
                        new_row["share"] = round(share, 4) if pd.notna(share) else share
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
                        share = _safe_share(row[value_column], total_sum)
                        new_row["share"] = round(share, 4) if pd.notna(share) else share
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


# -----------------------------
# NEW FUNCTION: get_final_bioenergy_supply
# -----------------------------
def get_final_bioenergy_supply(
    results,
    scenarios,
    year="2050",
    include_electricity=True,
    include_waste=False,
    waste_biogenic_share=1,
    allocate_hybrids=True,
    aggregate=True,
    round_digits=1,
):
    """
    Calculate *final energy supply from bioenergy* (delivered energy basis), with
    an **optional** inclusion of municipal solid waste (MSW) based CHP scaled by
    a user-defined **biogenic share**.

    What is counted:
    - Positive output flows from **bio-derived conversion links** to **final-energy buses**
      (including **bio-hydrogen**) (fuels and end-use heat). Electricity from biomass CHP can be optionally
      included.
    - If `include_waste` is True, MSW CHP outputs are included and scaled by
      `waste_biogenic_share` (e.g. 0.5 → 50% biogenic).
    - **Hybrid attribution (optional):** when `allocate_hybrids` is True, outputs from hybrid links (currently: "electrobiofuels") are attributed to biomass proportionally to energy inputs (solid biomass vs H2). CO2 streams are ignored as they carry no energy.

    Avoids double counting by summing **only the final buses reached by the bio/waste links**
    (e.g. oil/gas/methanol/heat), not intermediate carriers.

    Parameters
    ----------
    results : dict
        Output of `load_results()`.
    scenarios : list[str]
        Scenario folder names to include.
    year : str, optional
        Year filter passed to `get_data` (default "2050").
    include_electricity : bool, optional
        If True, include electricity from solid-biomass CHP (and waste CHP if enabled).
        Default False.
    include_waste : bool, optional
        If True, include *biogenic share* of waste CHP outputs.
    waste_biogenic_share : float, optional
        Fraction (0–1) of MSW assumed **biogenic**. Applied to heat/electricity
        from `waste CHP`/`waste CHP CC`. Default 0.5.
    aggregate : bool, optional
        If True, return aggregated categories (bio-liquids, bio-methane,
        bio-methanol, bio-heat industry, bio-heat buildings, optional
        bio-electricity, optional waste-heat buildings, waste-electricity).
        If False, return each contributing link as its own row. Default True.
    round_digits : int, optional
        Rounding for values. Default 1.

    Returns
    -------
    dict
        Dictionary keyed like other extractors, with fields
        {folder, year, data_name, values, (optional share)}.
    """
    # -----------------------------
    # Define BIO routes (final outputs only)
    # -----------------------------
    bio_fields = [
        # Bio-liquids (oil products)
        ["Link", "biomass to liquid", "oil"],
        ["Link", "biomass to liquid CC", "oil"],
        ["Link", "electrobiofuels", "oil"],        
        # Bio-hydrogen (direct biomass → H2)
        ["Link", "solid biomass to hydrogen", "H2"],
        # Bio-methanol (as an energy carrier to end-uses)
        ["Link", "biomass-to-methanol", "methanol"],
        ["Link", "biomass-to-methanol CC", "methanol"],
        # Bio-methane (gas delivered to end-uses)
        ["Link", "biogas to gas", "gas"],
        ["Link", "biogas to gas CC", "gas"],
        ["Link", "BioSNG", "gas"],
        ["Link", "BioSNG CC", "gas"],
        # Direct bioheat for industry (final heat vectors)
        ["Link", "solid biomass for mediumT industry", "mediumT industry"],
        ["Link", "solid biomass for mediumT industry CC", "mediumT industry"],
        ["Link", "lowT industry solid biomass", "lowT industry"],
        ["Link", "lowT industry solid biomass CC", "lowT industry"],
        # Naming variants seen in some runs
        ["Link", "solid biomass for lowT industry", "lowT industry"],
        ["Link", "solid biomass for lowT industry CC", "lowT industry"],
        ["Link", "solid biomass for industry", "mediumT industry"],
        ["Link", "solid biomass for industry CC", "mediumT industry"],
        ["Link", "solid biomass for industry", "highT industry"],
        ["Link", "solid biomass for industry CC", "highT industry"],
        # Direct bioheat for buildings/districts
        ["Link", "rural biomass boiler", "rural heat"],
        ["Link", "urban decentral biomass boiler", "urban decentral heat"],
        ["Link", "urban central biomass boiler", "urban central heat"],
        ["Link", "urban central solid biomass CHP", "urban central heat"],
        ["Link", "urban central solid biomass CHP CC", "urban central heat"],
    ]

    # Optionally add bio-electricity from solid-biomass CHP / bioliquids
    bio_elec_fields = []
    if include_electricity:
        bio_elec_fields = [
            ["Link", "urban central solid biomass CHP", "AC"],
            ["Link", "urban central solid biomass CHP CC", "AC"],
            ["Generator", "bioliquids", "AC"],  # if present in some runs
        ]

    # Aggregation (categories) for BIO
    bio_merge = []
    if aggregate:
        bio_merge = [
            [["biomass to liquid", "biomass to liquid CC", "electrobiofuels"], "bio-liquids"],
            [["biogas to gas", "biogas to gas CC", "BioSNG", "BioSNG CC"], "bio-methane"],
            [["biomass-to-methanol", "biomass-to-methanol CC"], "bio-methanol"],
            [[
                "solid biomass for mediumT industry",
                "solid biomass for mediumT industry CC",
                "lowT industry solid biomass",
                "lowT industry solid biomass CC",
                "solid biomass for lowT industry",
                "solid biomass for lowT industry CC",
                "solid biomass for industry",
                "solid biomass for industry CC",
            ], "bio-heat industry"],
            [[
                "rural biomass boiler",
                "urban decentral biomass boiler",
                "urban central biomass boiler",
                "urban central solid biomass CHP",
                "urban central solid biomass CHP CC",
            ], "bio-heat buildings"],
            [["solid biomass to hydrogen"], "bio-hydrogen"],
        ]
        if include_electricity:
            bio_merge.append([["urban central solid biomass CHP", "urban central solid biomass CHP CC", "bioliquids"], "bio-electricity"])

    # Extract BIO (no shares yet — we'll recompute after optional waste scaling)
    bio = get_data(
        results,
        scenarios,
        "energy_balance",
        bio_fields + bio_elec_fields,
        bio_merge,
        "D",
        "B",
        year,
        filter_positive=True,
        remove_list=["biomass transport"],
        calculate_share=False,
        round_digits=round_digits,
    )

    # -----------------------------
    # Hybrid attribution: split electrobiofuels by biomass vs H2 energy input
    # -----------------------------
    if allocate_hybrids:
        # Get inputs (negative flows) for electrobiofuels: H2 and solid biomass
        eb_in_h2 = get_data(
            results, scenarios, "energy_balance",
            [["Link", "electrobiofuels", "H2"]], [], "D", "B", year,
            filter_positive=False, calculate_share=False, round_digits=round_digits,
        )
        eb_in_bio = get_data(
            results, scenarios, "energy_balance",
            [["Link", "electrobiofuels", "solid biomass"]], [], "D", "B", year,
            filter_positive=False, calculate_share=False, round_digits=round_digits,
        )
        # Get output (positive flow) to oil
        eb_oil = get_data(
            results, scenarios, "energy_balance",
            [["Link", "electrobiofuels", "oil"]], [], "D", "B", year,
            filter_positive=True, calculate_share=False, round_digits=round_digits,
        )

        # Build folder-wise biomass shares and adjust
        folders = set([c["folder"] for c in eb_oil.values()]) if eb_oil else set()
        for f in folders:
            h2_in = abs(next((v["values"] for v in eb_in_h2.values() if v["folder"] == f), 0.0))
            bio_in = abs(next((v["values"] for v in eb_in_bio.values() if v["folder"] == f), 0.0))
            denom = h2_in + bio_in
            bio_share = (bio_in / denom) if denom > 0 else 1.0

            oil_out = next((v["values"] for v in eb_oil.values() if v["folder"] == f), 0.0)
            biomass_attributed_oil = round(oil_out * bio_share, round_digits)
            non_bio_part = round(oil_out - biomass_attributed_oil, round_digits)

            if aggregate:
                # Reduce the aggregated bio-liquids bucket by the non-biomass share
                # Find the entry for this folder
                for k, r in bio.items():
                    if r["folder"] == f and r["data_name"] == "bio-liquids":
                        r["values"] = round(r["values"] - non_bio_part, round_digits)
                        bio[k] = r
                        break
            else:
                # Directly set the electrobiofuels→oil row to the biomass-attributed amount
                for k, r in bio.items():
                    if r["folder"] == f and r["data_name"] == "electrobiofuels":
                        r["values"] = biomass_attributed_oil
                        bio[k] = r
                        break

    # -----------------------------
    # Define WASTE routes (optional) and extract
    # -----------------------------
    waste = {}
    if include_waste:
        waste_heat_fields = [
            ["Link", "waste CHP", "urban central heat"],
            ["Link", "waste CHP CC", "urban central heat"],
        ]
        waste_elec_fields = []
        if include_electricity:
            waste_elec_fields = [
                ["Link", "waste CHP", "AC"],
                ["Link", "waste CHP CC", "AC"],
            ]

        waste_merge = []
        if aggregate:
            # Keep waste heat separate from bio-heat to avoid mixing categories
            waste_merge = [
                [["waste CHP", "waste CHP CC"], "waste-heat buildings"],
            ]
            if include_electricity:
                # electricity is a separate category
                waste_merge.append([["waste CHP", "waste CHP CC"], "waste-electricity"])

        # Get waste HEAT (and optional ELEC) first, then scale by biogenic share
        waste = get_data(
            results,
            scenarios,
            "energy_balance",
            waste_heat_fields + waste_elec_fields,
            waste_merge,
            "D",
            "B",
            year,
            filter_positive=True,
            calculate_share=False,
            round_digits=round_digits,
        )

        # Scale waste outputs by the biogenic share BEFORE combining and computing shares
        for key, content in list(waste.items()):
            content["values"] = round(content["values"] * waste_biogenic_share, round_digits)
            waste[key] = content

    # -----------------------------
    # Combine and compute shares across all included categories
    # -----------------------------
    combined = {}
    combined.update(bio)
    combined.update(waste)

    # Compute per-folder shares now (uses the helper already present in this file)
    combined = calculate_share(combined) if aggregate else combined

    # Append a total per folder entry
    totals = {}
    for key, content in combined.items():
        folder = content["folder"]
        totals.setdefault(folder, 0)
        totals[folder] += content["values"]

    for folder, total in totals.items():
        total_key = f"{folder}_{year}_bioenergy_final_supply_total"
        total_row = {
            "folder": folder,
            "year": year,
            "data_name": "bioenergy final supply (total)",
            "values": round(total, round_digits),
        }
        # If shares exist, set to 1.0 for the total rows
        if any("share" in v for v in combined.values()):
            total_row["share"] = 1.0
        combined[total_key] = total_row

    return combined


def add_costs(data, shadow_prices):
    costs = {  # Euro/MWh_LHV
        "agricultural waste": 11.32275524454902,
        "fuelwood residues": 13.533604337722517,
        "fuelwoodRW": 11.121582112826298,
        "manure": 19.440634202419798,
        "residues from landscape care": 9.238953786361055,
        "secondary forestry residues": 7.198446278808251,
        "coal": 9.5542,
        "fuelwood": 14.5224,
        "gas": 24.568,
        "oil primary": 52.9111,
        "woody crops": 39.1074178603587,
        "grasses": 16.703166765916077,
        "sludge": 19.42966385933722,
        "solid biomass import": 54,
        "sawdust": 5.696405201022603,
        "C&P_RW": 22.389579273883896,
    }
    emission_factors = get_emission_factors(new_names=False, add_imported_biomass=True)
    shadow_price_dict = {}
    for key, content in shadow_prices.items():
        shadow_price_dict[content["folder"]] = content["values"]
    if add_costs:
        for key, content in data.items():
            if content["data_name"] in costs:
                content["costs"] = costs[content["data_name"]]
            else:
                content["costs"] = None
            if "default" not in content["folder"]:
                content["CO2 costs"] = emission_factors.get(content["data_name"],0) * shadow_price_dict[content["folder"]]
            else:
                content["CO2 costs"] = 0
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


def calc_share(
    data,
    scenarios,
    process = "gas"
):
    results = {}
    for folder in scenarios:
        values = 0
        total = 0
        for key, content in data.items():
            if content["folder"] == folder:
                if content["data_name"] == process:
                    values += content["values"]
                total += content["values"]
        share = values / total
        results[folder] = {process: values, "total": total, "share": share}
    return results


def split_CHP(data):
    electricity_share = 0.248
    heat_share = 0.752
    
    # Track entries to add/update/delete
    new_entries = {}  # Using dict instead of list for better key management
    deleted_entries = []
    
    # Track electricity and heat values by folder/year
    electricity_by_folder = {}
    heat_by_folder = {}
    
    # First pass: calculate total electricity and heat by folder/year
    for key, content in data.items():
        if "CHP" in content["data_name"]:
            folder_year = (content["folder"], content["year"])
            
            # Initialize if needed
            if folder_year not in electricity_by_folder:
                electricity_by_folder[folder_year] = {"values": 0, "share": 0}
                heat_by_folder[folder_year] = {"values": 0, "share": 0}
            
            # Add this CHP's contribution to electricity and heat
            electricity_by_folder[folder_year]["values"] += content["values"] * electricity_share
            electricity_by_folder[folder_year]["share"] += content.get("share", 0) * electricity_share
            
            heat_by_folder[folder_year]["values"] += content["values"] * heat_share
            heat_by_folder[folder_year]["share"] += content.get("share", 0) * heat_share
            
            # Mark for deletion
            deleted_entries.append(key)
    
    # Second pass: find existing electricity/heat entries to update
    existing_elec = {}
    existing_heat = {}
    
    for key, content in data.items():
        folder_year = (content["folder"], content["year"])
        if content["data_name"] == "electricity production":
            existing_elec[folder_year] = key
        elif content["data_name"] == "heat production":
            existing_heat[folder_year] = key
    
    # Final pass: create or update entries
    for folder_year, elec_values in electricity_by_folder.items():
        folder, year = folder_year
        
        # Create/update electricity entry
        if folder_year in existing_elec:
            key = existing_elec[folder_year]
            existing_values = data[key].get("values", 0)
            existing_share = data[key].get("share", 0)
            
            new_entries[key] = {
                "folder": folder,
                "year": year,
                "data_name": "electricity production",
                "values": existing_values + elec_values["values"],
                "share": existing_share + elec_values["share"]
            }
        else:
            new_key = f"{folder}_{year}_electricity_production"
            new_entries[new_key] = {
                "folder": folder,
                "year": year,
                "data_name": "electricity production",
                "values": elec_values["values"],
                "share": elec_values["share"]
            }
    
    # Same for heat entries
    for folder_year, heat_values in heat_by_folder.items():
        folder, year = folder_year
        
        if folder_year in existing_heat:
            key = existing_heat[folder_year]
            existing_values = data[key].get("values", 0)
            existing_share = data[key].get("share", 0)
            
            new_entries[key] = {
                "folder": folder,
                "year": year,
                "data_name": "heat production",
                "values": existing_values + heat_values["values"],
                "share": existing_share + heat_values["share"]
            }
        else:
            new_key = f"{folder}_{year}_heat_production"
            new_entries[new_key] = {
                "folder": folder,
                "year": year,
                "data_name": "heat production",
                "values": heat_values["values"],
                "share": heat_values["share"]
            }
    
    # Apply changes to the data
    for key in deleted_entries:
        del data[key]
    
    for key, content in new_entries.items():
        data[key] = content
    
    # folder_sums2 = {} # for controlling output sums - that they match the total supply
    # for key, content in data.items():
    #     folder_year = (content["folder"], content["year"])
    #     if folder_year in folder_sums2:
    #         folder_sums2[folder_year] += content["values"]
    #     else:
    #         folder_sums2[folder_year] = content["values"]

    # print(f"folder_sums2: {folder_sums2}")
    
    return data

def calculate_supply_difference_and_emission_difference(
    data,capacity_data, scenario1, scenario2, year
):
    results = {}
    emission_factors = get_emission_factors(
        config_file_path="config/config.yaml",
        new_names=False,
        add_imported_biomass=True,
    )
    renewable_factors = {
        "solar": 8.4,
        "onwind": 0.95,
        "solar-hsat": 12.41
    }
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
    for key, content in capacity_data.items():
        if content["year"] == year:
            if content["folder"] == scenario1:
                key2 = key.replace(scenario1, scenario2)
                if key2 in capacity_data:
                    difference = content["values"] - capacity_data[key2]["values"]
                    new_key = key.replace(scenario1, f"{scenario1}_{scenario2}")
                    results[new_key] = {
                        "year": year,
                        "data_name": content["data_name"],
                        f"values_{scenario1}": content["values"],
                        f"values_{scenario2}": capacity_data[key2]["values"],
                        "difference": difference,
                        "emission_difference": (
                            difference
                            * renewable_factors.get(
                                content["data_name"], 0
                            )
                        ),
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


def calculate_upstream_emissions(data, scenarios, capacity_data):
    emission_factors = get_emission_factors(
        config_file_path="config/config.yaml",
        new_names=False,
        add_imported_biomass=True,
    )
    renewable_factors = {
        "solar": 8.4,
        "onwind": 0.95,
        "solar-hsat": 12.41
    }
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
    for key, content in capacity_data.items():
        for scenario in scenarios:
            if content["folder"] == scenario:
                # calculate upstream emissions based on capacity
                upstream_emissions = (
                    content["values"]
                    * renewable_factors.get(
                        content["data_name"], 0
                    ) 
                )
                results[key] = {
                    "folder": content["folder"],
                    "data_name": content["data_name"],
                    "year": content["year"],
                    "upstream emissions": upstream_emissions,
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
            ["Link", "biomass-to-methanol CC", "co2 stored"],            
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
            ["Link", "biomass to liquid CC", "co2"],
            ["Link", "biogas to gas CC", "co2"],
            ["Link", "BioSNG CC", "co2"],
            ["Link", "biomass-to-methanol", "co2"],
            ["Link", "biomass-to-methanol CC", "co2"],
        ],
        [[[""], "All utilised"]],
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
            #["Link", "BioSNG CC", "co2 stored"],
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
        "CCGT": "gas",
        "OCGT methanol": "liquid fuels",
        "SMR": "gas",
        "SMR CC": "gas",
        "agriculture machinery oil": "liquid fuels",
        "BioSNG": "gas",
        "biogas to gas": "gas",
        "biogas to gas CC": "gas",
        "biomass to liquid": "liquid fuels",
        "biomass to liquid CC": "liquid fuels",
        "biomass-to-methanol": "liquid fuels",
        "biomass-to-methanol CC": "liquid fuels",
        "electrobiofuels": "liquid fuels",
        "gas for highT industry": "gas",
        "gas for mediumT industry": "gas",
        "industry methanol": "gas",
        "kerosene for aviation": "liquid fuels",
        "lowT industry methane": "gas",
        "lowT industry methane CC": "gas",
        "methanolisation": "liquid fuels",
        "municipal solid waste": "municipal solid waste",
        "oil refining": "crude oil",
        "process emissions": "cement",
        "process emissions CC": "cement",
        "rural gas boiler": "gas",
        "shipping methanol": "liquid fuels",
        "solid biomass for mediumT industry CC": "solid biomass",
        "urban central gas CHP": "gas",
        "urban central gas boiler": "gas",
        "urban decentral gas boiler": "gas",
        "Fischer-Tropsch": "liquid fuels",
        "waste CHP CC": "municipal solid waste",
        "onwind landuse emission": "indirect emissions from renewables",
        "solar landuse emission": "indirect emissions from renewables",
        "solar-hsat landuse emission": "indirect emissions from renewables",
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
        "waste CHP": "municipal solid waste",
    }
    def _resolve_sink(data_name, key, content):
        """
        Resolve Sankey source/sink node from data_name.
        Strict mode: fail fast with explicit context for unmapped carriers.
        """
        sink = sink_dict.get(data_name)
        if sink is None:
            raise KeyError(
                "Unmapped CO2 Sankey carrier "
                f"'{data_name}' in entry '{key}' "
                f"(folder='{content.get('folder')}', "
                f"emission_type='{content.get('emission_type')}', "
                f"value={content.get('values')}). "
                "Add this carrier to sink_dict in modify_co2_data()."
            )
        return sink

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
                new_data[key]["from_sink"] = _resolve_sink(content["data_name"], key, content)
                new_data[key]["to_sink"] = content["emission_type"]
            else:
                new_data[key]["from_sink"] = content["emission_type"]
                new_data[key]["to_sink"] = _resolve_sink(content["data_name"], key, content)
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

def main(results_dir="results", export_dir="export",scenarios=["default", "carbon_costs"],difference_scenarios=["default", "carbon_costs"]):

    results = load_results(results_dir, scenarios)
    export_biomass_avg_transport_cost_by_type(
        results_dir=results_dir,
        scenarios=scenarios,
        export_dir=export_dir,
    )

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
        ["Link", "waste CHP", "non-sequestered HVC"],
        ["Link", "waste CHP CC", "non-sequestered HVC"],
        ["StorageUnit", "hydro", "AC"],
        ["Generator", "", "biogas"],
        ["Link", "", "biogas"],
        ["Generator", "", "coal"],
        ["Generator", "", "gas"],
        ["Generator", "solar rooftop", "low voltage"],
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
                "solid biomass import",
            ],
            "solid biomass",
        ],
        [["manure", "sludge"], "biogas"],
        [["onwind","offwind-ac","offwind-dc","offwind-float"], "wind"],
        [["solar","solar-hsat","solar rooftop"], "solar"],
        [["hydro", "ror"], "hydro"],
        [["waste CHP","waste CHP CC"], "municipal waste"],
    ]
    remove_list = ["biomass transport"]
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
        convert_fields=["waste CHP","waste CHP CC"],
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
    renewable_capacity = get_data(
        results,
        scenarios,
        "capacities",
        [["Generator", "solar"], ["Generator", "onwind"], ["Generator", "solar-hsat"]],
        [],
        "C",
        "B",
        "2050",
        filter_positive=True,
    )
    difference = calculate_supply_difference_and_emission_difference(
        biomass_supply,renewable_capacity, difference_scenarios[0], difference_scenarios[1], "2050"
    )
    export_results(difference, "supply_difference.csv", simply_print=True, export_dir=export_dir)
    difference2 = calculate_supply_difference_and_emission_difference(
        biomass_supply,renewable_capacity, "default_710", "cscs_710", "2050"
    )
    export_results(difference2, "supply_difference_variants.csv", simply_print=True, export_dir=export_dir)

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

    all_gas_generation = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Generator", "", "gas"], ["Link", "", "gas"]],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
        remove_list=["gas pipeline"],
    )
    gas_shares = calc_share(all_gas_generation, scenarios)

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
        [[["", "", "biomass boiler"], ["", "", "lowT industry solid biomass"]], "biomass boiler", None],
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
        difference_scenarios[0],
        difference_scenarios[1],
        "costs",
        ["A", "B", "C"],
        "D",
        "2050",
        merge_fields,
        remove_list=[],
        round_digits=0,
    )

    # Reallocate urban central gas CHP costs to natural gas / biogas boiler
    # according to scenario-specific gas shares.
    scenario1 = difference_scenarios[0]
    scenario2 = difference_scenarios[1]
    keys_to_split = [
        key
        for key, content in cost_difference.items()
        if "urban_central_gas_CHP" in str(content.get("data_name", ""))
    ]
    if keys_to_split:
        split_accumulator = {"natural gas": [0.0, 0.0], "biogas boiler": [0.0, 0.0]}

        for key in keys_to_split:
            content = cost_difference[key]
            v1 = content.get(f"values_{scenario1}", 0.0)
            v2 = content.get(f"values_{scenario2}", 0.0)
            v1 = 0.0 if pd.isna(v1) else float(v1)
            v2 = 0.0 if pd.isna(v2) else float(v2)

            ng_share_1 = float(gas_shares.get(scenario1, {}).get("share", 1.0))
            ng_share_2 = float(gas_shares.get(scenario2, {}).get("share", 1.0))
            ng_share_1 = min(max(ng_share_1, 0.0), 1.0)
            ng_share_2 = min(max(ng_share_2, 0.0), 1.0)

            ng_1 = v1 * ng_share_1
            ng_2 = v2 * ng_share_2
            bg_1 = v1 - ng_1
            bg_2 = v2 - ng_2

            split_accumulator["natural gas"][0] += ng_1
            split_accumulator["natural gas"][1] += ng_2
            split_accumulator["biogas boiler"][0] += bg_1
            split_accumulator["biogas boiler"][1] += bg_2

        for key in keys_to_split:
            del cost_difference[key]

        year = "2050"
        if cost_difference:
            year = next(iter(cost_difference.values())).get("year", year)

        for category, (val1, val2) in split_accumulator.items():
            existing_key = next(
                (
                    key
                    for key, value in cost_difference.items()
                    if value.get("data_name") == category and value.get("year") == year
                ),
                None,
            )
            if existing_key is None:
                split_key = (
                    f"{scenario1}_{scenario2}_{year}_{category.replace(' ', '_')}_split"
                )
                cost_difference[split_key] = {
                    "year": year,
                    "data_name": category,
                    f"values_{scenario1}": val1,
                    f"values_{scenario2}": val2,
                    "difference": val2 - val1,
                }
            else:
                cost_difference[existing_key][f"values_{scenario1}"] += val1
                cost_difference[existing_key][f"values_{scenario2}"] += val2
                cost_difference[existing_key]["difference"] = (
                    cost_difference[existing_key][f"values_{scenario2}"]
                    - cost_difference[existing_key][f"values_{scenario1}"]
                )

    # Merge biomass-to-methanol capital + marginal parts into one category.
    methanol_parts = {
        "capital_Link_biomass-to-methanol",
        "marginal_Link_biomass-to-methanol",
    }
    methanol_keys = [
        key
        for key, content in cost_difference.items()
        if content.get("data_name") in methanol_parts
    ]
    if methanol_keys:
        methanol_val1 = 0.0
        methanol_val2 = 0.0
        methanol_year = "2050"
        for key in methanol_keys:
            content = cost_difference[key]
            methanol_val1 += float(content.get(f"values_{scenario1}", 0.0) or 0.0)
            methanol_val2 += float(content.get(f"values_{scenario2}", 0.0) or 0.0)
            methanol_year = content.get("year", methanol_year)
            del cost_difference[key]

        existing_key = next(
            (
                key
                for key, value in cost_difference.items()
                if value.get("data_name") == "biomass-to-methanol"
                and value.get("year") == methanol_year
            ),
            None,
        )
        if existing_key is None:
            merged_key = f"{scenario1}_{scenario2}_{methanol_year}_biomass-to-methanol"
            cost_difference[merged_key] = {
                "year": methanol_year,
                "data_name": "biomass-to-methanol",
                f"values_{scenario1}": methanol_val1,
                f"values_{scenario2}": methanol_val2,
                "difference": methanol_val2 - methanol_val1,
            }
        else:
            cost_difference[existing_key][f"values_{scenario1}"] += methanol_val1
            cost_difference[existing_key][f"values_{scenario2}"] += methanol_val2
            cost_difference[existing_key]["difference"] = (
                cost_difference[existing_key][f"values_{scenario2}"]
                - cost_difference[existing_key][f"values_{scenario1}"]
            )

    # Merge all methanolisation cost types (capital, marginal, etc.) into one category.
    methanolisation_keys = [
        key
        for key, content in cost_difference.items()
        if "methanolisation" in str(content.get("data_name", ""))
    ]
    if methanolisation_keys:
        methanolisation_val1 = 0.0
        methanolisation_val2 = 0.0
        methanolisation_year = "2050"
        for key in methanolisation_keys:
            content = cost_difference[key]
            methanolisation_val1 += float(content.get(f"values_{scenario1}", 0.0) or 0.0)
            methanolisation_val2 += float(content.get(f"values_{scenario2}", 0.0) or 0.0)
            methanolisation_year = content.get("year", methanolisation_year)
            del cost_difference[key]

        merged_key = f"{scenario1}_{scenario2}_{methanolisation_year}_methanolisation"
        cost_difference[merged_key] = {
            "year": methanolisation_year,
            "data_name": "methanolisation",
            f"values_{scenario1}": methanolisation_val1,
            f"values_{scenario2}": methanolisation_val2,
            "difference": methanolisation_val2 - methanolisation_val1,
        }

    export_results(cost_difference, "cost_difference.csv", include_difference=True, export_dir=export_dir, scenario1=difference_scenarios[0], scenario2=difference_scenarios[1])

    hydrogen_production = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "", "H2"]],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
        remove_list=["pipeline"],
    )
    export_results(hydrogen_production, "hydrogen_production.csv", export_dir=export_dir)

    # Final energy *supply* from bioenergy (delivered energy basis)
    final_bioenergy = get_final_bioenergy_supply(
        results,
        scenarios,
        year="2050",
        include_electricity=True, 
        aggregate=True,
    )
    export_results(final_bioenergy, "final_bioenergy_supply.csv", include_share=True, export_dir=export_dir)

    heat_pumps = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "rural air heat pump", "low voltage"],         
        ["Link", "rural ground heat pump", "low voltage"],
        ["Link", "urban central air heat pump", "low voltage"],
        ["Link", "urban central ptes heat pump", "low voltage"],
        ["Link", "urban decentral air heat pump", "low voltage"],
        ],
        [],
        "D",
        "B",
        "2050",
        filter_positive=False,
        multiplier=-1,
    )
    export_results(heat_pumps, "heat_pumps.csv", export_dir=export_dir)

    gas_use = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "", "gas"]],
        [
            [["biogas to gas"], "biogas upgrading with CC"],
            [["biogas to gas CC"], "biogas upgrading without CC"],
            [["SMR CC","gas for industry CC","gas for lowT industry CC","gas for mediumT industry CC","gas for highT industry CC","urban central gas CHP CC"], "gas use with CC"],
            [["urban decentral gas boiler","OCGT","SMR","gas for industry","gas for lowT industry","gas for mediumT industry","gas for highT industry", "rural gas boiler","urban central gas CHP","urban central gas boiler"], "gas use without CC"],
        ],
        "D",
        "B",
        "2050",
        filter_positive=False,
        multiplier=-1,
        remove_list=["gas pipeline"],
    )
    export_results(gas_use, "gas_use.csv", export_dir=export_dir)

    shadow_price = get_data(
        results,
        scenarios,
        "metrics",
        [["co2_shadow"]],
        [[["co2_shadow"], "CO2 shadow price"]],
        "B",
        "A",
        "2050",
        multiplier=-1,
        filter_positive=False,
    )
    export_results(shadow_price, "shadow_price.csv", export_dir=export_dir)

    weighted_prices = get_data(
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
        filter_positive=None,
        remove_list=["agriculture machinery oil"],
    )
    weighted_prices = add_costs(weighted_prices, shadow_price)
    export_results(weighted_prices, "weighted_prices.csv",export_dir=export_dir, simply_print=True)
    export_renewable_lcoe(results, scenarios, export_dir=export_dir, year="2050")

    solid_biomass_supply = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "", "solid biomass"]],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
        remove_list=["biomass transport"],
    )
    digestable_biomass_supply = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "","biogas"]],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
    )
    co2_use = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "","co2 stored"]],
        [],
        "D",
        "B",
        "2050",
        filter_positive=False,
        multiplier=-1,
    )
    export_results(co2_use, "co2_use.csv", include_share=True, export_dir=export_dir)

    co2_capture = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "","co2 stored"]],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
    )
    export_results(co2_capture, "co2_capture.csv", include_share=True, export_dir=export_dir)

    sabatier_shares = calc_share(all_gas_generation, scenarios,"Sabatier")
    share_sequestered = calculate_share_of_sequestration(co2_use, scenarios)

    carbon_utilisation = calc_beccus(
        results,
        scenarios,
        solid_biomass_supply,
        digestable_biomass_supply,
        gas_shares,
        share_sequestered,
    )  
    export_carbon_removal(carbon_utilisation, "CCUS.csv", export_dir=export_dir)

    all_biomass_supply = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "", "solid biomass"], ["Link", "", "biogas"]],
        [
            [[""], "biomass"],
        ],
        "D",
        "B",
        "2050",
        filter_positive=True,
        remove_list=["biomass transport", "solid biomass for industry","solid biomass for industry CC"],
    )
    export_results(all_biomass_supply, "all_biomass_supply.csv", export_dir=export_dir)

    merge_fields = [
        [["biomass to liquid","biomass to liquid CC", "electrobiofuels","biomass-to-methanol","biomass-to-methanol CC"], "conversion to liquid fuels"],
        [["solid biomass for industry","solid biomass for lowT industry","solid biomass for mediumT industry", "solid biomass for industry CC","solid biomass for lowT industry CC","solid biomass for mediumT industry CC","rural biomass boiler","urban decentral biomass boiler","urban central biomass boiler",
          "gas for lowT industry","gas for mediumT industry","gas for highT industry","gas for industry","gas for lowT industry CC","gas for mediumT industry CC","gas for highT industry CC","gas for industry CC","rural gas boiler","urban decentral gas boiler","urban central gas boiler","lowT industry solid biomass","lowT industry solid biomass CC",
          "lowT industry methane","lowT industry methane CC",
          ], "heat production"],
        [["urban central solid biomass CHP","urban central solid biomass CHP CC"], "CHP"],
        [["solid biomass to hydrogen","solid biomass to hydrogen CC","SMR","SMR CC"], "hydrogen production"],
        [["OCGT", "CCGT"], "electricity production"],
    ]
    solid_biomass_use_by_sector = (
        get_data(  # this doesn't account for the gas share yet
            results,
            scenarios,
            "energy_balance",
            [["Link", "", "solid biomass"]],
            merge_fields,
            "D",
            "B",
            "2050",
            filter_positive=False,
            multiplier=-1,
            calculate_share=False,
            remove_list=["transport", "import", "pipeline", "biogas to gas", "BioSNG","BioSNG CC"],
        )
    )
    digestable_biomass_use_by_sector = (
        get_data(  # this doesn't account for the gas share yet
            results,
            scenarios,
            "energy_balance",
            [["Link", "", "gas"]],
            merge_fields,
            "D",
            "B",
            "2050",
            filter_positive=False,
            multiplier=-1,
            calculate_share=False,
            remove_list=["transport", "import", "pipeline"],
        )
    )
    for key, content in digestable_biomass_use_by_sector.items():
        # mulitply the values by the 1-gas share
        content["values"] = content["values"] * (
            1 - gas_shares[content["folder"]]["share"]-sabatier_shares[content["folder"]]["share"]
        )

    biomass_use_by_sector = solid_biomass_use_by_sector.copy()
    for key, value in digestable_biomass_use_by_sector.items():
        if key in biomass_use_by_sector:
            biomass_use_by_sector[key]["values"] += value["values"]
        else:
            biomass_use_by_sector[key] = value
    biomass_use_by_sector = calculate_share(biomass_use_by_sector)
    biomass_use_by_sector = split_CHP(biomass_use_by_sector) # here sth doesn't work

    export_results(
        biomass_use_by_sector, "biomass_use_by_sector.csv", include_share=True, export_dir=export_dir
    )

    solid_biomass_use = (
        get_data(  # this doesn't account for the gas share yet
            results,
            scenarios,
            "energy_balance",
            [["Link", "", "solid biomass"]],
            [],
            "D",
            "B",
            "2050",
            filter_positive=False,
            multiplier=-1,
            calculate_share=False,
            remove_list=["transport", "import", "pipeline", "biogas to gas", "SNG"],
        )
    )
    digestable_biomass_use = (
        get_data(  # this doesn't account for the gas share yet
            results,
            scenarios,
            "energy_balance",
            [["Link", "", "gas"]],
            [],
            "D",
            "B",
            "2050",
            filter_positive=False,
            multiplier=-1,
            calculate_share=False,
            remove_list=["transport", "import", "pipeline", "biogas to gas"],
        )
    )
    for key, content in digestable_biomass_use.items():
        # mulitply the values by the 1-gas share
        content["values"] = content["values"] * (
            1 - gas_shares[content["folder"]]["share"]-sabatier_shares[content["folder"]]["share"]
        )

    #merge digestable and solid biomass use by sector and sort dict
    biomass_use = calculate_share({**digestable_biomass_use, **solid_biomass_use})
    biomass_use = dict(sorted(biomass_use.items()))

    export_results(
        biomass_use, "biomass_use.csv", include_share=True, export_dir=export_dir
    )

    renewable_capacity = get_data(
        results,
        scenarios,
        "capacities",
        [["Generator", "solar"], ["Generator", "onwind"], ["Generator", "solar-hsat"]],
        [],
        "C",
        "B",
        "2050",
        filter_positive=True,
    )

    upstream_emissions = calculate_upstream_emissions(biomass_supply, scenarios,renewable_capacity)
    export_results(upstream_emissions, "upstream_emissions.csv", simply_print=True, export_dir=export_dir)

    remove_list = ["agriculture machinery oil1"]
    oil_production = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "","oil"]],
        [],
        "D",
        "B",
        "2050",
        filter_positive=True,
        remove_list=remove_list,
        calculate_share=True,
    )
    export_results(oil_production, "oil_production.csv", include_share=True, export_dir=export_dir)

    beccs = get_data(
        results,
        scenarios,
        "energy_balance",
        [
            ["Link", "BioSNG CC", "solid biomass"],
            ["Link", "biogas to gas CC", "solid biomass"],
            ["Link", "biomass to liquid CC", "solid biomass"],
            ["Link", "biomass-to-methanol CC", "solid biomass"],
            ["Link", "solid biomass for industry CC", "solid biomass"],
            ["Link", "solid biomass for lowT industry CC", "solid biomass"],
            ["Link", "solid biomass for mediumT industry CC", "solid biomass"],
            ["Link", "urban central solid biomass CHP CC", "solid biomass"],
        ],
        [],
        "D",
        "B",
        "2050",
        multiplier=-1,
        filter_positive=False,
        calculate_share=True,
    )
    export_results(beccs, "beccs.csv", export_dir=export_dir)

    capacities = get_data(
        results,
        scenarios,
        "capacities",
        ["",""],
        [],
        "C",
        "B",
        "2050",
        filter_positive=None,
        remove_list=["solar-hsat landuse emission","solar landuse emission","onwind landuse emission"]
    )
    export_results(capacities, "capacities.csv", export_dir=export_dir)

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

    co2 = get_data(
        results,
        scenarios,
        "energy_balance",
        [["Link", "", "co2"],["Generator", "", "co2"],["Link","","co2 sequestered"],["Link", "", "co2 stored"]],
        [],
        "D",
        "B",
        "2050",
        filter_positive=None,
        optional_columns=[["emission_type", "C"]],
    )

    #remove the share and the year keys
    for key, content in co2.items():
        if "share" in content:
            del content["share"]
        if "year" in content:
            del content["year"]

    export_results(co2, "co2.csv", include_share=False, export_dir=export_dir, simply_print=True)

    co2_sankey = modify_co2_data(co2,100)
    for key, content in co2.items():
        if "emission_type" in content:
            del content["emission_type"]

    export_results(
        co2_sankey,
        "co2_sankey.csv",
        include_share=False,
        export_dir=export_dir,
        simply_print=True,
    )

    results = load_results(results_dir, scenarios)
    capacity_factors = get_data(
        results,
        scenarios,
        "capacity_factors",
        [["Generator", "solar"],["Generator", "solar-hsat"],["Generator", "onwind"]],
        [],
        "C",
        "B",
        "2050",
        filter_positive=True,
        remove_list=["urban central solar thermal","urban decentral solar thermal","solar rooftop","rural solar thermal","onwind landuse emission","solar landuse emission","solar-hsat landuse emission"],
        round_digits =3,
    )
    export_results(
        capacity_factors,
        "capacity_factors.csv",
        export_dir=export_dir,
        round_digits = 3,
    )
    renewable_capacity = get_data(
        results,
        scenarios,
        "capacities",
        [["Generator", "solar"], ["Generator", "onwind"], ["Generator", "solar-hsat"], ["Generator", "offwind-ac"], ["Generator", "offwind-dc"],["Generator", "offwind-float"],["Generator", "solar rooftop"],],
        [[["offwind-dc", "onwind", "offwind-ac", "offwind-float"], "wind"],
         [["solar", "solar-hsat", "solar rooftop"], "solar"],
        ],
        "C",
        "B",
        "2050",
        filter_positive=True,
        )
    export_results(renewable_capacity, "renewable_capacity.csv", export_dir=export_dir)
    nuclear_capacity = get_data(
        results,
        scenarios,
        "capacities",
        [["Generator", "nuclear"]],
        [],
        "C",
        "B",
        "2050",
        filter_positive=True,
    )
    export_results(nuclear_capacity, "nuclear_capacity.csv", export_dir=export_dir)

def get_mga_results(results_dir="results/MGA", export_dir="export/mga"):
    """
    Extract and export MGA (Modeling to Generate Alternatives) results.
    
    Parameters
    ----------
    results_dir : str
        Directory containing MGA results
    export_dir : str
        Directory to export processed results
    """
    
    # Export total costs from main results
    main_results = load_results("results/main", "all")
    costs = get_data(
        main_results,
        "all",
        "metrics",
        [["total costs"]],
        [[["total costs"], "Total costs (Billion €)"]],
        "B",
        "A",
        "2050",
        1e-9,
    )
    export_results(costs, "total_costs.csv", export_dir=export_dir)

    # Load MGA results and categorize scenarios
    results = load_results(results_dir, "all")
    scenario_groups = _categorize_scenarios(results.keys())
    
    # Common parameters for data extraction
    common_params = {
        "dataframe": "energy_balance",
        "value_column": "D",
        "data_name_column": "B", 
        "year": "2050",
        "filter_positive": True,
    }
    
    # Biomass-specific parameters
    biomass_params = {
        **common_params,
        "fields_list": [["Link", "", "solid biomass"], ["Link", "", "biogas"]],
        "merge_fields": [[[""], "biomass"]],
        "remove_list": ["biomass transport", "solid biomass for industry", "solid biomass for industry CC"],
    }
    
    # Fossil fuel-specific parameters
    fossil_fuel_params = {
        **common_params,
        "fields_list": [["Generator", "", "oil primary"], ["Generator", "", "gas"], ["Generator", "", "coal"]],
        "merge_fields": [],
    }
    
    # Process each scenario group for both biomass and fossil fuels
    scenario_mappings = {
        "cscs": {
            "biomass": "biomass_use_carbon_costs.csv",
            "fossil": "fossil_fuel_supply_carbon_costs.csv"
        },
        "cscs_710": {
            "biomass": "biomass_use_carbon_costs_710.csv",
            "fossil": "fossil_fuel_supply_carbon_costs_710.csv"
        },
        "defaults": {
            "biomass": "biomass_use_default.csv", 
            "fossil": "fossil_fuel_supply_default.csv"
        },
        "defaults_710": {
            "biomass": "biomass_use_default_710.csv",
            "fossil": "fossil_fuel_supply_default_710.csv"
        }
    }
    
    for group_name, filenames in scenario_mappings.items():
        if scenario_groups[group_name]:  # Only process if group has scenarios
            # Extract and export biomass data
            biomass_data = get_data(results, scenario_groups[group_name], **biomass_params)
            export_results(biomass_data, filenames["biomass"], export_dir=export_dir)
            
            # Extract and export fossil fuel data
            fossil_data = get_data(results, scenario_groups[group_name], **fossil_fuel_params)
            export_results(fossil_data, filenames["fossil"], export_dir=export_dir)


def _categorize_scenarios(folder_names):
    """
    Categorize scenario folders based on naming patterns.
    
    Parameters
    ----------
    folder_names : iterable
        Collection of folder names to categorize
        
    Returns
    -------
    dict
        Dictionary with scenario categories as keys and lists of matching folders as values
    """
    categories = {
        "cscs": [],
        "cscs_710": [], 
        "defaults": [],
        "defaults_710": []
    }
    
    for folder in folder_names:
        has_710 = "710" in folder
        has_default = "default" in folder
        
        if has_default and has_710:
            categories["defaults_710"].append(folder)
        elif has_default and not has_710:
            categories["defaults"].append(folder)
        elif not has_default and has_710:
            categories["cscs_710"].append(folder)
        elif not has_default and not has_710:
            categories["cscs"].append(folder)
    
    return categories

 
def check_result_quality(results_dir="results"):
    """
    Check each folder in the results directory for "Numerical trouble encountered"
    in the logs/base_s_39___2025_solver.log file.

    Parameters
    ----------
    results_dir : str
        Path to the results directory.
    """
    troubled_folders = []
    suboptimal_folders = []

    # Iterate through each folder in the results directory
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        log_file_path = os.path.join(folder_path, "logs/base_s_39___2050_solver.log")
        
        # Check if the log file exists
        if os.path.isfile(log_file_path):
            with open(log_file_path, "r") as log_file:
                log_content = log_file.read()
                if "Numerical trouble encountered" in log_content:
                    troubled_folders.append(folder)
                if "Sub-optimal termination" in log_content:
                    suboptimal_folders.append(folder)

    # Print the folders with numerical trouble
    if troubled_folders:
        print("Folders with 'Numerical trouble encountered':")
        for folder in troubled_folders:
            print(folder)
    else:
        print("No folders with 'Numerical trouble encountered' found.")
    
    # Print the folders with suboptimal solutions
    if suboptimal_folders:
        print("Folders with 'Suboptimal solution found':")
        for folder in suboptimal_folders:
            print(folder)
    else:
        print("No folders with 'Suboptimal solution found' found.")


if __name__ == "__main__":
    #results_dir = "results"

    # scenarios = ["default_optimal", "optimal", "default_710_optimal", "710_optimal"]
    # difference_scenarios = ["default_optimal", "optimal"]
    # export_dir = "export/seq"

    results_dir = "results/main"
    scenarios = ["default","default_710","cscs", "cscs_710"]
    difference_scenarios = ["default", "cscs"]
    export_dir = "export/main"

    # check_result_quality(results_dir="results/MGA")

    # scenarios = ["default_optimal", "optimal"]
    # export_dir = "export/basic"

    main(results_dir=results_dir, export_dir=export_dir, scenarios=scenarios, difference_scenarios=difference_scenarios)

    #get_biomass_potentials(export_dir=export_dir)

    get_mga_results(results_dir="results/MGA", export_dir="export/mga")


    # results = load_results("results/GSA", "all")
    # nuclear_capacity = get_data(
    #     results,
    #     "all",
    #     "capacities",
    #     [["Generator", "nuclear"]],
    #     [],
    #     "C",
    #     "B",
    #     "2050",
    #     filter_positive=True,
    # )
    # export_results(nuclear_capacity, "nuclear_capacity.csv", export_dir="GSA/export")
