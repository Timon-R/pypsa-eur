# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
import os
import string

import pandas as pd

results = {}
# For every subfolder in results, make a list of all CSV files in the csvs folder within that subfolder
for folder in os.listdir("results"):
    folder_path = os.path.join("results", folder, "csvs")
    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    # Initialize an empty dictionary to store dataframes
    dataframes = {}
    # Loop through the CSV files and load each into a dataframe
    print(f"Loading data from {folder}...")
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        # Load the CSV file and drop the first 4 rows immediately
        df = pd.read_csv(file_path).drop(index=range(3))
        # rename the columns in numbers
        df.columns = list(string.ascii_uppercase[: len(df.columns)])
        # Use the filename without extension as the key
        key = os.path.splitext(file)[0]
        dataframes[key] = df
    results[folder] = dataframes


def get_total_costs(results, folders):
    """
    Get the total costs for each scenario.

    Parameters
    ----------
    results (dict): Dictionary containing the dataframes.
    folders (list): List of folders to process.

    Returns
    -------
    dict: Dictionary containing the total costs for each folder.
    """
    total_costs = {}

    for folder, dataframes in results.items():
        if folder in folders:
            if "costs" in dataframes:
                costs_df = dataframes["costs"]
                # Filter the rows based on the specified fields
                costs_df.iloc[:, 3] = pd.to_numeric(
                    costs_df.iloc[:, 3], errors="coerce"
                )
                total_costs_1 = costs_df.iloc[:, 3].sum()
                total_costs[folder] = total_costs_1 / 1e9  # Convert to billion €

    return total_costs


def get_data(results, folders, dataframe, fields, value_column, data_name):
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

    Returns
    -------
    dict: Dictionary containing the processed data.
    """
    result_data = {}

    for folder, dataframes in results.items():
        if folder in folders:
            data_df = dataframes[dataframe]
            # Dynamically construct the filtering condition based on the number of fields
            condition = True
            for i, field in enumerate(fields):
                condition &= data_df.iloc[:, i].str.contains(
                    field, case=False, na=False
                )

            # Filter the rows based on the constructed condition
            data = data_df[condition].reset_index(drop=True)
            if len(data_name) == 1:
                if data_name in data.columns:
                    text = data[data_name].values
                else:
                    text = f"Warning: '{data_name}' does not match any column name. Use capital letter like A,B ..."
                    text = data_name
            else:
                text = data_name

            result_data[folder] = {
                "data_name": text,
                "values": data[value_column].values.tolist(),
            }

    return result_data


def print_data(data):
    """
    Print the data in a readable format.

    Parameters
    ----------
    data (dict): Dictionary containing the processed data.
    """
    for folder, content in data.items():
        print(f"Folder: {folder}")
        print(f"Data Name: {content['data_name']}")
        print(f"Values: {content['values']}")
        print()


def export_results(results, folders, dataframe, column_names):
    """
    Export the results to a CSV file.

    Parameters
    ----------
    results (dict): Dictionary containing the dataframes.
    folders (list): List of folders to process.
    dataframe (str): The key to access the specific dataframe in the dictionary.
    column_names (list): List of column names to overwrite existing ones.
    """
    export_dir = "export"
    # Create the export directory if it doesn't exist
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    for folder, dataframes in results.items():
        if folder in folders:
            data_df = dataframes[dataframe]
            # Overwrite the column names
            if len(column_names) == len(data_df.columns):
                data_df.columns = column_names
            else:
                print(
                    f"Warning: The number of column names provided does not match the number of columns in the dataframe for folder '{folder}' and dataframe '{dataframe}'. Skipping column renaming."
                )
            # Construct the output filename
            output_filename = f"{folder}_{dataframe}.csv"
            output_path = os.path.join(export_dir, output_filename)
            # Export the dataframe to a CSV file
            data_df.to_csv(output_path, index=False)
            print(f"Data exported to: {output_path}")


# folders = ['no_emissions', 'em_100', 'em_150', 'em_200', 'em_250', 'em_300', 'em_1000']
# folders = ['low_price_no_em', 'low_price_em', 'low_price_high_em', 'high_price_no_em', 'high_price_em', 'high_price_high_em']
# no and high only
# folders = ['low_price_no_em',  'low_price_high_em', 'high_price_no_em', 'high_price_high_em']
# all results folders in results

biomass_types = [
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
]

folders = os.listdir("results")

# Print total costs
total_costs = get_total_costs(results, folders)
for folder, cost in total_costs.items():
    print(f"Billion € costs for: {folder}")
    print(f"2050: {cost}")
    print()

# Get data for each biomass type and print the results

for biomass in biomass_types:
    supply_data = get_data(
        results,
        folders,
        "supply_energy",
        [biomass, "stores", ""],
        "D",
        f"{biomass} supply",
    )
    print_data(supply_data)
for biomass in biomass_types:
    costs_data = get_data(
        results,
        folders,
        "costs",
        ["stores", "marginal", biomass],
        "D",
        f"Costs of {biomass}",
    )
    print_data(costs_data)

# Total use of biomass used
# Calculate total supply of biomass used
# need to make new

# Redirection of biomass use
biomass_use = get_data(
    results, folders, "supply_energy", ["solid biomass", "links", ""], "D", "C"
)
# Filter out positive values and corresponding data names
biomass_use = {
    folder: {
        "data_name": [
            data["data_name"][i] for i, v in enumerate(data["values"]) if float(v) < 0
        ],
        "values": [float(v) for v in data["values"] if float(v) < 0],
    }
    for folder, data in biomass_use.items()
}
print_data(biomass_use)
# Other areas of the energy sector that are significantly affected? (wind and solar, fossil fuels, CCS)

# Shadow price of CO2
shadow_price = get_data(
    results, folders, "metrics", ["co2_shadow"], "B", "CO2 shadow price"
)
print_data(shadow_price)

gas_supply = get_data(
    results, folders, "supply_energy", ["gas", "generators", ""], "D", "Gas supply"
)
print_data(gas_supply)


# export_results(results, folders, 'supply_energy', ['Carrier','Component','Technology','Value'])
