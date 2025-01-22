# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import yaml

# This is not part of the snakemake workflow

# 2010 Euro! so needs to be inflation adjusted to 2020 costs (which is done in the tech-data script)

# Scenarios: ENS_Low, ENS_Med, ENS_High
# Years: 2010, 2020, ... , 2050


def create_biomass_costs(
    output_folder,
    config="config/config.yaml",
    scenario="ENS_Med",
    year=2050,
    database_file="data/ENSPRESO_BIOMASS.xlsx",
):
    ######## Read in data from ENSPRESO and create csv file for biomass costs
    costs = pd.read_excel(
        database_file, sheet_name="COST - NUTS0 EnergyCom", header=0
    ).fillna(0)

    potentials = pd.read_excel(
        database_file, sheet_name="ENER - NUTS0 EnergyCom", header=0
    ).fillna(0)

    name_dict = {
        "MINBIOAGRW1": "Agricultural waste",
        "MINBIOGAS1": "Manure solid, liquid",
        "MINBIOFRSR1a": "Residues from landscape care",
        "MINBIOCRP11": "Bioethanol barley, wheat, grain maize, oats, other cereals and rye",
        "MINBIOCRP21": "Sugar from sugar beet",
        "MINBIOCRP31": "Miscanthus, switchgrass, RCG",
        "MINBIOCRP41": "Willow",
        "MINBIOCRP41a": "Poplar",
        "MINBIOLIQ1": "Sunflower, soya seed",
        "MINBIORPS1": "Rape seed",
        "MINBIOFRSR1": "Fuelwood residues",
        "MINBIOWOO": "FuelwoodRW",
        "MINBIOWOOa": "C&P_RW",
        "MINBIOWOOW1": "Secondary Forestry residues - woodchips",
        "MINBIOWOOW1a": "Sawdust",
        "MINBIOMUN1": "Municipal waste",
        "MINBIOSLU1": "Sludge",
    }

    # Load the config.yaml file
    with open(config) as file:
        config = yaml.safe_load(file)

    # Extract the dictionary
    biomass_classes = config["biomass"]["classes"]
    # scenario = config['biomass']['scenario']
    # year = config['biomass']['year']

    # Filter the costs and potentials DataFrames based on the year and scenario
    filtered_costs = costs[(costs["Year"] == year) & (costs["Scenario"] == scenario)]
    filtered_potentials = potentials[
        (potentials["Year"] == year) & (potentials["Scenario"] == scenario)
    ]
    # Rename the third column (energy commodity) based on the name_dict
    filtered_costs = filtered_costs.replace({"Energy Commodity": name_dict})
    filtered_potentials = filtered_potentials.replace({"Energy Commodity": name_dict})
    # Convert costs from GJ to MWh by dividing by 3.6
    filtered_costs["NUTS0 Energy Commodity Cost "] = (
        filtered_costs["NUTS0 Energy Commodity Cost "] * 3.6
    )

    # Select the desired columns (4, 3, 5)
    selected_costs = filtered_costs[
        ["NUTS0", "Energy Commodity", "NUTS0 Energy Commodity Cost "]
    ]  # Replace 'Cost1', 'Cost2', 'Cost3' with actual column names
    selected_potentials = filtered_potentials[
        ["NUTS0", "Energy Commodity", "Value"]
    ]  # Replace with actual column names

    # Create the CSV file
    csv_filename = output_folder + f"biomass_costs_all_{year}_{scenario}.csv"
    selected_costs.to_csv(csv_filename, index=False)

    # Merge the costs and potentials DataFrames on 'NUTS0' and 'Energy Commodity'
    merged_df = pd.merge(
        selected_costs, selected_potentials, on=["NUTS0", "Energy Commodity"]
    )

    # Calculate the weighted average costs for each biomass type
    weighted_avg_costs = (
        merged_df.groupby("Energy Commodity")
        .agg(
            Weighted_Average_Cost=(
                "NUTS0 Energy Commodity Cost ",
                lambda x: np.average(x, weights=merged_df.loc[x.index, "Value"]),
            )
        )
        .reset_index()
    )

    # Create the CSV file for weighted average costs
    weighted_avg_csv_filename = (
        f"{output_folder}/weighted_average_costs_{year}_{scenario}.csv"
    )
    weighted_avg_costs.to_csv(weighted_avg_csv_filename, index=False)

    # Group the biomass types as specified in the biomass_classes
    grouped_costs = []
    for group, types in biomass_classes.items():
        if group == "not included":
            continue
        group_df = merged_df[merged_df["Energy Commodity"].isin(types)]
        if not group_df.empty:
            weighted_avg_cost = np.average(
                group_df["NUTS0 Energy Commodity Cost "], weights=group_df["Value"]
            )
            total_potential = group_df["Value"].sum()
            grouped_costs.append(
                {
                    "Group": group,
                    "Weighted Average Cost": weighted_avg_cost,
                    "Total Potential": total_potential,
                }
            )

    grouped_costs_df = pd.DataFrame(grouped_costs)

    # Create the CSV file for grouped weighted average costs
    grouped_weighted_avg_csv_filename = (
        f"{output_folder}/grouped_weighted_average_costs_{year}_{scenario}.csv"
    )
    grouped_costs_df.to_csv(grouped_weighted_avg_csv_filename, index=False)


if __name__ == "__main__":
    create_biomass_costs("export/", scenario="ENS_Med", year=2030)
