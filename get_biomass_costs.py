# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd


def get_biomass_costs(scenario="ENS_Med", year=2050):
    # Scenarios: ENS_Low, ENS_Med, ENS_High
    # Years: 2010, 2020, ... , 2050

    enspreso = "data/ENSPRESO_BIOMASS.xlsx"
    enspreso = pd.ExcelFile(enspreso)

    database_file = enspreso

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

    # Filter the costs and potentials DataFrames based on the year and scenario
    filtered_costs = costs[(costs["Year"] == year) & (costs["Scenario"] == scenario)]
    filtered_potentials = potentials[
        (potentials["Year"] == year) & (potentials["Scenario"] == scenario)
    ]
    # Rename the third column (energy commodity) based on the name_dict
    filtered_costs = filtered_costs.replace({"Energy Commodity": name_dict})
    filtered_potentials = filtered_potentials.replace({"Energy Commodity": name_dict})
    # Convert costs from per GJ to per MWh
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
    )  # in Euro2010/MWh

    # Define the new technology data
    parameter = "fuel"
    unit = "Euro/MWh_th"
    source = f"Weighted average costs from JLC ENSPRESO, scenario {scenario}, year 2050"
    description = "Weighted by country potentials"
    currency_year = 2010

    # Create a list of tuples for the MultiIndex
    indices = [
        (row["Energy Commodity"], parameter) for _, row in weighted_avg_costs.iterrows()
    ]

    # Create DataFrame with MultiIndex
    data = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(indices),
        columns=["value", "unit", "source", "further description", "currency_year"],
    )

    for _, row in weighted_avg_costs.iterrows():
        index = (row["Energy Commodity"], parameter)
        data.loc[index, "value"] = row["Weighted_Average_Cost"]
        data.loc[index, "unit"] = unit
        data.loc[index, "source"] = source
        data.loc[index, "further description"] = description
        data.loc[index, "currency_year"] = currency_year

    return data


def create_biomass_costs_comparison(year=2050):
    # Scenarios to compare
    scenarios = ["ENS_Low", "ENS_Med", "ENS_High"]

    # Create an empty DataFrame to store the comparison
    comparison_df = pd.DataFrame()

    # Get data for each scenario
    for scenario in scenarios:
        data = get_biomass_costs(scenario, year)

        # Extract biomass type and cost value
        scenario_costs = data.reset_index()
        scenario_costs.columns = [
            "Biomass_Type",
            "Parameter",
            "Value",
            "Unit",
            "Source",
            "Description",
            "Currency_Year",
        ]

        # Only keep biomass type and value
        scenario_costs = scenario_costs[["Biomass_Type", "Value"]]

        # Rename the Value column to the scenario name
        scenario_costs.rename(columns={"Value": scenario}, inplace=True)

        # If this is the first scenario, use it as the base for comparison_df
        if comparison_df.empty:
            comparison_df = scenario_costs
        else:
            # Otherwise merge with the existing comparison_df
            comparison_df = pd.merge(
                comparison_df, scenario_costs, on="Biomass_Type", how="outer"
            )

    # Save to CSV
    output_file = f"biomass_costs_comparison_{year}.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"Comparison saved to {output_file}")

    return comparison_df


if __name__ == "__main__":
    # Create the comparison CSV for year 2050
    biomass_costs_df = create_biomass_costs_comparison(2050)

    # Display the first few rows
    print(biomass_costs_df.head())
