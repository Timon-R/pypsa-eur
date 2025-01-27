# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


def load_csv(file_path):
    """
    Load a CSV file and return the data as a pandas DataFrame.

    Parameters
    ----------
    file_path (str): The path to the CSV file to load.

    Returns
    -------
    pd.DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)


def plot_difference_bar(
    df,
    title,
    x_label,
    y_label,
    file_path,
    custom_order=None,
    remove_last_letters=0,
    axis2_ticks=500,
):
    # Pivot the DataFrame for plotting

    if custom_order is not None:
        df = df.assign(
            Folder=pd.Categorical(df["Folder"], categories=custom_order, ordered=True)
        )
        # Sort the data by the new folder order
        df = df.sort_values(by=["Folder", "Year"])

    if remove_last_letters != 0:
        df["Data Name"] = df["Data Name"].str[:-remove_last_letters]

    df["Data Name"] = df["Data Name"].str.replace("1", "")

    pivot_df = df.pivot(index="Data Name", columns="Folder", values=["Values", "Share"])

    # Ensure there are exactly two scenarios to calculate differences
    if len(pivot_df["Values"].columns) != 2:
        raise ValueError(
            "The DataFrame must contain exactly two scenarios to calculate differences."
        )

    # Calculate differences between the two scenarios
    value_diff = (
        pivot_df["Values"].iloc[:, 1] - pivot_df["Values"].iloc[:, 0]
    )  # Scenario 2 - Scenario 1
    share_diff = pivot_df["Share"].iloc[:, 1] - pivot_df["Share"].iloc[:, 0]

    # Define x-axis positions and bar width
    x = np.arange(len(value_diff))  # Positions for each Data Name
    bar_width = 0.6

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(
        x,
        value_diff / 1e6,
        bar_width,
        color=["green" if val > 0 else "red" for val in value_diff],
        label="Difference",
    )

    # Add absolute values and share differences as labels
    for i, (value, share_delta) in enumerate(zip(value_diff, share_diff)):
        # Absolute value on the bars
        absolute_text = f"Δ {value/1e6:.0f}"
        ax.text(
            x[i],
            value / 1e6 + 0.03 * max(value_diff) / 1e6,
            absolute_text,
            ha="center",
            fontsize=12,
            color="black",
        )

        # Share difference above or below the bars
        delta_text = f"Δ {share_delta * 100:.1f}%"
        ax.text(
            x[i],
            value / 1e6 + 0.08 * max(value_diff) / 1e6,
            delta_text,
            ha="center",
            fontsize=12,
            color="blue",
        )

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))  # Add 10% to the y-axis limit

    # Customise the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks() * 3.6)
    ax2.set_ylabel("PJ")

    secondary_locator = MultipleLocator(axis2_ticks)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    plt.tight_layout()
    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()


def plot_bar_with_shares(
    df,
    title,
    x_label,
    y_label,
    file_path,
    custom_order=None,
    axis2_ticks=500,
    remove_last_letters=0,
    width=14,
    threshold=0.001,
):
    # Pivot the DataFrame for plotting

    plt.rcParams.update({"font.size": 14})
    if custom_order is not None:
        df = df.assign(
            Folder=pd.Categorical(df["Folder"], categories=custom_order, ordered=True)
        )
        # Sort the data by the new folder order
        df = df.sort_values(by=["Folder", "Year"])

    if remove_last_letters != 0:
        df["Data Name"] = df["Data Name"].str[:-remove_last_letters]

    # delete 1 from data_name if it has it
    df["Data Name"] = df["Data Name"].str.replace("1", "")
    df["Data Name"] = df["Data Name"].str.replace("2", "")
    df["Data Name"] = df["Data Name"].str.replace("3", "")
    df["Data Name"] = df["Data Name"].str.replace("4", "")

    pivot_df = df.pivot(index="Data Name", columns="Folder", values=["Values", "Share"])

    # only use data names that have a share higher than 0.001
    pivot_df = pivot_df[pivot_df["Share"].max(axis=1) > threshold]

    # Extract values and shares for plotting
    values = pivot_df["Values"] / 1e6  # Convert to TWh
    shares = pivot_df["Share"]

    # Define x-axis positions and bar width
    x = np.arange(len(values))  # Positions for each Data Name
    bar_width = 0.35

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(width, 6))
    for i, folder in enumerate(values.columns):
        ax.bar(x + i * bar_width, values[folder], bar_width, label=folder)

        # Add share values as labels above the bars
        for j, value in enumerate(values[folder]):
            share_text = f"{shares[folder].iloc[j] * 100:.1f}%"
            absolute_text = f"{value:.0f}"
            ax.text(
                x[j] + i * bar_width,
                value + max(values.max()) * 0.01,
                absolute_text + "\n" + share_text,
                ha="center",
                fontsize=8,
            )

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))  # Add 10% to the y-axis limit

    # Customise the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(values.index, rotation=45, ha="right")
    ax.legend(title="Scenario")

    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks() * 3.6)
    ax2.set_ylabel("PJ")

    secondary_locator = MultipleLocator(axis2_ticks)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    plt.tight_layout()
    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()


def plot_shares(df, title, x_label, y_label, file_path, custom_order=None):
    """
    Creates a stacked bar chart based on the provided DataFrame and saves it to file.

    Args:
        df (pd.DataFrame): Input DataFrame with columns: 'Folder', 'Year', 'Data Name', 'Value', 'Share'.
        title (str): Title of the chart.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        file_path (str): The name of the image to save the plot.
    """

    # Convert 'Folder' column to a categorical type with the specified order
    if custom_order is not None:
        df = df.assign(
            Folder=pd.Categorical(df["Folder"], categories=custom_order, ordered=True)
        )
        # Sort the data by the new folder order
        df = df.sort_values(by=["Folder", "Year"])

    # Filter out shares less or equal than 0.01
    df_filtered = df[df["Share"] > 0.01].copy()

    # Get unique years and folders for chart layout
    unique_years = df_filtered["Year"].unique()
    unique_folders = df_filtered["Folder"].unique()
    num_folders = len(unique_folders)

    unique_data_names = df_filtered["Data Name"].unique()
    palette = sns.color_palette("viridis", len(unique_data_names))
    data_name_colors = {
        data_name: palette[i] for i, data_name in enumerate(unique_data_names)
    }

    # Figure and axes setup
    fig, ax = plt.subplots(figsize=(12, 8))

    # Bar width and spacing setup
    bar_width = 0.6 / num_folders  # Reduced the total bar width to have space
    group_spacing = 0.05  # Space between groups of bars within the year

    x_positions = np.arange(len(unique_years))

    # Plot the stacked bars
    for i, folder in enumerate(unique_folders):
        bar_positions = x_positions + (
            i * (bar_width + group_spacing)
        )  # Adjust position to the right of each other

        for j, year in enumerate(unique_years):
            folder_year_data = df_filtered[
                (df_filtered["Folder"] == folder) & (df_filtered["Year"] == year)
            ]  # Filter df

            bottom = np.zeros(1)
            for dataname in folder_year_data["Data Name"].unique():
                share = folder_year_data[folder_year_data["Data Name"] == dataname][
                    "Share"
                ].values[0]  # Get value of share
                ax.bar(
                    bar_positions[j],
                    share,
                    bar_width,
                    bottom=bottom,
                    color=data_name_colors[dataname],
                    label=dataname if i == 0 else None,
                )  # Set label only in first bar of a group to not repeat
                bottom += share  # Update bottom for the next stack

            ax.text(
                bar_positions[j],
                -0.07,
                folder,
                ha="center",
                va="top",
                fontsize=8,
                rotation=90,
            )  # Added legend below each bar

    # Set X-Axis labels to the years
    ax.set_xticks(x_positions + ((num_folders - 1) * (bar_width + group_spacing) / 2))
    ax.set_xticklabels(unique_years)

    # Add labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add legend for all unique Data Names
    handles = [
        plt.Line2D([0], [0], color=data_name_colors[dataname], lw=4)
        for dataname in df_filtered["Data Name"].unique()
    ]
    labels = df_filtered["Data Name"].unique()
    ax.legend(
        handles, labels, title="Data Name", bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    # Layout adjustments
    plt.tight_layout()

    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    full_file_path = os.path.join(export_dir, file_path)
    plt.savefig(full_file_path)
    plt.close()


def plot_data(data, title, x_label, y_label, file_path, custom_order=None):
    """
    Plot data
    """
    if custom_order is not None:
        # Convert 'Folder' column to a categorical type with the specified order
        data = data.assign(
            Folder=pd.Categorical(data["Folder"], categories=custom_order, ordered=True)
        )
        # Sort the data by the new folder order
        data = data.sort_values(by=["Folder", "Year"])

    # Recreate the bar chart with the reordered folders
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=data, x="Year", y="Values", hue="Folder", palette="viridis", orient="v"
    )
    # Add labels and title
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(title="Folder", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Show the plot
    plt.tight_layout()

    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()


def plot_biomass_use(df, title, x_label, y_label, file_path, year=2050):
    # Filter by year

    plt.rcParams.update({"font.size": 18})
    emission_factors = {
        "agricultural waste": 0.108,
        "fuelwood residues": 0.036,
        "secondary forestry residues": 0.144,
        "sawdust": 0.108,
        "residues from landscape care": 0,
        "grasses": 0.216,
        "woody crops": 0.18,
        "fuelwoodRW": 0.288,
        "manure": 0.072,
        "sludge": 0,
    }

    df = df[df["Year"] == year]
    # Conversion factors
    mwh_to_twh = 1e-6
    mwh_to_pj = 3.6e-6

    # Pivot the table for easier plotting
    df_pivot = df.pivot_table(
        index="Data Name", columns="Folder", values="Values"
    ).reset_index()
    df_pivot = df_pivot.rename(
        columns={"endogenous_0_ef": "Scenario A", "endogenous_1_ef": "Scenario B"}
    )

    # Convert values to TWh
    df_pivot["Scenario A"] = df_pivot["Scenario A"] * mwh_to_twh
    df_pivot["Scenario B"] = df_pivot["Scenario B"] * mwh_to_twh

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    bar_width = 0.8
    gap = 0.6  # Adjust this value to change the space between bars
    x_coords = np.arange(len(df_pivot)) * (bar_width + gap)

    for index, row in df_pivot.iterrows():
        biomass_type = row["Data Name"]
        value_a = row["Scenario A"]
        value_b = row["Scenario B"]

        # Base bar (Scenario A)
        ax.bar(
            x_coords[index],
            value_a,
            width=bar_width,
            label="Scenario A" if index == 0 else "",
            color="LightGreen",
            hatch="///",
        )
        ax.bar(
            x_coords[index],
            value_b,
            width=bar_width,
            label="Scenario B" if index == 0 else "",
            color="LightGreen",
        )

        # Add emission factor below the x-axis
        emission_factor = emission_factors.get(biomass_type, "N/A")
        ef_g_per_MJ = round(emission_factor / 0.0036)
        ax.text(
            x_coords[index],
            value_a + 1,
            f"{emission_factor} | {ef_g_per_MJ} \n ton/MWh | g/MJ",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    a_patch = mpl.patches.Patch(
        facecolor="LightGreen", label="Biomass use - no emissions considered"
    )
    b_patch = mpl.patches.Patch(
        facecolor="none",
        hatch="///",
        label="Difference in biomass use - emissions considered",
    )
    plt.legend(handles=[a_patch, b_patch], fontsize=16)

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))  # Add 10% to the y-axis limit

    # Customize plot
    ax.set_xticks(x_coords)
    # new_labels = [f"{name}\n{emission_factors.get(name, 'N/A')} | {emission_factors.get(name, 'N/A') / 0.0036 if emission_factors.get(name, 'N/A') != 'N/A' else 'N/A'} \n (ton/MWh | g/MJ)" for name in df_pivot['Data Name']]
    ax.set_xticklabels(df_pivot["Data Name"], rotation=45, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=26)

    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks() * mwh_to_pj / mwh_to_twh)
    ax2.set_ylabel("PJ")

    secondary_locator = MultipleLocator(250)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    plt.tight_layout()

    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()


def plot_efs():
    emission_factors = {
        "agricultural waste": 0.108,
        "fuelwood residues": 0.036,
        "secondary forestry residues": 0.144,
        "sawdust": 0.108,
        "residues from landscape care": 0,
        "grasses": 0.216,
        "woody crops": 0.18,
        "fuelwoodRW": 0.288,
        "manure": 0.072,
        "sludge": 0,
    }

    # font size
    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(14, 10))
    bar_width = 0.8
    gap = 0.6  # Adjust this value to change the space between bars
    x_coords = np.arange(len(emission_factors)) * (bar_width + gap)

    for index, (biomass_type, ef) in enumerate(emission_factors.items()):
        ax.bar(
            x_coords[index], ef, width=bar_width, label=biomass_type, color="LightGreen"
        )

    # Add gas emission factor
    x_coords = np.append(x_coords, [x_coords[-1] + (bar_width + gap)])
    ax.bar(x_coords[-1], 0.198, width=bar_width, label="natural gas", color="Grey")

    # Add coal
    x_coords = np.append(x_coords, [x_coords[-1] + (bar_width + gap)])
    ax.bar(x_coords[-1], 0.3361, width=bar_width, label="coal", color="Black")

    ax.set_xticks(x_coords)
    # ax.set_xticklabels(emission_factors.keys(), rotation=45, ha="right")
    ax.set_xticklabels(
        list(emission_factors.keys()) + ["Natural Gas", "Coal"], rotation=45, ha="right"
    )
    ax.set_xlabel("")
    ax.set_ylabel("ton/MWh")
    ax.set_title("Emission factors for different feedstocks", fontsize=26)

    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks() / 0.0036)
    ax2.set_ylabel("g/MJ")

    secondary_locator = MultipleLocator(10)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    plt.tight_layout()

    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, "emission_factors.png")
    plt.savefig(file_path)
    plt.close()


def plot_bar_with_totals(
    df,
    title,
    x_label,
    y_label,
    file_path,
    custom_order=None,
    remove_last_letters=0,
    axis2_ticks=500,
    include_total=True,
):
    # plots the data in a bar chart and adds a sum of all values at the end
    # Pivot the DataFrame for plotting

    if custom_order is not None:
        df = df.assign(
            Folder=pd.Categorical(df["Folder"], categories=custom_order, ordered=True)
        )
        # Sort the data by the new folder order
        df = df.sort_values(by=["Folder", "Year"])

    if remove_last_letters != 0:
        df["Data Name"] = df["Data Name"].str[:-remove_last_letters]

    df["Data Name"] = df["Data Name"].str.replace("1", "")
    df["Data Name"] = df["Data Name"].str.replace("2", "")
    df["Data Name"] = df["Data Name"].str.replace("3", "")
    df["Data Name"] = df["Data Name"].str.replace("4", "")

    # Calculate the total values for each scenario
    if include_total:
        total_values = df.groupby("Folder")["Values"].sum().reset_index()
        total_values["Data Name"] = "Total"

        # Append the total values to the original DataFrame
        df = pd.concat([df, total_values], ignore_index=True, sort=False)

        # Ensure 'Total' is at the end
        data_names = list(df["Data Name"].unique())
        if "Total" in data_names:
            data_names.remove("Total")
        data_names.append("Total")
        df["Data Name"] = pd.Categorical(
            df["Data Name"], categories=data_names, ordered=True
        )
        df = df.sort_values(by="Data Name")

    pivot_df = df.pivot(index="Data Name", columns="Folder", values="Values")
    # convert from MWh to TWh
    pivot_df = pivot_df / 1e6

    # Define x-axis positions and bar width
    x = np.arange(len(pivot_df))  # Positions for each Data Name
    bar_width = 0.35

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, folder in enumerate(pivot_df.columns):
        ax.bar(x + i * bar_width, pivot_df[folder], bar_width, label=folder)

    # Add values as labels above the bars
    for i, folder in enumerate(pivot_df.columns):
        for j, value in enumerate(pivot_df[folder]):
            ax.text(
                x[j] + i * bar_width,
                value + max(pivot_df.max()) * 0.01,
                f"{value:.0f}",
                ha="center",
                fontsize=8,
            )

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))  # Add 10% to the y-axis limit

    # Customise the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
    ax.legend(title="Scenario")

    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks() * 3.6)
    ax2.set_ylabel("PJ")

    secondary_locator = MultipleLocator(axis2_ticks)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    plt.tight_layout()
    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()


def __main__():
    # Define the desired folder order
    # custom_order = [
    #     "exogenous_0_ef",
    #     "endogenous_0_ef",
    #     "endogenous_0_9_ef",
    #     "endogenous_1_ef",
    #     "endogenous_1_1_ef",
    #     "endogenous_1_ef_2_gas",
    # ]
    custom_order = [
        "endogenous_0_ef",
        "endogenous_1_ef",
    ]

    data = load_csv("export/biomass_supply.csv")
    plot_biomass_use(data, "Biomass Supply", "", "TWh", "biomass_supply.png")

    plot_efs()

    data = load_csv("export/oil_production_2050.csv")
    plot_bar_with_shares(
        data,
        "Liquid Fuel Production in 2050",
        "",
        "TWh",
        "oil_production_2050.png",
        custom_order,
        remove_last_letters=1,
        width=10,
    )

    data = load_csv("export/electricity_generation_share_2050.csv")
    plot_bar_with_shares(
        data,
        "Electricity Generation in 2050",
        "",
        "TWh",
        "electricity_generation_share_2050.png",
        custom_order,
        axis2_ticks=5000,
        width=10,
    )
    plot_difference_bar(
        data,
        "Difference in Electricity Generation Considering Biomass Emissions",
        "",
        "TWh",
        "electricity_generation_share_2050_diff.png",
        custom_order,
    )

    data = load_csv("export/beccs.csv")
    plot_bar_with_totals(
        data,
        "BECCS",
        "",
        "TWh",
        "beccs.png",
        custom_order,
        remove_last_letters=1,
        axis2_ticks=500,
    )

    data = load_csv("export/industrial_energy_2050.csv")
    plot_bar_with_totals(
        data,
        "Industrial Heat Supply in 2050",
        "",
        "TWh",
        "industrial_energy_2050.png",
        custom_order,
        axis2_ticks=500,
        include_total=False,
    )

    data = load_csv("export/heating_energy_2050.csv")
    plot_bar_with_totals(
        data,
        "Heating Energy Supply in 2050",
        "",
        "TWh",
        "heating_energy_2050.png",
        custom_order,
        axis2_ticks=1000,
        include_total=False,
    )

    data = load_csv("export/primary_energy_2050.csv")
    plot_bar_with_shares(
        data,
        "Primary Energy Supply in 2050",
        "",
        "TWh",
        "primary_energy_2050.png",
        custom_order,
        axis2_ticks=5000,
        width=10,
        threshold=0.0001,
    )

    # data = load_csv("export/combined_costs.csv")
    # plot_data(data, "Total Costs", "Year", "Cost (Billion EUR)", "total_costs.png")

    # data = load_csv("export/combined_shadow_price.csv")
    # plot_data(
    #     data, "Shadow Prices", "Year", "Shadow Price (EUR/MWh)", "shadow_prices.png"
    # )

    # data = load_csv("export/gas_supply.csv")
    # plot_data(data, "Gas Supply", "Year", "Supply (MWh)", "gas_supply.png")

    # data = load_csv("export/biomass_use_sum.csv")
    # plot_data(data, "Biomass Use", "Year", "Use (MWh)", "biomass_use.png")

    # data = load_csv("export/electricity_generation_share.csv")
    # unique_data_names = data["Data Name"].unique()
    # for data_name in unique_data_names:
    #     filtered_data = data[data["Data Name"] == data_name]
    #     title = f"{data_name}"
    #     output_file = f"{data_name.replace(' ', '_')}.png"
    #     plot_data(filtered_data, title, "Year", "Generation (MWh)", output_file)

    # # Plot sum of all wind
    # wind_data = data[data["Data Name"].str.contains("wind")]
    # title = "Wind Power Generation"
    # output_file = "wind_generation.png"
    # wind_data = wind_data.groupby(["Year", "Folder"]).sum().reset_index()
    # plot_data(wind_data, title, "Year", "Generation (MWh)", output_file)

    # # Plot sum of all solar
    # solar_data = data[data["Data Name"].str.contains("solar")]
    # solar_rooftop = load_csv("export/solar_rooftop.csv")
    # solar_data = pd.concat([solar_data, solar_rooftop], ignore_index=True)
    # title = "Solar Power Generation"
    # output_file = "solar_generation.png"
    # solar_data = solar_data.groupby(["Year", "Folder"]).sum().reset_index()
    # plot_data(solar_data, title, "Year", "Generation (MWh)", output_file)

    # data = load_csv("export/biomass_use.csv")
    # unique_data_names = data["Data Name"].unique()
    # for data_name in unique_data_names:
    #     filtered_data = data[data["Data Name"] == data_name]
    #     title = f"{data_name}"
    #     output_file = f"{data_name.replace(' ', '_')}.png"
    #     plot_data(filtered_data, title, "Year", "Biomass Input in MWh", output_file)

    # data = load_csv("export/biomass_supply.csv")
    # unique_data_names = data["Data Name"].unique()
    # for data_name in unique_data_names:
    #     filtered_data = data[data["Data Name"] == data_name]
    #     title = f"{data_name}"
    #     output_file = f"{data_name.replace(' ', '_')}.png"
    #     plot_data(filtered_data, title, "Year", "Biomass Supply in MWh", output_file)

    # data = load_csv("export/EV_data.csv")
    # unique_data_names = data["Data Name"].unique()
    # for data_name in unique_data_names:
    #     filtered_data = data[data["Data Name"] == data_name]
    #     title = f"{data_name}"
    #     output_file = f"{data_name.replace(' ', '_')}.png"
    #     plot_data(filtered_data, title, "Year", data_name, output_file)

    # data = load_csv("export/storage_data.csv")
    # unique_data_names = data["Data Name"].unique()
    # for data_name in unique_data_names:
    #     filtered_data = data[data["Data Name"] == data_name]
    #     title = f"{data_name}"
    #     output_file = f"{data_name.replace(' ', '_')}.png"
    #     plot_data(filtered_data, title, "Year", data_name, output_file)
    # data = load_csv("export/oil_supply.csv")
    # plot_data(data, "Oil Supply", "Year", "Supply (MWh)", "oil_supply.png")

    # data = load_csv("export/biomass_prices.csv")
    # plot_data(data, "Biomass Prices", "Year", "Price (EUR/MWh)", "biomass_prices.png")

    # data = load_csv("export/mediumT_share.csv")
    # plot_shares(
    #     data, "Medium Temperature Heat Industry", "Year", "Share", "mediumT_shares.png"
    # )

    # data = load_csv("export/highT_share.csv")
    # plot_shares(
    #     data, "High Temperature Heat Industry", "Year", "Share", "highT_shares.png"
    # )

    # data = load_csv("export/lowT_share.csv")
    # plot_shares(
    #     data, "Low Temperature Heat Industry", "Year", "Share", "lowT_shares.png"
    # )

    # data = load_csv("export/urban_central_heat_share.csv")
    # plot_shares(
    #     data, "Urban Central Heat", "Year", "Share", "urban_central_heat_shares.png"
    # )

    # data = load_csv("export/urban_decentral_heat_share.csv")
    # plot_shares(
    #     data, "Urban Decentral Heat", "Year", "Share", "urban_decentral_heat_shares.png"
    # )

    # data = load_csv("export/rural_heating_share.csv")
    # plot_shares(data, "Rural Heating", "Year", "Share", "rural_heating_shares.png")

    # data = load_csv("export/low_voltage_share_output.csv")
    # plot_shares(
    #     data, "Low Voltage Output", "Year", "Share", "low_voltage_output_shares.png"
    # )

    # data = load_csv("export/solar_rooftop.csv")
    # plot_data(data, "Solar Rooftop", "Year", "Generation (MWh)", "solar_rooftop.png")

    # data = load_csv("export/electricity_generation_share.csv")
    # plot_shares(
    #     data,
    #     "Electricity Generation Share (AC)",
    #     "Year",
    #     "Share",
    #     "electricity_generation_shares.png",
    # )


if __name__ == "__main__":
    __main__()
