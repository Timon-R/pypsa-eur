# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def plot_shares(df, title, x_label, y_label, file_path):
    """
    Creates a stacked bar chart based on the provided DataFrame and saves it to file.

    Args:
        df (pd.DataFrame): Input DataFrame with columns: 'Folder', 'Year', 'Data Name', 'Value', 'Share'.
        title (str): Title of the chart.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        file_path (str): The name of the image to save the plot.
    """
    custom_order = [
        "exogenous_0_ef",
        "endogenous_0_ef",
        "endogenous_0_9_ef",
        "endogenous_1_ef",
        "endogenous_1_1_ef",
        "endogenous_1_ef_2_gas",
    ]
    # Convert 'Folder' column to a categorical type with the specified order
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


def plot_data(data, title, x_label, y_label, file_path):
    """
    Plot data
    """
    custom_order = [
        "exogenous_0_ef",
        "endogenous_0_ef",
        "endogenous_0_9_ef",
        "endogenous_1_ef",
        "endogenous_1_1_ef",
        "endogenous_1_ef_2_gas",
    ]

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


def __main__():
    # Define the desired folder order

    data = load_csv("export/combined_costs.csv")
    plot_data(data, "Total Costs", "Year", "Cost (Billion EUR)", "total_costs.png")

    data = load_csv("export/combined_shadow_price.csv")
    plot_data(
        data, "Shadow Prices", "Year", "Shadow Price (EUR/MWh)", "shadow_prices.png"
    )

    data = load_csv("export/gas_supply.csv")
    plot_data(data, "Gas Supply", "Year", "Supply (MWh)", "gas_supply.png")

    data = load_csv("export/biomass_use_sum.csv")
    plot_data(data, "Biomass Use", "Year", "Use (MWh)", "biomass_use.png")

    data = load_csv("export/electricity_generation_share.csv")
    unique_data_names = data["Data Name"].unique()
    for data_name in unique_data_names:
        filtered_data = data[data["Data Name"] == data_name]
        title = f"{data_name}"
        output_file = f"{data_name.replace(' ', '_')}.png"
        plot_data(filtered_data, title, "Year", "Generation (MWh)", output_file)

    # Plot sum of all wind
    wind_data = data[data["Data Name"].str.contains("wind")]
    title = "Wind Power Generation"
    output_file = "wind_generation.png"
    wind_data = wind_data.groupby(["Year", "Folder"]).sum().reset_index()
    plot_data(wind_data, title, "Year", "Generation (MWh)", output_file)

    # Plot sum of all solar
    solar_data = data[data["Data Name"].str.contains("solar")]
    solar_rooftop = load_csv("export/solar_rooftop.csv")
    solar_data = pd.concat([solar_data, solar_rooftop], ignore_index=True)
    title = "Solar Power Generation"
    output_file = "solar_generation.png"
    solar_data = solar_data.groupby(["Year", "Folder"]).sum().reset_index()
    plot_data(solar_data, title, "Year", "Generation (MWh)", output_file)

    data = load_csv("export/biomass_use.csv")
    unique_data_names = data["Data Name"].unique()
    for data_name in unique_data_names:
        filtered_data = data[data["Data Name"] == data_name]
        title = f"{data_name}"
        output_file = f"{data_name.replace(' ', '_')}.png"
        plot_data(filtered_data, title, "Year", "Biomass Input in MWh", output_file)

    data = load_csv("export/biomass_supply.csv")
    unique_data_names = data["Data Name"].unique()
    for data_name in unique_data_names:
        filtered_data = data[data["Data Name"] == data_name]
        title = f"{data_name}"
        output_file = f"{data_name.replace(' ', '_')}.png"
        plot_data(filtered_data, title, "Year", "Biomass Supply in MWh", output_file)

    data = load_csv("export/EV_data.csv")
    unique_data_names = data["Data Name"].unique()
    for data_name in unique_data_names:
        filtered_data = data[data["Data Name"] == data_name]
        title = f"{data_name}"
        output_file = f"{data_name.replace(' ', '_')}.png"
        plot_data(filtered_data, title, "Year", data_name, output_file)

    data = load_csv("export/storage_data.csv")
    unique_data_names = data["Data Name"].unique()
    for data_name in unique_data_names:
        filtered_data = data[data["Data Name"] == data_name]
        title = f"{data_name}"
        output_file = f"{data_name.replace(' ', '_')}.png"
        plot_data(filtered_data, title, "Year", data_name, output_file)
    data = load_csv("export/oil_supply.csv")
    plot_data(data, "Oil Supply", "Year", "Supply (MWh)", "oil_supply.png")

    data = load_csv("export/biomass_prices.csv")
    plot_data(data, "Biomass Prices", "Year", "Price (EUR/MWh)", "biomass_prices.png")

    data = load_csv("export/mediumT_share.csv")
    plot_shares(
        data, "Medium Temperature Heat Industry", "Year", "Share", "mediumT_shares.png"
    )

    data = load_csv("export/highT_share.csv")
    plot_shares(
        data, "High Temperature Heat Industry", "Year", "Share", "highT_shares.png"
    )

    data = load_csv("export/lowT_share.csv")
    plot_shares(
        data, "Low Temperature Heat Industry", "Year", "Share", "lowT_shares.png"
    )

    data = load_csv("export/urban_central_heat_share.csv")
    plot_shares(
        data, "Urban Central Heat", "Year", "Share", "urban_central_heat_shares.png"
    )

    data = load_csv("export/urban_decentral_heat_share.csv")
    plot_shares(
        data, "Urban Decentral Heat", "Year", "Share", "urban_decentral_heat_shares.png"
    )

    data = load_csv("export/rural_heating_share.csv")
    plot_shares(data, "Rural Heating", "Year", "Share", "rural_heating_shares.png")

    data = load_csv("export/low_voltage_share_output.csv")
    plot_shares(
        data, "Low Voltage Output", "Year", "Share", "low_voltage_output_shares.png"
    )

    data = load_csv("export/solar_rooftop.csv")
    plot_data(data, "Solar Rooftop", "Year", "Generation (MWh)", "solar_rooftop.png")

    data = load_csv("export/electricity_generation_share.csv")
    plot_shares(
        data,
        "Electricity Generation Share (AC)",
        "Year",
        "Share",
        "electricity_generation_shares.png",
    )


if __name__ == "__main__":
    __main__()
