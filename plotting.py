# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
import os

import matplotlib as mpl
import matplotlib.patches as mpatches
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


def reorder_data(data, custom_order):
    """
    Reorder data based on custom order.
    """
    if custom_order is not None:
        data = data.assign(
            Folder=pd.Categorical(data["Folder"], categories=custom_order, ordered=True)
        )
        data = data.sort_values(by=["Folder", "Year"])
    return data


def plot_stacked_bar(
    df,
    title,
    x_label,
    y_label,
    file_path,
    custom_order=None,
    multiplier=1,
    remove_last_letters=0,
):
    """
    Plot a stacked bar chart for each folder (scenario) using the provided DataFrame.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the data.
    title : str
        Title of the plot.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    file_path : str
        Path to save the plot.
    custom_order : list, optional
        Custom order of folders.
    """
    if custom_order is not None:
        df = reorder_data(df, custom_order)

    if remove_last_letters != 0:
        df["Data Name"] = df["Data Name"].str[:-remove_last_letters]

    df["Values"] = df["Values"] * multiplier
    for row in df.iterrows():
        if row[1]["Share"] < 0.005:
            # remove the row
            df.drop(row[0], inplace=True)

    # Pivot the DataFrame for plotting
    pivot_df = df.pivot(index="Data Name", columns="Folder", values="Values").fillna(0)

    # Define x-axis positions and bar width
    x = np.arange(len(pivot_df.columns))  # Positions for each Folder
    bar_width = 0.5

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    bottom = np.zeros(len(pivot_df.columns))

    for data_name in pivot_df.index:
        ax.bar(x, pivot_df.loc[data_name], bar_width, label=data_name, bottom=bottom)
        bottom += pivot_df.loc[data_name]

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))  # Add 10% to the y-axis limit

    # Customise the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right")
    ax.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()


def plot_biomass_supply_with_carbon_captured(
    df, df_captured, title, x_label, y_label, file_path, custom_order=None
):
    if custom_order is not None:
        df = reorder_data(df, custom_order)

    # convert to TWh
    df["Values"] = df["Values"] / 1e6

    # I want to add the column
    df = df.merge(df_captured, on=["Folder"], how="left")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    bar_width = 0.8
    gap = 0.6  # Adjust this value to change the space between bars
    x_coords = np.arange(len(df)) * (bar_width + gap)

    color1 = "LightGreen"
    color2 = "DarkGreen"

    for index, row in df.iterrows():
        supply = row["Values"]
        captured = row["Share Removed"]
        ax.bar(x_coords[index], supply, width=bar_width, color=color1)
        ax.text(
            x_coords[index],
            supply + 0.02 * max(df["Values"]),
            f"{supply:.0f} TWh",
            ha="center",
            va="center",
            fontsize=14,
            color="black",
        )
        ax.bar(
            x_coords[index],
            supply * captured,
            width=bar_width,
            color=color2,  # , hatch="///"
        )
        # write the share in the center of the 2end bar
        ax.text(
            x_coords[index],
            supply * captured / 2,
            f"{captured * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=14,
            color="white",
        )

    # add legend
    supply_patch = mpl.patches.Patch(
        facecolor=color1, label="Share of biogenic carbon not captured"
    )
    captured_patch = mpl.patches.Patch(
        facecolor=color2, label="Share of biogenic carbon captured"
    )
    plt.legend(handles=[supply_patch, captured_patch], fontsize=14)

    ax.set_xticks(x_coords)
    ax.set_xticklabels(df["Folder"], rotation=45, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))  # Add 10% to the y-axis limit

    plt.tight_layout()
    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()


def plot_costs_vs_prices(df, title, x_label, y_label, file_path, scenario, usage_dict):
    """
    Plot biomass types with costs on the x-axis and values on the y-axis for a specific scenario,
    with circle fill indicating usage percentage.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing price and cost data.
    title : str
        Title of the plot.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    file_path : str
        Path to save the plot.
    scenario : str
        Scenario (folder) to filter the data.
    usage_dict : dict
        Dictionary mapping biomass types to their usage percentage (0 to 100).
    """
    # Filter data for the specified scenario
    scenario_df = df[(df["Folder"] == scenario)]

    # Filter for biomass types with available costs
    biomass_types = [
        "agricultural waste",
        "fuelwood residues",
        "secondary forestry residues",
        "sawdust",
        "residues from landscape care",
        "grasses",
        "woody crops",
        "fuelwoodRW",
        "C&P_RW",
        "manure",
        "sludge",
        "solid biomass import",
    ]

    biomass_df = scenario_df[
        (scenario_df["Data Name"].isin(biomass_types))
        & (scenario_df["Costs"].notnull())
    ]

    # Create export directory
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)

    # Set color palette
    palette = sns.color_palette("tab10", n_colors=len(biomass_types))
    color_mapping = {
        biomass: palette[i % len(palette)] for i, biomass in enumerate(biomass_types)
    }

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal")  # Ensure equal scaling for both axes

    # Add diagonal line from (0, 0) without adding to the legend
    max_limit = (
        max(biomass_df["Costs"].max(), biomass_df["Values"].max()) * 1.1
    )  # Add 10% headroom
    ax.plot(
        [0, max_limit],
        [0, max_limit],
        color="grey",
        linestyle="--",
        zorder=0,
        label="_nolegend_",
    )

    for biomass in biomass_types:
        subset = biomass_df[biomass_df["Data Name"] == biomass]
        if not subset.empty:
            for _, row in subset.iterrows():
                usage = usage_dict.get(biomass, 0)  # Default to 0 if not provided
                color = color_mapping[biomass]

                if usage == 0:
                    # Empty circle (just outline)
                    ax.scatter(
                        row["Costs"],
                        row["Values"],
                        s=100,
                        facecolors="none",
                        edgecolors=color,
                        label=biomass,
                    )
                elif usage == 100:
                    # Fully filled circle
                    ax.scatter(
                        row["Costs"],
                        row["Values"],
                        s=100,
                        color=color,
                        label=biomass,
                        alpha=0.8,
                    )
                else:
                    # Partially filled circle
                    theta1 = 90
                    theta2 = 90 - 360 * (usage / 100)
                    wedge = mpatches.Wedge(
                        (row["Costs"], row["Values"]),
                        1,
                        theta2,
                        theta1,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.8,
                    )
                    ax.add_patch(wedge)
                    ax.scatter(
                        row["Costs"],
                        row["Values"],
                        s=100,
                        facecolors="none",
                        edgecolors=color,
                        label=biomass,
                        alpha=0.8,
                    )

    # Add legend before the diagonal line
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        title="Biomass Types",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0,
    )

    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} ({scenario})", loc="left")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0.3)
    plt.close()


def plot_feedstock_prices(df, title, x_label, y_label, file_path, custom_order=None):
    """
    Plot feedstock prices with aggregated categories and price ranges for each scenario (folder).

    Parameters
    ----------
    df : DataFrame
        The dataframe containing price data.
    title : str
        Title of the plot.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    file_path : str
        Path to save the plot.
    custom_order : list, optional
        Custom order of feedstocks.
    axis2_ticks : int, optional
        Ticks for the secondary axis.
    """
    # Remove unwanted feedstocks
    df = df[~df["Data Name"].isin(["biogas", "solid biomass"])]

    # Define categories
    digestible_biomass = ["manure", "sludge"]
    solid_biomass = [
        "agricultural waste",
        "fuelwood residues",
        "secondary forestry residues",
        "sawdust",
        "residues from landscape care",
        "grasses",
        "woody crops",
        "fuelwoodRW",
        "C&P_RW",
    ]

    # Plotting for each scenario (folder)
    folders = df["Folder"].unique()
    feedstocks = [
        "solid biomass",
        "digestible biomass",
        "liquid fuels",
        "primary oil",
        "gas",
    ]

    # Set color palette
    palette = sns.color_palette("Set2", n_colors=len(folders))

    fig, ax = plt.subplots(figsize=(14, 8))

    # Adjust position for each scenario within each feedstock
    width = 0.15  # Bar width for spacing
    x = np.arange(len(feedstocks))  # Base positions

    for idx, scenario in enumerate(folders):
        scenario_df = df[df["Folder"] == scenario]

        # Aggregate prices
        aggregated_data = {
            "solid biomass": scenario_df.loc[
                scenario_df["Data Name"].isin(solid_biomass), "Values"
            ].values,
            "digestible biomass": scenario_df.loc[
                scenario_df["Data Name"].isin(digestible_biomass), "Values"
            ].values,
            "liquid fuels": scenario_df.loc[
                scenario_df["Data Name"] == "oil", "Values"
            ].values,
            "primary oil": scenario_df.loc[
                scenario_df["Data Name"] == "oil primary", "Values"
            ].values,
            "gas": scenario_df.loc[scenario_df["Data Name"] == "gas", "Values"].values,
        }

        # Apply custom order if provided
        if custom_order:
            aggregated_data = {
                key: aggregated_data[key]
                for key in custom_order
                if key in aggregated_data
            }

        # Plotting with horizontal shifts
        for i, (key, values) in enumerate(aggregated_data.items()):
            shift = x[i] + (idx - len(folders) / 2) * width

            if len(values) > 0:
                if key in ["solid biomass", "digestible biomass"]:
                    # Plot individual dots for each type
                    ax.plot(
                        [shift] * len(values),
                        values,
                        "o",
                        markersize=8,
                        color=palette[idx],
                        label=scenario if i == 0 else "",
                    )
                    # Plot range line
                    ax.plot(
                        [shift, shift], [values.min(), values.max()], color=palette[idx]
                    )
                else:
                    # Plot single value
                    ax.plot(
                        shift,
                        values[0],
                        "o",
                        markersize=8,
                        color=palette[idx],
                        label=scenario if i == 0 else "",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(feedstocks)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y")

    # Add legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Scenarios")

    plt.tight_layout()
    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()


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
        df = reorder_data(df, custom_order)

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
        absolute_text = f"Δ {value / 1e6:.0f}"
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
        df = reorder_data(df, custom_order)

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
        df = reorder_data(df, custom_order)

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


def plot_costs(df, title, x_label, y_label, file_path):
    """
    Plot costs
    """
    # convert to billion
    df["Difference"] = df["Difference"] / 1e9
    # remove all have an absolute value less than 1
    df = df[df["Difference"].abs() > 1]
    plt.rcParams.update({"font.size": 14})
    # Recreate the bar chart with the reordered folders
    plt.figure(figsize=(12, 6))

    # Create a gradient color palette based on the values
    norm = plt.Normalize(df["Difference"].min(), df["Difference"].max())
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])

    # Apply the color mapping to the bars
    colors = df["Difference"].apply(lambda x: sm.to_rgba(x)).tolist()

    ax = sns.barplot(
        data=df, x="Year", y="Difference", hue="Data Name", palette=colors, orient="v"
    )
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))

    # Add values as labels above the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=12, padding=5)

    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title="Data Name", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Show the plot
    plt.tight_layout()

    # Save the plot to a file
    export_dir = "export/plots"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()


def plot_data(
    data, title, x_label, y_label, file_path, custom_order=None, color_palette="viridis"
):
    """
    Plot data
    """
    if custom_order is not None:
        reorder_data(data, custom_order)
    plt.rcParams.update({"font.size": 14})
    # Recreate the bar chart with the reordered folders
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=data, x="Year", y="Values", hue="Folder", palette=color_palette, orient="v"
    )
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))

    # Add values as labels above the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=12, padding=5)

    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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
    df["Data Name"] = df["Data Name"].str.replace("1", "")
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
        "C&P_RW": 0.144,
        "solid biomass import": 0.3667 * 0.1,
    }

    biomass_potentials_TWh = {
        "agricultural waste": 306,
        "fuelwood residues": 541.7,
        "fuelwoodRW": 75.4,
        "grasses": 504,
        "manure": 338.1,
        "municipal solid waste": 151.2,
        # "not included": 398.5,
        "residues from landscape care": 74.7,
        "sawdust": 32.4,
        "secondary forestry residues": 94.3,
        "sludge": 9.2,
        "woody crops": 117.3,
        "solid biomass import": 0,
        "C&P_RW": 576.5,
        # "unsustainable solid biomass": 0,
        # "unsustainable biogas": 0,
        # "unsustainable bioliquids": 0
    }

    custom_order = [
        "woody crops",
        "grasses",
        "fuelwoodRW",
        "C&P_RW",
        "secondary forestry residues",
        "sawdust",
        "fuelwood residues",
        "agricultural waste",
        "residues from landscape care",
        "sludge",
        "manure",
        "solid biomass import",
    ]

    df = df[df["Year"] == year]
    # Conversion factors
    mwh_to_twh = 1e-6
    mwh_to_pj = 3.6e-6

    # Pivot the table for easier plotting
    df_pivot = df.pivot_table(
        index="Data Name", columns="Folder", values="Values"
    ).reset_index()

    # # Reorder according to custom order
    # df_pivot["order"] = df_pivot["Data Name"].map(lambda x: custom_order.index(x) if x in custom_order else float('inf'))
    # df_pivot = df_pivot.sort_values(by="order").drop(columns=["order"])

    # Reorder data according to custom order
    df_pivot = df_pivot.set_index("Data Name").reindex(custom_order).reset_index()

    for folder in df_pivot.columns[1:]:
        df_pivot[folder] = df_pivot[folder] * mwh_to_twh

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    bar_width = 0.8
    gap = 0.6  # Adjust this value to change the space between bars
    x_coords = np.arange(len(df_pivot)) * (bar_width + gap)

    for index, row in df_pivot.iterrows():
        biomass_type = row["Data Name"]
        value_a = row["default"]
        value_b = row["biomass_emissions"]
        potential = biomass_potentials_TWh.get(biomass_type, 0)
        if value_a - value_b > -1:
            color = "LightGreen"
            # Base bar
            ax.bar(
                x_coords[index],
                value_a,
                width=bar_width,
                label="without biomass emissions" if index == 0 else "",
                color=color,
                hatch="///",
            )
            ax.bar(
                x_coords[index],
                value_b,
                width=bar_width,
                label="with biomass emissions" if index == 0 else "",
                color=color,
            )
        else:
            color = "red"
            ax.bar(
                x_coords[index],
                value_b,
                width=bar_width,
                label="with biomass emissions" if index == 0 else "",
                color=color,
                hatch="///",
            )
            # Base bar
            ax.bar(
                x_coords[index],
                value_a,
                width=bar_width,
                label="without biomass emissions" if index == 0 else "",
                color=color,
            )

        # Add potential outline (black box)
        ax.bar(
            x_coords[index],
            potential,
            width=bar_width,
            edgecolor="black",
            facecolor="none",
            linewidth=1.5,
            label="Potential" if index == 0 else "",
        )

        # Add emission factor below the x-axis
        emission_factor = emission_factors.get(biomass_type, "N/A")
        ef_g_per_MJ = round(emission_factor / 0.0036)
        if biomass_type == "solid biomass import":
            potential = value_b
        ax.text(
            x_coords[index],
            potential + 1,
            f"{emission_factor} | {ef_g_per_MJ} \n ton/MWh | g/MJ",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add legend
    potential_patch = mpl.patches.Patch(
        edgecolor="black",
        facecolor="none",
        label="Biomass Potential",
        linewidth=1.5,
    )
    a_patch = mpl.patches.Patch(
        facecolor="LightGreen", label="Biomass use - no emissions considered"
    )
    b_patch = mpl.patches.Patch(
        facecolor="none",
        hatch="///",
        label="Difference in biomass use - emissions considered",
    )
    red_patch = mpl.patches.Patch(
        facecolor="red",
        hatch="///",
        label="More biomass used with emissions considered",
    )
    plt.legend(handles=[potential_patch, a_patch, b_patch, red_patch], fontsize=14)

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.3))  # Add 10% to the y-axis limit

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
        "woody crops": 0.18,
        "grasses": 0.216,
        "fuelwoodRW": 0.288,
        "C&P_RW": 0.144,
        "secondary forestry residues": 0.144,
        "sawdust": 0.108,
        "fuelwood residues": 0.036,
        "agricultural waste": 0.108,
        "residues from landscape care": 0,
        "sludge": 0,
        "manure": 0.072,
        "imported biomass": 0.3667 * 0.1,
    }

    biomass_costs = {  # Euro/MWh_LHV
        "agricultural waste": 12.8786,
        "fuelwood residues": 15.3932,
        "fuelwoodRW": 12.6498,
        "manure": 22.1119,
        "residues from landscape care": 10.5085,
        "secondary forestry residues": 8.1876,
        "coal": 9.5542,
        "fuelwood": 14.5224,
        "gas": 24.568,
        "oil": 52.9111,
        "woody crops": 44.4,
        "grasses": 18.9983,
        "sludge": 22.0995,
        "imported biomass": 54,
        "sawdust": 6.4791,
        "C&P_RW": 25.4661,
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
        # Add cost as text above the bar
        cost = biomass_costs.get(biomass_type, "N/A")
        ax.text(
            x_coords[index],
            ef + 0.01,  # Position the text slightly above the bar
            f"{cost:.2f}\n€/MWh" if cost != "N/A" else "N/A",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )

    # Add gas emission factor
    x_coords = np.append(x_coords, [x_coords[-1] + (bar_width + gap)])
    ax.bar(x_coords[-1], 0.198, width=bar_width, label="natural gas", color="Grey")
    ax.text(
        x_coords[-1],
        0.198 + 0.01,
        f"{biomass_costs['gas']:.2f}\n€/MWh",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
    )

    # Add oil emission factor
    x_coords = np.append(x_coords, [x_coords[-1] + (bar_width + gap)])
    ax.bar(x_coords[-1], 0.2571, width=bar_width, label="oil", color="Brown")
    ax.text(
        x_coords[-1],
        0.2571 + 0.01,
        f"{biomass_costs['oil']:.2f}\n€/MWh",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
    )

    # Add coal emission factor
    x_coords = np.append(x_coords, [x_coords[-1] + (bar_width + gap)])
    ax.bar(x_coords[-1], 0.3361, width=bar_width, label="coal", color="Black")
    ax.text(
        x_coords[-1],
        0.3361 + 0.01,
        f"{biomass_costs['coal']:.2f}\n€/MWh",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
    )

    ax.set_xticks(x_coords)
    # ax.set_xticklabels(emission_factors.keys(), rotation=45, ha="right")
    ax.set_xticklabels(
        list(emission_factors.keys()) + ["natural gas", "oil", "coal"],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("")
    ax.set_ylabel("tonCO2/MWh")
    ax.set_title("Emission factors for different feedstocks", fontsize=26)

    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks() / 0.0036)
    ax2.set_ylabel("g/MJ")

    secondary_locator = MultipleLocator(10)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))

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
    remove_letters=None,
    axis2_ticks=500,
    include_total=True,
):
    # plots the data in a bar chart and adds a sum of all values at the end
    # Pivot the DataFrame for plotting

    if custom_order is not None:
        df = reorder_data(df, custom_order)

    if remove_letters is not None:
        for replace_letter in remove_letters:
            df["Data Name"] = df["Data Name"].str.replace(f"{replace_letter}", "")

    # Calculate the total values for each scenario
    if include_total:
        total_values = (
            df.groupby("Folder", observed=False)["Values"].sum().reset_index()
        )
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


def get_usage_dict(df, scenario, year=2050):
    potentials = {  # TWh
        "agricultural waste": 306,
        "fuelwood residues": 541.7,
        "fuelwoodRW": 75.4,
        "grasses": 504,
        "manure": 338.1,
        "municipal solid waste": 151.2,
        "residues from landscape care": 74.7,
        "sawdust": 32.4,
        "secondary forestry residues": 94.3,
        "sludge": 9.2,
        "woody crops": 117.3,
        "solid biomass import": 1390,
        "C&P_RW": 576.5,
    }
    # remove 1 from data_name
    df["Data Name"] = df["Data Name"].str.replace("1", "")
    # filter year
    df = df[df["Year"] == year]
    usage = df[df["Folder"] == scenario].set_index("Data Name")["Values"]
    # convert to TWh
    usage = usage / 1e6
    usage_dict = usage.to_dict()
    # divide by potential to and multiply by 100 to get percentage
    for key in usage_dict:
        usage_value = usage_dict[key] / potentials[key] * 100
        if usage_value > 99.5:
            usage_value = 100
        elif usage_value < 0.5:
            usage_value = 0
        usage_dict[key] = usage_value
    return usage_dict


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
    custom_order = ["default", "biomass_emissions"]

    data = load_csv("export/costs2050.csv")
    plot_data(
        data,
        "Total Costs",
        "Year",
        "Cost (Billion EUR)",
        "total_costs.png",
        custom_order,
    )

    data = load_csv("export/biomass_supply.csv")
    plot_biomass_use(data, "Biomass Use", "", "TWh", "biomass_supply.png")

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
        remove_letters=[1],
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

    data = load_csv("export/biomass_use_2050.csv")
    plot_bar_with_shares(
        data,
        "Solid Biomass Use in 2050",
        "",
        "TWh",
        "biomass_use_2050.png",
        custom_order,
        axis2_ticks=500,
        width=10,
    )

    data = load_csv("export/biomass_use_by_sector_2050.csv")
    plot_bar_with_shares(
        data,
        "Biomass Use by Sector in 2050",
        "",
        "TWh",
        "biomass_use_by_sector_2050.png",
        custom_order,
        axis2_ticks=500,
        width=10,
    )

    data = load_csv("export/fossil_fuel_supply.csv")
    plot_bar_with_totals(
        data,
        "Fossil Fuel Supply",
        "",
        "TWh",
        "fossil_fuel_supply.png",
        custom_order,
        include_total=False,
    )

    data = load_csv("export/cost_difference.csv")
    plot_costs(
        data,
        "Extra Costs Due to Biomass Emissions (bigger than 1 Billion EUR)",
        "",
        "Cost (Billion EUR)",
        "cost_difference.png",
    )

    data = load_csv("export/shadow_price_2050.csv")
    plot_data(
        data, "Shadow Prices", "Year", "Shadow Price (EUR/tonCO2)", "shadow_prices.png"
    )

    data = load_csv("export/hydrogen_production_2050.csv")
    plot_bar_with_totals(
        data,
        "Hydrogen Production",
        "",
        "TWh",
        "hydrogen_production_2050.png",
        custom_order,
        remove_letters=[1],
        axis2_ticks=500,
        include_total=True,
    )

    data = load_csv("export/heat_pumps_2050.csv")
    plot_bar_with_totals(
        data,
        "Heat Pump Electricity Consumption",
        "",
        "TWh",
        "heat_pumps_2050.png",
        custom_order,
        remove_letters=[0],
        axis2_ticks=500,
        include_total=True,
    )

    data = load_csv("export/gas_use_2050.csv")
    plot_bar_with_totals(
        data,
        "(Bio)Gas Use",
        "",
        "TWh",
        "gas_use_2050.png",
        custom_order,
        axis2_ticks=500,
        include_total=False,
    )

    supply_data = load_csv("export/biomass_supply.csv")
    usage_dict_default = get_usage_dict(supply_data, "default")
    usage_dict_biomass_emissions = get_usage_dict(supply_data, "biomass_emissions")

    data = load_csv("export/weighted_prices_2050.csv")
    plot_feedstock_prices(
        data,
        "Weighted Feedstock Prices in 2050",
        "",
        "EUR/MWh",
        "weighted_feedstock_prices_2050.png",
    )
    plot_costs_vs_prices(
        data,
        "Weighted Feedstock Prices vs. Costs in 2050",
        "Costs in Euro/MWh",
        "Prices in EUR/MWh",
        "prices_costs_default_2050.png",
        scenario="default",
        usage_dict=usage_dict_default,
    )
    plot_costs_vs_prices(
        data,
        "Weighted Feedstock Prices vs. Costs in 2050",
        "Costs in Euro/MWh",
        "Prices in EUR/MWh",
        "prices_costs_biomass_emissions_2050.png",
        scenario="biomass_emissions",
        usage_dict=usage_dict_biomass_emissions,
    )

    data1 = load_csv("export/all_biomass_supply.csv")
    data2 = load_csv("export/carbon_removal.csv")
    plot_biomass_supply_with_carbon_captured(
        data1,
        data2,
        "Biomass Supply",
        "",
        "TWh",
        "biomass_supply_and_carbon_captured.png",
    )

    data = load_csv("export/co2_use.csv")
    plot_stacked_bar(
        data,
        "CO2 Use",
        "",
        "MtCO2",
        "co2_use.png",
        custom_order,
        multiplier=1e-6,
        remove_last_letters=1,
    )
    data = load_csv("export/co2_capture.csv")
    plot_stacked_bar(
        data,
        "CO2 Capture",
        "",
        "MtCO2",
        "co2_capture.png",
        custom_order,
        multiplier=1e-6,
        remove_last_letters=1,
    )


if __name__ == "__main__":
    __main__()
