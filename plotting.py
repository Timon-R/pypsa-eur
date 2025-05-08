# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
import os

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerPatch
from matplotlib.path import Path
from matplotlib.ticker import MultipleLocator

from result_analysis import get_emission_factors, get_biomass_potentials

# config_file_path = "config/config.yaml"

# # Open and load the YAML file
# with open(config_file_path, "r") as file:
#     config = yaml.safe_load(file)

# # Extract biomass types
# biomass_types = list(config["biomass"]["classes"].keys())

emission_factors = get_emission_factors(add_imported_biomass=True)
emission_factors_new_names = get_emission_factors(new_names=True, add_imported_biomass=True)
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

biomass_potentials_TWh = {
    "agricultural waste": 290.4185038259868,
    "fuelwood residues": 547.1662388608518,
    "secondary forestry residues": 87.15159435646442,
    "sawdust": 30.14011967737334,
    "residues from landscape care": 70.67024100998525,
    "grasses": 472.9656047672824,
    "woody crops": 111.53964182929641,
    "fuelwoodRW": 86.6139452513965,
    "C&P_RW": 666.8196265469048,
    "not included": 0.0,
    "manure": 345.45933182016194,
    "sludge": 13.908196004274894,
    "solid biomass import": 1390,
}

biomass_costs = {  # Euro/MWh_LHV
    "agricultural waste": 12.8786,
    "fuelwood residues": 15.3932,
    "fuelwoodRW": 12.6498,
    "manure": 22.1119,
    "residues from landscape care": 10.5085,
    "secondary forestry residues": 8.1876,
    "woody crops": 44.4,
    "grasses": 18.9983,
    "sludge": 22.0995,
    "solid biomass import": 54,
    "sawdust": 6.4791,
    "C&P_RW": 25.4661,
}

def configure_for_pgf():
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",  # or 'xelatex'
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })

def load_csv(file_path, folder_path="export"):
    """
    Load a CSV file and return the data as a pandas DataFrame.

    Parameters
    ----------
    file_path (str): The path to the CSV file to load.

    Returns
    -------
    pd.DataFrame: The loaded data.
    """
    file_path = os.path.join(folder_path, file_path)
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

    # Create a custom elliptical wedge for the filled portion


def create_elliptical_wedge(
    center_x, center_y, width, height, theta1, theta2, num_points=100
):
    # Convert angles from degrees to radians
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)

    # Create the angle array
    theta = np.linspace(theta2_rad, theta1_rad, num_points)

    # Create the points on the arc
    x = center_x + width * np.cos(theta)
    y = center_y + height * np.sin(theta)

    # Add the center point at the beginning
    x = np.insert(x, 0, center_x)
    y = np.insert(y, 0, center_y)

    # Create the vertices
    vertices = np.column_stack([x, y])

    # Create the codes
    codes = [Path.MOVETO] + [Path.LINETO] * (num_points - 1) + [Path.CLOSEPOLY]

    return Path(vertices, codes)


def create_gravitational_plot(
    title,
    file_name,
    multiplier=1e-6,
    biomass_supply=None,
    scenario=None,
    export_dir="export/plots",
    file_type="png", 
):

    file_path = f"{file_name}.{file_type}"


    if biomass_supply is not None and scenario is not None:
        biomass_supply = biomass_supply[biomass_supply["Folder"] == scenario]
        # remove the 1 from the data_name
        biomass_supply.loc[:, "Data Name"] = biomass_supply["Data Name"].str.replace(
            "1", ""
        )

    # Extract data for plotting
    biomass_types = list(emission_factors.keys())
    emissions = [emission_factors[bt] for bt in biomass_types]
    costs = [biomass_costs[bt] for bt in biomass_types]
    potentials = [biomass_potentials_TWh[bt] for bt in biomass_types]

    # Normalize potentials for circle sizes
    max_potential = max(potentials)
    sizes = [max_potential * (p / max_potential) for p in potentials]

    fig, ax = plt.subplots(figsize=(10, 6))

    dig_biomass_color = "blue"
    solid_biomass_color = "green"

    # Draw the plot first to get the limits
    for i, bt in enumerate(biomass_types):
        if bt in ["manure", "sludge"]:
            color = dig_biomass_color
        else:
            color = solid_biomass_color
        plt.scatter(
            costs[i],
            emissions[i],
            s=sizes[i],
            alpha=1,
            facecolors="none",
            edgecolors=color,
            linewidth=1,
        )
        location = emissions[i] + 2 * sizes[i] / max_potential * 0.015 + 0.01
        if (
            bt == "secondary forestry residues"
        ):  # below the point
            location = emissions[i] - 2 * sizes[i] / max_potential * 0.015 - 0.015
        plt.text(
            costs[i],
            location,
            new_names_dict[bt],
            fontsize=9,
            ha="center",
        )

    # Set up the axes and draw to ensure limits are calculated
    plt.xlabel("Costs in Euro/MWh_LHV")
    plt.ylabel("Emission Factors in tonCO2/MWh")
    plt.title(title)
    plt.xlim(0, max(costs) + 5)
    plt.ylim(-0.02, max(emissions) + 0.05)
    fig.canvas.draw()

    # Get the actual data ratio
    data_ratio = ax.get_data_ratio()

    # Get the physical dimensions ratio
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_inches, height_inches = bbox.width, bbox.height
    physical_ratio = height_inches / width_inches

    # Calculate the combined adjustment factor
    adjustment_factor = data_ratio / physical_ratio

    # Now draw the wedges with the proper adjustment
    for i, bt in enumerate(biomass_types):
        if bt in ["manure", "sludge"]:
            color = dig_biomass_color
        else:
            color = solid_biomass_color
        if biomass_supply is not None and bt in biomass_supply["Data Name"].values:
            supply = (
                biomass_supply[biomass_supply["Data Name"] == bt]["Values"].values[0]
                * multiplier
            )
            potential = biomass_potentials_TWh[bt]
            usage = supply / potential * 100
            if usage > 99:
                usage = 100
            theta1 = 90
            theta2 = 90 - 360 * (usage / 100)

            # Correctly convert from area (sizes[i]) to radius, matching the scatter plot circles
            # The scatter plot uses s=area, so we need sqrt(sizes[i]/pi) to get equivalent radius
            circle_radius = np.sqrt(sizes[i] / np.pi)
            width = circle_radius * 0.09  # Scale factor for visual appearance
            height = width * adjustment_factor

            if usage >= 99.5:  # Special case for (nearly) 100% usage
                # Draw a filled ellipse instead of a wedge
                ellipse = mpatches.Ellipse(
                    (costs[i], emissions[i]),
                    width=width * 2,  # Diameter = 2*radius
                    height=height * 2,
                    facecolor=color,
                    edgecolor="none",
                    alpha=1,
                )
                ax.add_patch(ellipse)
            else:
                # Normal case: draw a wedge
                theta1 = 90
                theta2 = 90 - 360 * (usage / 100)
                wedge_path = create_elliptical_wedge(
                    costs[i], emissions[i], width, height, theta1, theta2
                )
                wedge_patch = mpatches.PathPatch(
                    wedge_path, facecolor=color, edgecolor="none", alpha=1
                )
                ax.add_patch(wedge_patch)

    # Add legend for circle sizes
    for size in [100, 200, 500]:  # Example sizes
        plt.scatter(
            [],
            [],
            s=max_potential * (size / max_potential),
            edgecolors="black",
            facecolors="none",
            label=f"{size} TWh",
        )
    plt.legend(
        scatterpoints=1,
        frameon=False,
        labelspacing=1,
        title="Potential",
        loc="lower right",
        bbox_to_anchor=(1, 0),
    )

    # # Add legend for biomass types
    # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor=colors[i], markersize=10, label=bt) for i, bt in enumerate(biomass_types)]
    # plt.legend(handles=handles, title='Biomass Types', bbox_to_anchor=(1.05, 1), loc='upper left')

    # add some space to the top and left
    plt.xlim(0, max(costs) + 5)
    plt.ylim(-0.02, max(emissions) + 0.05)

    if file_path.endswith(".pgf"):
        configure_for_pgf()

    # Save the plot to a file
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()

    print(f"Gravitional plot saved to {file_path}")


def plot_stacked_bar(
    df,
    title,
    x_label,
    y_label,
    file_name,
    custom_order=None,
    multiplier=1,
    remove_last_letters=0,
    column="Values",
    columns="Folder",
    index="Data Name",
    threshold=0.005,
    threshold_column="Share",
    export_dir="export/plots",
    file_type="png",
    no_xticks=False,
):
    file_path = f"{file_name}.{file_type}"
    # rename data_name column to Data Name
    df.rename(columns={"data_name": "Data Name"}, inplace=True)
    if custom_order is not None:
        df = reorder_data(df, custom_order)

    if remove_last_letters != 0:
        df["Data Name"] = df["Data Name"].str[:-remove_last_letters]

    df[column] = df[column] * multiplier
    for row in df.iterrows():
        if abs(row[1][threshold_column]) < threshold:
            # remove the row
            df.drop(row[0], inplace=True)

    if index is None:
        df["Index"] = range(len(df))
        index = "Index"
    pivot_df = df.pivot(index=index, columns=columns, values=column).fillna(0)

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

    # no x-ticks
    if no_xticks:
        ax.set_xticks([])
        ax.set_xticklabels([])

    if file_path.endswith(".pgf"):
        configure_for_pgf()

    plt.tight_layout()
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()

    print(f"Stacked bar plot saved to {file_path}")


def plot_BECCUS(
    df,
    title,
    x_label,
    y_label,
    file_name,
    custom_order=None,
    multiplier=1e-6,
    upstream_data=None,
    export_dir="export/plots",
    file_type="png",
):
    file_path = f"{file_name}.{file_type}"

    if custom_order is not None:
        df = reorder_data(df, custom_order)

    df["Not Captured"] = (
        df["Total Carbon"] - df["Carbon Stored"] - df["Carbon Utilised"]
    )

    # Select only the relevant columns for plotting
    df_plot = df[["Folder", "Carbon Stored", "Carbon Utilised", "Not Captured"]]

    if upstream_data is not None:
        upstream_total = upstream_data[upstream_data["data_name"] == "total"].copy()
        upstream_total["Upstream Emissions"] = (
            upstream_total["upstream emissions"] * multiplier
        )
        upstream_total["Folder"] = upstream_total["folder"]
        upstream_total = upstream_total[["Folder", "Upstream Emissions"]]

        # Merge upstream_total with df_plot
        df_plot = df_plot.merge(upstream_total, on="Folder", how="left")

    # Convert the values to MtCO2
    df_plot["Biogenic Carbon Sequestered"] = df_plot["Carbon Stored"] * multiplier
    df_plot["Biogenic Carbon Utilised"] = df_plot["Carbon Utilised"] * multiplier
    df_plot["Biogenic Carbon Not Captured"] = df_plot["Not Captured"] * multiplier

    # Set the index to "Folder" for plotting
    df_plot.set_index("Folder", inplace=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    bar_width = 0.35  # Width of the bars
    x = np.arange(len(df_plot.index))  # Positions for each Folder

    # Plot the stacked bars for carbon data
    bottom = np.zeros(len(df_plot.index))
    for column in [
        "Biogenic Carbon Sequestered",
        "Biogenic Carbon Utilised",
        "Biogenic Carbon Not Captured",
    ]:
        ax.bar(
            x - bar_width / 2, df_plot[column], bar_width, label=column, bottom=bottom
        )
        bottom += df_plot[column]

    if upstream_data is not None:
        # Plot the upstream emissions bars next to the stacked bars
        ax.bar(
            x + bar_width / 2,
            df_plot["Upstream Emissions"],
            bar_width,
            label="Upstream Emissions",
            color="grey",
            alpha=0.6,
        )

    # Add labels and title
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=45, ha="right")

    # Add legend
    ax.legend(title="Legend", fontsize=16)

    # Add percentages on the bars
    for container in ax.containers[:-1]:
        for i, bar in enumerate(container):
            height = bar.get_height()
            total = sum([c[i].get_height() for c in ax.containers[:-1]])
            percentage = height / total * 100
            if percentage < 4:
                color = "black"
            else:
                color = "white"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height / 2,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=12,
                color=color,
            )

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))  # Add 10% to the y-axis limit

    plt.tight_layout()
    # Save the plot to a file

    if file_path.endswith(".pgf"):
        configure_for_pgf()
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()

    print(f"BECCUS plot saved to {file_path}")


class HandlerWedge(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        center = 0.5 * width, 0.5 * height
        p = mpatches.Wedge(
            center,
            0.5 * min(width, height),
            orig_handle.theta1,
            orig_handle.theta2,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def plot_costs_vs_prices(df, title, x_label, y_label, file_name, scenario, usage_dict,export_dir="export/plots",file_type="png"):
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
    file_name : str
        Name to save the plot.
    scenario : str
        Scenario (folder) to filter the data.
    usage_dict : dict
        Dictionary mapping biomass types to their usage percentage (0 to 100).
    """
    file_path = f"{file_name}.{file_type}"
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

    legend_handles = []

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
                elif usage >= 99:
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
                        1.9,
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

                # Add to legend handles
                if usage == 0:
                    legend_handles.append(
                        mlines.Line2D(
                            [],
                            [],
                            marker="o",
                            linestyle="None",
                            markersize=10,
                            markerfacecolor="none",
                            markeredgecolor=color,
                            label=biomass,
                        )
                    )
                elif usage >= 99:
                    legend_handles.append(
                        mlines.Line2D(
                            [],
                            [],
                            marker="o",
                            linestyle="None",
                            markersize=10,
                            markerfacecolor=color,
                            markeredgecolor=color,
                            label=biomass,
                        )
                    )
                else:
                    legend_handles.append(
                        mpatches.Wedge(
                            (0, 0),
                            1,
                            theta2,
                            theta1,
                            facecolor=color,
                            edgecolor=color,
                            alpha=0.8,
                            label=biomass,
                        )
                    )

    # Add legend before the diagonal line
    by_label = {handle.get_label(): handle for handle in legend_handles}
    ax.legend(
        by_label.values(),
        by_label.keys(),
        title="Biomass Types",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0,
        handler_map={mpatches.Wedge: HandlerWedge()},
    )

    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} ({scenario})", loc="left")
    ax.grid(True)

    if file_path.endswith(".pgf"):
        configure_for_pgf()

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    print(f"Costs vs Prices plot saved to {file_path}")


def plot_feedstock_prices(df, title, x_label, y_label, file_name, custom_order=None, export_dir="export/plots",file_type="png"):
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
    file_name : str
        Path to save the plot.
    custom_order : list, optional
        Custom order of feedstocks.
    axis2_ticks : int, optional
        Ticks for the secondary axis.
    """
    file_path = f"{file_name}.{file_type}"
    # Remove unwanted feedstocks
    df = df[~df["Data Name"].isin(["solid biomass"])]

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
        # "biogas",
        # "liquid fuels",
        # "primary oil",
        # "natural gas",
    ]

    # Set color palette
    palette = sns.color_palette("Set2", n_colors=len(folders))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Adjust position for each scenario within each feedstock
    width = 0.3  # Bar width for spacing
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
            # "biogas": scenario_df.loc[
            #     scenario_df["Data Name"] == "biogas", "Values"
            # ].values,
            # "liquid fuels": scenario_df.loc[
            #     scenario_df["Data Name"] == "oil", "Values"
            # ].values,
            # "primary oil": scenario_df.loc[
            #     scenario_df["Data Name"] == "oil primary", "Values"
            # ].values,
            # "natural gas": scenario_df.loc[
            #     scenario_df["Data Name"] == "gas", "Values"
            # ].values,
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

    # Adjust x-tick positions to be between the two bars
    x_ticks = x + (width * (len(folders) - 1) / 2)
    ax.set_xticks(x_ticks - width)
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

    if file_path.endswith(".pgf"):
        configure_for_pgf()
    # Save the plot to a file
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()

    print(f"Feedstock prices plot saved to {file_path}")


def plot_difference_bar(
    df,
    title,
    x_label,
    y_label,
    file_name,
    custom_order=None,
    remove_last_letters=0,
    axis2_ticks=500,
    export_dir="export/plots",
    file_type="png",
):
    
    file_path = f"{file_name}.{file_type}"
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
    ax.set_xticks()
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks() * 3.6)
    ax2.set_ylabel("PJ")

    secondary_locator = MultipleLocator(axis2_ticks)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    plt.tight_layout()
    # Save the plot to a file
    if file_path.endswith(".pgf"):
        configure_for_pgf()
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()

    print(f"Difference bar plot saved to {file_path}")


def plot_bar_with_shares(
    df,
    title,
    x_label,
    y_label,
    file_name,
    custom_order=None,
    axis2_ticks=500,
    remove_last_letters=0,
    width=14,
    threshold=0.001,
    export_dir="export/plots",
    file_type="png",
):
    file_path = f"{file_name}.{file_type}"
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
    bar_width = 0.35*2/len(custom_order)

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
                fontsize=8/(len(custom_order)/2),
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

    if file_path.endswith(".pgf"):
        configure_for_pgf()
    # Save the plot to a file
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()

    print(f"Bar plot with shares saved to {file_path}")


def plot_shares(df, title, x_label, y_label, file_name, custom_order=None, export_dir="export/plots",file_type="png"):
    """
    Creates a stacked bar chart based on the provided DataFrame and saves it to file.

    Args:
        df (pd.DataFrame): Input DataFrame with columns: 'Folder', 'Year', 'Data Name', 'Value', 'Share'.
        title (str): Title of the chart.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        file_name (str): The name of the image to save the plot.
    """
    file_path = f"{file_name}.{file_type}"
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

    if file_path.endswith(".pgf"):
        configure_for_pgf()

    # Save the plot to a file
    os.makedirs(export_dir, exist_ok=True)
    full_file_path = os.path.join(export_dir, file_path)
    plt.savefig(full_file_path)
    plt.close()

    print(f"Shares plot saved to {full_file_path}")


def plot_costs(df, title, x_label, y_label, file_name, export_dir="export/plots",file_type="png"):
    """
    Plot costs
    """
    file_path = f"{file_name}.{file_type}"
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
    ax.set_ylim(ylim[0] - abs(ylim[1] * 0.1), ylim[1] + abs(ylim[1] * 0.1))

    # Add values as labels above the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=12, padding=5)

    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Remove x-ticks and their labels
    ax.set_xticks([])
    ax.set_xticklabels([])

    # Show the plot
    plt.tight_layout()

    if file_path.endswith(".pgf"):
        configure_for_pgf()

    # Save the plot to a file
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()

    print(f"Costs plot saved to {file_path}")


def plot_data(
    data, title, x_label, y_label, file_name, custom_order=None, color_palette="viridis", export_dir="export/plots", file_type="png"
):
    """
    Plot data
    """
    file_path = f"{file_name}.{file_type}"
    if custom_order is not None:
        data = reorder_data(data, custom_order)
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

    # no x-ticks
    ax.set_xticks([])
    ax.set_xticklabels([])

    if file_path.endswith(".pgf"):
        configure_for_pgf()

    # Save the plot to a file
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()

    print(f"Data plot saved to {file_path}")


def plot_biomass_use(df, title, x_label, y_label, file_name, year=2050,export_dir="export/plots",file_type="png"):
    file_path = f"{file_name}.{file_type}"

    plt.rcParams.update({"font.size": 18})
    df["Data Name"] = df["Data Name"].str.replace("1", "")

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
        value_a = row["no_biomass_emissions"]
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
        facecolor="LightGreen",
        label="Biomass use in scenario no biomass emissions",
    )
    b_patch = mpl.patches.Patch(
        facecolor="none",
        hatch="///",
        label="Reduction in biomass use when emissions considered",
    )
    # red_patch = mpl.patches.Patch(
    #     facecolor="red",
    #     hatch="///",
    #     label="More biomass used with emissions considered",
    # )
    plt.legend(handles=[potential_patch, a_patch, b_patch], fontsize=14)

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

    secondary_locator = MultipleLocator(500)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    plt.tight_layout()

    if file_path.endswith(".pgf"):
        configure_for_pgf()

    # Save the plot to a file
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()

    print(f"Biomass use plot saved to {file_path}")


def plot_efs(export_dir="export/plots",file_type="png"):
    biomass_costs = {  # Euro/MWh_LHV
        "crop residues": 12.8786,
        "logging residues": 15.3932,  # fuelwood residues
        "stemwood": 12.6498,
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
        "chips and pellets": 25.4661,  # C&P_RW
    }

    # font size
    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(14, 10))
    bar_width = 0.8
    gap = 0.6  # Adjust this value to change the space between bars
    x_coords = np.arange(len(emission_factors_new_names)) * (bar_width + gap)

    for index, (biomass, ef) in enumerate(emission_factors_new_names.items()):
        ax.bar(x_coords[index], ef, width=bar_width, label=biomass, color="LightGreen")
        # Add cost as text above the bar
        cost = biomass_costs.get(biomass, "N/A")
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
        list(emission_factors_new_names.keys()) + ["natural gas", "oil", "coal"],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("")
    ax.set_ylabel("tonCO2/MWh")
    ax.set_title("Emission Factors and Costs for Different Feedstocks", fontsize=26)

    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks() / 0.0036)
    ax2.set_ylabel("g/MJ")

    secondary_locator = MultipleLocator(10)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.3))

    plt.tight_layout()

    file_path = f"emission_factors.{file_type}"
    if file_path.endswith(".pgf"):
        configure_for_pgf()
    # Save the plot to a file
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir,file_path)
    plt.savefig(file_path)
    plt.close()
    
    print(f"Emission factors plot saved to {file_path}")

def plot_bar_with_totals(
    df,
    title,
    x_label,
    y_label,
    file_name,
    custom_order=None,
    remove_letters=None,
    axis2_ticks=500,
    include_total=True,
    export_dir="export/plots",
    file_type="png",
):
    # plots the data in a bar chart and adds a sum of all values at the end
    # Pivot the DataFrame for plotting
    file_path = f"{file_name}.{file_type}"

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
    bar_width = 0.2

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
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
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right", fontsize=12)
    ax.legend(title="Scenario", fontsize=12)

    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks() * 3.6)
    ax2.set_ylabel("PJ")

    secondary_locator = MultipleLocator(axis2_ticks)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    plt.tight_layout()
    # Save the plot to a file

    if file_path.endswith(".pgf"):
        configure_for_pgf()

    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    plt.savefig(file_path)
    plt.close()

    print(f"Bar plot with totals saved to {file_path}")


def get_usage_dict(df, scenario, year=2050):
    potentials = biomass_potentials_TWh
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


def main():

    file_type = "png"
    # file_type = "pgf"

    custom_order = ["no_biomass_emissions", "biomass_emissions"]    
    export_dir = "export/plots"
    data_folder = "export"

    # export_dir = "export/land_use_scenarios/plots"
    # data_folder = "export/land_use_scenarios"
    # custom_order= ["no_biomass_emissions", "no_biomass_emissions_re_em", "biomass_emissions_no_re_em", "biomass_emissions"]

    data = load_csv("costs2050.csv",folder_path=data_folder)
    plot_data(
        data,
        "Total Costs",
        "",
        "Cost (Billion EUR)",
        "total_costs",
        custom_order,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("biomass_supply.csv",folder_path=data_folder)
    plot_biomass_use(data, "Biomass Use", "", "TWh", "biomass_supply", export_dir=export_dir)
    plot_bar_with_totals(
        data,
        "Biomass Supply",
        "",
        "TWh",
        "biomass_supply_totals",
        custom_order,
        remove_letters=[1],
        axis2_ticks=500,
        include_total=False,
        export_dir=export_dir,
        file_type=file_type,
    )

    plot_efs(export_dir=export_dir)

    data = load_csv("oil_production_2050.csv",folder_path=data_folder)
    plot_bar_with_shares(
        data,
        "Liquid Fuel Production in 2050",
        "",
        "TWh",
        "oil_production_2050",
        custom_order,
        width=10,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("electricity_generation_share_2050.csv",folder_path=data_folder)
    plot_bar_with_shares(
        data,
        "Electricity Generation in 2050",
        "",
        "TWh",
        "electricity_generation_share_2050",
        custom_order,
        axis2_ticks=5000,
        width=10,
        export_dir=export_dir,
        file_type=file_type,
    )
    # plot_difference_bar(
    #     data,
    #     "Difference in Electricity Generation Considering Biomass Emissions",
    #     "",
    #     "TWh",
    #     "electricity_generation_share_2050_diff",
    #     custom_order,
    # )

    data = load_csv("beccs.csv",folder_path=data_folder)
    plot_bar_with_totals(
        data,
        "BECCS",
        "",
        "TWh",
        "beccs",
        custom_order,
        remove_letters=[1],
        axis2_ticks=500,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("industrial_energy_2050.csv",folder_path=data_folder)
    plot_bar_with_totals(
        data,
        "Industrial Heat Supply in 2050",
        "",
        "TWh",
        "industrial_energy_2050",
        custom_order,
        axis2_ticks=500,
        include_total=False,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("heating_energy_2050.csv",folder_path=data_folder)
    plot_bar_with_totals(
        data,
        "Heating Energy Supply in 2050",
        "",
        "TWh",
        "heating_energy_2050",
        custom_order,
        axis2_ticks=1000,
        include_total=False,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("primary_energy_2050.csv",folder_path=data_folder)
    plot_bar_with_shares(
        data,
        "Primary Energy Supply in 2050",
        "",
        "TWh",
        "primary_energy_2050",
        custom_order,
        axis2_ticks=5000,
        width=10,
        threshold=0.0001,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("biomass_use_by_sector_2050.csv",folder_path=data_folder)
    plot_bar_with_shares(
        data,
        "Biomass Use by Sector in 2050",
        "",
        "TWh",
        "biomass_use_by_sector_shares",
        custom_order,
        axis2_ticks=500,
        width=10,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("fossil_fuel_supply.csv",folder_path=data_folder)
    plot_bar_with_totals(
        data,
        "Fossil Fuel Supply",
        "",
        "TWh",
        "fossil_fuel_supply",
        custom_order,
        include_total=False,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("cost_difference.csv",folder_path=data_folder)
    plot_costs(
        data,
        "Extra Costs Due to Biomass Emissions (bigger than 1 Billion EUR)",
        "",
        "Cost (Billion EUR)",
        "cost_difference",
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("shadow_price_2050.csv",folder_path=data_folder)
    plot_data(
        data,
        "CO2 Shadow Prices",
        "",
        "Shadow Price (EUR/tonCO2)",
        "shadow_prices",
        custom_order,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("hydrogen_production_2050.csv",folder_path=data_folder)
    plot_bar_with_totals(
        data,
        "Hydrogen Production",
        "",
        "TWh",
        "hydrogen_production_2050",
        custom_order,
        remove_letters=[1],
        axis2_ticks=500,
        include_total=True,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("heat_pumps_2050.csv",folder_path=data_folder)
    plot_bar_with_totals(
        data,
        "Heat Pump Electricity Consumption",
        "",
        "TWh",
        "heat_pumps_2050",
        custom_order,
        remove_letters=[0],
        axis2_ticks=500,
        include_total=True,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("gas_use_2050.csv",folder_path=data_folder)
    plot_bar_with_totals(
        data,
        "(Bio)Gas Use",
        "",
        "TWh",
        "gas_use_2050",
        custom_order,
        axis2_ticks=500,
        include_total=False,
        export_dir=export_dir,
        file_type=file_type,
    )

    supply_data = load_csv("biomass_supply.csv",folder_path=data_folder)
    usage_dict_default = get_usage_dict(supply_data, "no_biomass_emissions")
    usage_dict_biomass_emissions = get_usage_dict(supply_data, "biomass_emissions")

    data = load_csv("weighted_prices_2050.csv",folder_path=data_folder)
    plot_feedstock_prices(
        data,
        "Weighted Feedstock Prices in 2050",
        "",
        "EUR/MWh",
        "weighted_feedstock_prices_2050",
        export_dir=export_dir,
        file_type=file_type,
    )
    plot_costs_vs_prices(
        data,
        "Weighted Feedstock Prices vs. Costs in 2050",
        "Costs in Euro/MWh",
        "Prices in EUR/MWh",
        "prices_costs_default_2050",
        scenario="no_biomass_emissions",
        usage_dict=usage_dict_default,
        export_dir=export_dir,
        file_type=file_type,
    )
    plot_costs_vs_prices(
        data,
        "Weighted Feedstock Prices vs. Costs in 2050",
        "Costs in Euro/MWh",
        "Prices in EUR/MWh",
        "prices_costs_biomass_emissions_2050",
        scenario="biomass_emissions",
        usage_dict=usage_dict_biomass_emissions,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("co2_use.csv",folder_path=data_folder)
    plot_stacked_bar(
        data,
        "CO2 Use",
        "",
        "MtCO2",
        "co2_use",
        custom_order,
        multiplier=1e-6,
        export_dir=export_dir,
        file_type=file_type,
    )
    data = load_csv("co2_capture.csv",folder_path=data_folder)
    plot_stacked_bar(
        data,
        "CO2 Capture",
        "",
        "MtCO2",
        "co2_capture",
        custom_order,
        multiplier=1e-6,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("biomass_use_by_sector_2050.csv",folder_path=data_folder)
    plot_stacked_bar(
        data,
        "Biomass Use by Sector in 2050",
        "",
        "TWh",
        "biomass_use_by_sector_2050",
        custom_order,
        multiplier=1e-6,
        export_dir=export_dir,
        file_type=file_type,
    )

    upstream_data = load_csv("upstream_emissions.csv",folder_path=data_folder)

    data = load_csv("CCUS.csv",folder_path=data_folder)
    plot_BECCUS(
        data,
        "BECCUS—Biogenic CO2 Allocation",
        "",
        "Mt_CO2",
        "beccus",
        upstream_data=upstream_data,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("biomass_supply_difference.csv",folder_path=data_folder)
    data = data[data["data_name"] != "total"]
    data
    plot_stacked_bar(
        data,
        "Additional CO2 Emissions Due to Biomass Use",
        "",
        "Mt_CO2",
        "biomass_emission_difference",
        multiplier=1e-6,
        column="emission_difference",
        columns="year",
        index="Data Name",
        threshold=0.001,
        threshold_column="emission_difference",
        export_dir=export_dir,
        file_type=file_type,
    )

    create_gravitational_plot(
        "Gravitational Plot",
        "gravitational_plot",
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("biomass_supply.csv",folder_path=data_folder)
    create_gravitational_plot(
        "Gravitational Plot (biomass_emissions)",
        "gravitational_plot_biomass_emissions",
        biomass_supply=data,
        scenario="biomass_emissions",
        export_dir=export_dir,
        file_type=file_type,
    )
    create_gravitational_plot(
        "Gravitational Plot (no_biomass_emissions)",
        "gravitational_plot_no_biomass_emissions",
        biomass_supply=data,
        scenario="no_biomass_emissions",
        export_dir=export_dir,
        file_type=file_type,
    )


if __name__ == "__main__":
    main()
    # create_gravitational_plot(
    #     "Gravitational Plot",
    #     "gravitational_plot",
    # )
    # data = load_csv("biomass_supply.csv")
    # create_gravitational_plot(
    #     "Gravitational Plot (bm_em_710_seq)",
    #     "gravitational_plot_bm_em_710",
    #     biomass_supply=data,
    #     scenario="bm_em_710_seq",
    # )
    # create_gravitational_plot(
    #     "Gravitational Plot (bm_em_200_seq)",
    #     "gravitational_plot_bm_em_200",
    #     biomass_supply=data,
    #     scenario="biomass_emissions",
    # )
    # create_gravitational_plot(
    #     "Gravitational Plot (bm_em_150_seq)",
    #     "gravitational_plot_bm_em_150",
    #     biomass_supply=data,
    #     scenario="bm_em_150_seq",
    # )
