# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
import os
import io

from collections import defaultdict

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
from matplotlib.lines import Line2D

import plotly.graph_objects as go
import plotly.express as px
from result_analysis import get_emission_factors


import warnings


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
    print("Configuring matplotlib for PGF output...")
    mpl.rcParams.update({
        "pgf.texsystem": "xelatex",  # or 'xelatex'
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })
def configure_to_default():
    print("Configuring matplotlib to default...")    
    mpl.rcParams.update(mpl.rcParamsDefault)

def load_csv(file_path, folder_path="export", rename_scenarios=True):
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
    data = pd.read_csv(file_path)
    # Define the mapping for renaming
    if rename_scenarios:
        rename_dict = {
            "default_optimal": "Default",
            "optimal": "Carbon Costs",
            "default_710_optimal": "Default 710",
            "710_optimal": "Carbon Costs 710",
        }
        # Replace values in the entire DataFrame
        data = data.replace(rename_dict)
    return data


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
    capacity_factors=None,
    show_fossil_fuels=True,
    usage_threshold=False,
    variant_plot=False,
):

    file_path = f"{file_name}.{file_type}"


    if biomass_supply is not None and scenario is not None:
        biomass = biomass_supply[biomass_supply["Folder"] == scenario]
        # remove the 1 from the data_name
        biomass.loc[:, "Data Name"] = biomass["Data Name"].str.replace(
            "1", ""
        )
    if variant_plot:
        biomass_variant = biomass_supply[biomass_supply["Folder"] == f"{scenario}_710"]

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
            bt == "secondary forestry residues" or bt == "sludge" or "import" in bt
        ):  # below the point
            location = emissions[i] - 2 * sizes[i] / max_potential * 0.015 - 0.015
        plt.text(
            costs[i],
            location,
            new_names_dict[bt],
            fontsize=9,
            ha="center",
        )

    # Add renewable energy crosses if capacity factors are provided
    if capacity_factors is not None:

        # Filter capacity factors for the selected scenario
        if scenario is not None:
            scenario_cf = capacity_factors[capacity_factors["Folder"] == scenario]
        else:
            # Use default scenario if none specified
            scenario_cf = capacity_factors[capacity_factors["Folder"] == "Default"]
        
        # Define emission factors in ton/MW (from provided data)
        renewable_ef_per_mw = {
            "solar": 12.2,
            "onwind": 2.44,
            "solar-hsat": 73.2,
        }

        # Calc renewable costs LCOE with discounting
        investment_costs = {  # Euro/kW
            "solar": 320.8,
            "onwind": 1034.48,
            "solar-hsat": 384.3,
        }

        # Fixed O&M costs (Euro/MWh)
        marginal_costs = {
            "solar": 0.01,
            "onwind": 0.015,
            "solar-hsat": 0.01,
        }

        lifetimes = {  # years
            "solar": 40,
            "onwind": 30,
            "solar-hsat": 40,
        }

        discount_rate = 0.07

        lcoe = {}
        for tech in ["solar", "onwind", "solar-hsat"]:
            if tech in scenario_cf["Data Name"].values:
                cf = scenario_cf[scenario_cf["Data Name"] == tech]["Values"].values[0]
                invest = investment_costs[tech] * 1000  # Euro/kW → Euro/MW
                om = marginal_costs[tech]               # Euro/MWh
                lifetime = lifetimes[tech]

                # Capital Recovery Factor (CRF)
                crf = (discount_rate * (1 + discount_rate) ** lifetime) / ((1 + discount_rate) ** lifetime - 1)

                # LCOE formula with discounting
                lcoe[tech] = (invest * crf) / (cf * 8760) + om

                print(f"LCOE for {tech}: {lcoe[tech]:.2f} Euro/MWh")

        # For each renewable technology, calculate emission factor in ton/MWh
        for _, row in scenario_cf.iterrows():
            tech = row["Data Name"]
            if tech in renewable_ef_per_mw:
                cf = row["Values"]
                # Calculate emissions per MWh: ton/MW / (CF * 8760 hours/year) = ton/MWh
                emissions_per_mwh = renewable_ef_per_mw[tech] / (cf * 8760)
                
                # Plot as a cross with different color
                plt.scatter(
                    lcoe[tech],
                    emissions_per_mwh,
                    marker='p',
                    color='orange',
                    s=100,
                    linewidth=2,
                    label="_nolegend_"
                )
                location = emissions_per_mwh + 0.01
                if tech == "solar-hsat":  # below the point
                    location = emissions_per_mwh - 0.02                
                # Add text label
                plt.text(
                    lcoe[tech],
                    location,
                    f"{tech}",
                    fontsize=9,
                    ha="center",
                    color='black'
                )
    
    # Add fossil fuel markers
    if show_fossil_fuels:
        # Define fossil fuel data
        fossil_fuels = {
            "coal": {"cost": 9.55, "emission": 0.3361},
            "gas": {"cost": 24.57, "emission": 0.198},
            "oil": {"cost": 52.9, "emission": 0.2571}
        }
        
        # Plot fossil fuel markers
        for fuel, data in fossil_fuels.items():
            plt.scatter(
                data["cost"],
                data["emission"],
                marker='s',  # square marker to differentiate
                color='black',
                s=80,
                label="_nolegend_"
            )
            
            # Add text label
            plt.text(
                data["cost"],
                data["emission"] + 0.02,
                fuel,
                fontsize=9,
                ha="center",
                color='black',
            )

    # Set up the axes and draw to ensure limits are calculated
    plt.xlabel("Costs in Euro/MWh")
    plt.ylabel("Emission Factors in tonCO2/MWh")
    plt.title(title)
    plt.xlim(0, max(costs) + 5)  # Ensure this considers fossil fuel costs too
    plt.ylim(-0.02, max(emissions) + 0.05)  # Ensure this considers fossil fuel emissions too
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
        if variant_plot:
            if bt in ["manure", "sludge"]:
                color = "lightblue"
            else:
                color = "lightgreen"
            if biomass_variant is not None and bt in biomass_variant["Data Name"].values:
                supply = (
                    biomass_variant[biomass_variant["Data Name"] == bt]["Values"].values[0]
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
                circle_radius = np.sqrt(sizes[i] / np.pi)*0.95
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

        if bt in ["manure", "sludge"]:
            color = dig_biomass_color
        else:
            color = solid_biomass_color
        if biomass_supply is not None and bt in biomass["Data Name"].values:
            supply = (
                biomass[biomass["Data Name"] == bt]["Values"].values[0]
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
            circle_radius = np.sqrt(sizes[i] / np.pi)*0.95 #for some reason the circles are too big so they need to be scaled down
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

        # Plot usage threshold shading (if specified) behind all other elements
        if usage_threshold:

            # ----- 1.  input validation ------------------------------------
            if not isinstance(usage_threshold, list):
                raise ValueError("`usage_threshold` must be a list (max length = 2).")
            if len(usage_threshold) > 2:
                warnings.warn("Only the first two threshold specifications are used.")
                usage_threshold = usage_threshold[:2]

            # ----- 2.  build list of threshold descriptors -----------------
            thresh_specs = []   # each element: {"kind": "vertical"|"sloped", ...}

            for th in usage_threshold:
                if not (isinstance(th, dict)
                        and "biomass_type" in th
                        and "emission_cost" in th):
                    raise ValueError(
                        "Each threshold must be a dict with keys "
                        "'biomass_type' and 'emission_cost'."
                    )

                bt            = th["biomass_type"]
                emission_cost = th["emission_cost"]

                if bt not in biomass_types:
                    warnings.warn(f"Biomass type '{bt}' not found – skipping.")
                    continue

                idx        = biomass_types.index(bt)
                cost_ref   = costs[idx]          # €/MWh
                emit_ref   = emissions[idx]      # tCO₂/MWh
                total_ref  = cost_ref + emit_ref * emission_cost  # € per MWh
                #print(f"biomass type: {bt}, cost_ref: {cost_ref}, emit_ref: {emit_ref}, total_ref: {total_ref}")

                if emission_cost == 0:           # → vertical line
                    thresh_specs.append(
                        {"kind": "vertical", "x": cost_ref}
                    )
                else:                            # → sloped line
                    thresh_specs.append(
                        {"kind": "sloped",
                        "x_ref": cost_ref,
                        "total_ref": total_ref,
                        "emission_cost": emission_cost}
                    )

            if not thresh_specs:
                # nothing valid to plot
                pass
            else:
                # ----- 3.  prepare axis-aligned helpers -------------------
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                x_vals = np.linspace(x_min, x_max, 300)

                # build corresponding y arrays / vertical positions
                y_arrays = []
                for spec in thresh_specs:
                    if spec["kind"] == "vertical":
                        y_arrays.append(("vertical", spec["x"]))  # tag + x-pos
                    else:
                        ec   = spec["emission_cost"]
                        y_sl = (spec["total_ref"] - x_vals) / ec   # y(x)
                        y_arrays.append(y_sl)

                # ----- 4.  draw threshold(s) ------------------------------
                #  (always behind other plot elements → zorder=0)
                if len(y_arrays) == 1:            # ── single threshold
                    ya = y_arrays[0]
                    if isinstance(ya, tuple):     # vertical
                        ax.axvline(ya[1], color="grey",
                                alpha=0.5, linewidth=1, zorder=0)
                    else:                         # sloped “band” of zero height
                        ax.fill_between(x_vals, ya, ya,
                                        color="lightgrey", alpha=0.2,
                                        linewidth=0, zorder=0)

                else:                             # ── two thresholds
                    ya1, ya2 = y_arrays

                    # ---- case A: both vertical --------------------------
                    if isinstance(ya1, tuple) and isinstance(ya2, tuple):
                        x_left, x_right = sorted([ya1[1], ya2[1]])
                        ax.fill_betweenx([y_min, y_max], x_left, x_right,
                                        color="lightgrey", alpha=0.2, zorder=0)

                    # ---- case B: one vertical, one sloped ---------------
                    elif isinstance(ya1, tuple) ^ isinstance(ya2, tuple):
                        vert_x = ya1[1] if isinstance(ya1, tuple) else ya2[1]
                        y_sl   = ya2     if isinstance(ya1, tuple) else ya1
                        ax.fill_betweenx(y_sl, vert_x, x_vals,
                                        color="lightgrey", alpha=0.2, zorder=0)

                    # ---- case C: both sloped ----------------------------
                    else:
                        ax.fill_between(x_vals, ya1, ya2,
                                        color="lightgrey", alpha=0.2, zorder=0)

    # First legend: for circle sizes (potential)
    size_handles = [
        plt.scatter([], [], s=max_potential * (size / max_potential), 
                    edgecolors="black", facecolors="none", label=f"{size} TWh")
        for size in [100, 200, 500]
    ]
    # dummy_entry = plt.Line2D([0], [0], linestyle="none", marker= '',label="", alpha=0)
    # size_handles.append(dummy_entry)

    legend1 = ax.legend(
        handles=size_handles,
        scatterpoints=1,
        frameon=True,
        labelspacing=1,
        title="Potential",
        loc="upper right",
        borderpad=1.2,
    )
    ax.add_artist(legend1)  # Keep this legend when adding the next one

    colour_legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='green',
            markeredgecolor='green', label='Solid biomass', markersize=10, linewidth=0),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='lightgreen',
            markeredgecolor='lightgreen', label='Additional solid biomass use\nwith high co2 seq. potential', markersize=10, linewidth=0),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='blue',
            markeredgecolor='blue', label='Digestible biomass', markersize=10, linewidth=0),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='black',
            markeredgecolor='black', label='Fossil fuels', markersize=10, linewidth=0),
        Line2D([0], [0], marker='p', color='none', markerfacecolor='orange',
            markeredgecolor='orange', label='Renewable energy', markersize=10, linewidth=0),
    ]

    legend2 = ax.legend(
        handles=colour_legend_elements,
        loc="lower right",
        bbox_to_anchor=(1, 0),
        frameon=True,
    )

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
            label="Land Carbon Costs of Biomass",
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
                f"{percentage:.0f}%",
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
        delta_text = f"Δ {share_delta * 100:.0f}%"
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
            share_text = f"{shares[folder].iloc[j] * 100:.0f}%"
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


def plot_biomass_use(df, title, x_label, y_label, file_name, year=2050,export_dir="export/plots",file_type="png", labels=True):
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
        value_a = row["Default"]
        value_b = row["Carbon Costs"]
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
        if labels:
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

def plot_co2(df, scenario, file_name, export_dir="export/plots",file_type="png", unit = "ton", multiplier=1):

    major_sinks_colors = {
        'atmosphere': 'rgba(255, 0, 0, 0.8)',        # Red for atmospheric CO2
        'co2 captured': 'rgba(0, 0, 255, 0.8)',  # Blue for captured/stored CO2
        'co2 sequestered': 'rgba(0, 128, 0, 0.8)' # Green for sequestered CO2
    }
    default_node_color = 'rgba(200, 200, 200, 0.8)' # Light grey for other nodes
    link_color = 'rgba(180, 180, 180, 0.5)' # Light grey for links

    # Filter the DataFrame for the specified scenario
    df = df[df['folder'] == scenario]
    #mulitply the values by the multiplier
    df.loc[:, 'values'] = df['values'] * multiplier

    all_nodes_labels_original = pd.concat([df['from_sink'], df['to_sink']]).unique().tolist()

    # --- 2. Create a mapping from node label to index ---
    node_map = {node: i for i, node in enumerate(all_nodes_labels_original)}

    # --- 3. Prepare link data and calculate node in/out flows ---
    sources_indices = [] # Renamed from 'sources' to avoid conflict with a common variable name
    targets_indices = [] # Renamed from 'targets'
    values_list = []     # Renamed from 'values'
    link_hover_labels = []

    # To accumulate total incoming and outgoing flow values for each node
    node_in_values = defaultdict(float)  # Stores sum of flows INTO each node
    node_out_values = defaultdict(float) # Stores sum of flows OUT OF each node

    for _, row in df.iterrows():
        source_node_label = row['from_sink']
        target_node_label = row['to_sink']
        value = row['values']

        sources_indices.append(node_map[source_node_label])
        targets_indices.append(node_map[target_node_label])
        values_list.append(value)
        link_hover_labels.append(f"{source_node_label} → {target_node_label}: {value:,.0f}{unit}")

        node_out_values[source_node_label] += value
        node_in_values[target_node_label] += value

    # --- 4. Format node labels: Add total flow values to ALL nodes ---
    all_nodes_labels_formatted = []
    for node_label in all_nodes_labels_original:
        total_incoming = node_in_values[node_label]
        total_outgoing = node_out_values[node_label]
        
        # The display value for a node's throughput is max(in, out)
        # This aligns with Plotly's %{value} for node hovertemplate
        display_value_for_node = max(total_incoming, total_outgoing)
        
        if display_value_for_node > 0: # Only add value if there's flow
            all_nodes_labels_formatted.append(f"{node_label}<br><i><span style='font-size: 9px;'>({display_value_for_node:,.0f}{unit})</span></i>")
        else:
            # If a node somehow has 0 total flow (should be rare if it's part of a link)
            all_nodes_labels_formatted.append(node_label)

    # --- 5. Assign colors to nodes ---
    node_colors_list = []
    for node_label in all_nodes_labels_original:
        node_colors_list.append(major_sinks_colors.get(node_label, default_node_color))

    # --- 6. Create the Sankey diagram ---
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=all_nodes_labels_formatted, # Use selectively formatted labels
            color=node_colors_list,
            customdata=all_nodes_labels_original, # Original labels for hover
            # Hovertemplate shows total throughput (max of in/out) for ALL nodes
            hovertemplate='Node: %{customdata}<br>Total Flow: %{value:,.0f}{unit}<extra></extra>'
        ),
        link=dict(
            source=sources_indices,
            target=targets_indices,
            value=values_list,
            label=link_hover_labels,
            color=link_color,
            hovertemplate='Link: %{label}<extra></extra>'
        ),
        arrangement='freeform' # freeform, perpendicular, fixed, or snap
    )])

    fig.update_layout(
        title_text=f"Carbon Flow: {scenario}",
        font_size=10,
        height=max(800, len(all_nodes_labels_original) * 30), # Increased multiplier for height
        width=1200,
        margin=dict(l=50, r=50, b=50, t=100, pad=4)
    )

    #fig.show()
    # Save the plot to a file
    file_path = f"{file_name}.{file_type}"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    fig.write_image(file_path, scale=2)
    print(f"Sankey diagram saved to {file_path}")


def plot_co2_sankey(
    df,
    scenario,
    multiplier=1.0,
    output_dir='plots',
    unit_label="tonCO2",
    colour_links_by_technology=True,
    include_legend=True
):
    """
    Generate a Sankey diagram from a DataFrame of carbon flows for a given scenario.

    Parameters:
        df (pd.DataFrame): Input data with [folder, data_name, values, from_sink, to_sink].
        scenario (str): Scenario to plot (filters 'folder').
        multiplier (float): Multiplier for value scaling.
        output_dir (str): Output directory for the saved plot.
        unit_label (str): Unit string shown in hover tooltips.
        colour_links_by_technology (bool): Whether to colour flows by technology.
        include_legend (bool): Whether to add legend entries for technologies and sinks.

    Returns:
        plotly.graph_objects.Figure
    """
    # Filter for selected scenario
    scenario_data = df[df['folder'] == scenario].copy()
    if scenario_data.empty:
        raise ValueError(f"No data found for scenario: {scenario}")
    scenario_data['values'] *= multiplier

    # Distinct nodes
    nodes = list(set(scenario_data['from_sink']) | set(scenario_data['to_sink']))
    node_index = {label: i for i, label in enumerate(nodes)}

    # Colour palette for technologies
    extended_palette = (
        px.colors.qualitative.Safe +
        px.colors.qualitative.Bold +
        px.colors.qualitative.Vivid +
        px.colors.qualitative.Pastel +
        px.colors.qualitative.Set1 +
        px.colors.qualitative.Set3
    )
    tech_list = scenario_data['data_name'].unique()
    tech_colors = {tech: extended_palette[i % len(extended_palette)] for i, tech in enumerate(tech_list)}

    # Sink type node colouring
    sink_type_colors = {
        "Atmosphere": "#56B4E9",
        "CO2 Captured": "#332288",
        "CO2 Storage": "#117733",
        "CO2 Utilization": "#D95F02",
        "Other": "#AAAAAA"
    }

    def classify_sink(node):
        n = node.lower()
        if "atmosphere" in n:
            return "Atmosphere"
        elif "captured" in n or "capture" in n:
            return "CO2 Captured"
        elif "storage" in n or "sequester" in n:
            return "CO2 Storage"
        elif "utilization" in n or "utilised" in n or "utilized" in n:
            return "CO2 Utilization"
        else:
            return "Other"

    node_colors = [sink_type_colors[classify_sink(n)] for n in nodes]

    # Prepare links
    sources, targets, values, link_labels, link_colors = [], [], [], [], []

    for _, row in scenario_data.iterrows():
        sources.append(node_index[row['from_sink']])
        targets.append(node_index[row['to_sink']])
        values.append(row['values'])
        link_labels.append(row['data_name'])
        if colour_links_by_technology:
            link_colors.append(tech_colors[row['data_name']])
        else:
            link_colors.append("rgba(150,150,150,0.4)")  # uniform grey if not coloured


    # Node throughput for hover info
    node_in = {i: 0 for i in range(len(nodes))}
    node_out = {i: 0 for i in range(len(nodes))}
    for s, t, v in zip(sources, targets, values):
        node_out[s] += v
        node_in[t] += v

    throughput = [max(node_in[i], node_out[i]) for i in range(len(nodes))]

    # Sankey diagram
    fig = go.Figure(go.Sankey(
        arrangement="perpendicular", # alternatives: 'snap', 'freeform', 'perpendicular', 'fixed'
        node=dict(
            label=[f"{n}<br><span style='font-size:10px'>({throughput[node_index[n]]:,.0f} {unit_label})</span>" for n in nodes],
            color=node_colors,
            pad=30,
            thickness=15,
            line=dict(color="black", width=0.5)
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=link_colors,
        )
    ))

    fig.update_traces(
        selector=dict(type='sankey'),
        node_customdata=throughput,
        node_hovertemplate='%{label}: %{customdata:.2f} ' + unit_label + ' total<extra></extra>',
        link_hovertemplate='%{label}: %{value:.2f} ' + unit_label + '<extra></extra>'
    )

    # Legend
    if include_legend:
        if colour_links_by_technology:
            for tech, colour in tech_colors.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=colour, symbol="square"),
                    name=tech,
                    legendgroup="tech",
                    legendgrouptitle_text="Technology"
                ))
        for cat, colour in sink_type_colors.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=colour, symbol="circle"),
                name=cat,
                legendgroup="sink",
                legendgrouptitle_text="Sink Type"
            ))

    # Layout and export
    fig.update_layout(
        title_text=f"CO2 Flow Sankey – Scenario: {scenario}",
        font_size=10,
        plot_bgcolor='white',
        showlegend=True,
        width=1700,  # wider canvas
        height=1000, # taller if needed
        #autosize=False,
        margin=dict(l=100, r=400, t=80, b=200)  # add right margin for legend
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sankey_{scenario}.png")
    fig.write_image(out_path)
    print(f"Sankey diagram saved to {out_path}")

    return fig


def plot_mga(df, file_name, title = "Near Optimal Biomass Use", export_dir='export/plots', file_type='png', unit='MWh', multiplier=1):
    """
    Plot the near-optimal solution space for biomass use under various cost deviations.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with MGA results. Expected columns:
        - 'Folder': scenario (e.g., 'optimal', 'max_0.05', 'min_0.05', etc.)
        - 'Year': (optional) year or time period (not used in plotting)
        - 'Data Name': (optional) metric name (should contain 'biomass' for biomass usage)
        - 'Values': numeric values of the metric (biomass usage)
        The DataFrame should include an 'optimal' scenario and matching 'max_X'/'min_X' pairs for each cost deviation X.
    file_name : str
        Base name for the saved plot file (without extension).
    export_dir : str, optional
        Directory to save the plot in (default is 'export/plots').
    file_type : str, optional
        File format for saving (e.g., 'png', 'pdf'; default 'png').
    unit : str, optional
        Unit of biomass values (for y-axis label, default 'MWh').
    multiplier : float, optional
        Factor to scale the biomass values by (useful for unit conversion; default 1).
    """
    # Ensure the output directory exists
    os.makedirs(export_dir, exist_ok=True)
    
    # If DataFrame has a 'Data Name' column, filter to rows containing 'biomass' (case-insensitive)
    if 'Data Name' in df.columns:
        biomass_mask = df['Data Name'].str.contains('biomass', case=False, na=False)
        df_plot = df[biomass_mask].copy() if biomass_mask.any() else df.copy()
    else:
        df_plot = df.copy()
    
    # Dictionaries to hold min and max biomass values for each cost deviation percentage
    min_values = {}
    max_values = {}
    optimal_value = None  # to store the biomass value for the optimal scenario (0% deviation)
    
    # Process each scenario in the DataFrame
    for _, row in df_plot.iterrows():
        scenario = str(row['Folder']).lower()       # scenario name (e.g., 'optimal', 'max_0.05')
        value = float(row['Values']) * multiplier   # apply unit scaling to the value
        if 'optimal' in scenario:
            # Cost-optimal scenario (0% cost deviation)
            optimal_value = value
            min_values[0] = value
            max_values[0] = value
        elif 'min_' in scenario:
            # Scenario minimizing biomass at X% cost deviation
            dev_str = scenario.split('min_', 1)[1]  # e.g., '0.05' from 'min_0.05'
            # Convert the deviation string to a fraction (e.g., '0.05' -> 0.05)
            try:
                cost_dev_frac = float(dev_str)
            except ValueError:
                # Handle cases like 'min_5' or 'min_5%' by interpreting as percentage
                dev_str_num = dev_str.strip('%')
                cost_dev_frac = float(dev_str_num) / 100.0 if dev_str_num.replace('.', '', 1).isdigit() else None
            if cost_dev_frac is not None:
                cost_dev_percent = cost_dev_frac * 100  # convert to percentage (float -> int)
                min_values[cost_dev_percent] = value
        elif 'max_' in scenario:
            # Scenario maximizing biomass at X% cost deviation
            dev_str = scenario.split('max_', 1)[1]
            try:
                cost_dev_frac = float(dev_str)
            except ValueError:
                dev_str_num = dev_str.strip('%')
                cost_dev_frac = float(dev_str_num) / 100.0 if dev_str_num.replace('.', '', 1).isdigit() else None
            if cost_dev_frac is not None:
                cost_dev_percent = cost_dev_frac * 100
                max_values[cost_dev_percent] = value
    
    # Check that the optimal scenario was provided
    if optimal_value is None:
        raise ValueError("Optimal scenario data is missing in the DataFrame (expected a 'optimal' entry).")
    
    # Prepare sorted list of cost deviation percentages (e.g., [0, 5, 10, 15])
    cost_devs = sorted(set(min_values.keys()) | set(max_values.keys()))
    
    # Build lists of y-values (biomass use) for min and max lines in order of cost_devs
    x_vals = []
    y_min = []
    y_max = []
    for dev in cost_devs:
        if dev in min_values and dev in max_values:
            x_vals.append(dev)
            y_min.append(min_values[dev])
            y_max.append(max_values[dev])
    # (If any dev is missing a min or max, it will be skipped to avoid incomplete data plotting)
    
    # Create the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Fill the area between the min and max biomass usage curves (near-optimal solution space)
    ax.fill_between(x_vals, y_min, y_max, color='grey', alpha=0.3, label='Near-optimal solution space')
    # Plot the minimum biomass usage curve (blue solid line with circle markers)
    ax.plot(x_vals, y_min, color='tab:blue', marker='o', label='Min biomass use')
    # Plot the maximum biomass usage curve (orange solid line with triangle markers)
    ax.plot(x_vals, y_max, color='tab:orange', marker='o', label='Max biomass use')
    # Mark the cost-optimal solution (0% deviation) as a distinct point (black diamond marker)
    ax.plot([0], [optimal_value], color='black', marker='o', markersize=8, linestyle='none',
            label='Cost-optimal solution')
    
    # # Annotate each data point with its value and scenario label
    # for dev in x_vals:
    #     if dev == 0:
    #         # Optimal point annotation (e.g., "100 MWh (optimal)")
    #         ax.annotate(f'{optimal_value:.2f} {unit}\n(optimal)',
    #                     xy=(dev, optimal_value), xytext=(5, 5), textcoords='offset points',
    #                     ha='left', va='bottom', fontsize=9)
    #     else:
    #         # Annotate min point (e.g., "80 MWh (min 5%)") slightly below and to the right
    #         if dev in min_values:
    #             val_min = min_values[dev]
    #             ax.annotate(f'{val_min:.2f} {unit}\n(min {dev}%)',
    #                         xy=(dev, val_min), xytext=(5, -15), textcoords='offset points',
    #                         ha='left', va='top', fontsize=8, color='tab:blue')
    #         # Annotate max point (e.g., "120 MWh (max 5%)") slightly above and to the right
    #         if dev in max_values:
    #             val_max = max_values[dev]
    #             ax.annotate(f'{val_max:.2f} {unit}\n(max {dev}%)',
    #                         xy=(dev, val_max), xytext=(5, 5), textcoords='offset points',
    #                         ha='left', va='bottom', fontsize=8, color='tab:orange')
    
    # Configure the axes labels and ticks 
    ax.set_xlabel('Cost deviation from optimal')
    ax.set_ylabel(f'Biomass use ({unit})')
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f'{x:.1f}%' if x % 1 else f'{int(x)}%' for x in x_vals])
    ax.set_yscale('linear')  # ensure y-axis is linear (it is by default)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)  # add a light grid for readability
    
    # Add a legend to identify the plotted elements
    handles, labels = ax.get_legend_handles_labels()
    # Reorder legend entries for clarity
    legend_order = ['Cost-optimal solution', 'Max biomass use', 'Min biomass use', 'Near-optimal solution space']
    ordered = [(h, l) for h, l in zip(handles, labels) if l in legend_order]
    ordered_handles, ordered_labels = zip(*sorted(ordered, key=lambda x: legend_order.index(x[1])))
    ax.legend(ordered_handles, ordered_labels, loc='best')
    
    # Save the plot to a file in the specified directory and format
    output_path = os.path.join(export_dir, f"{file_name}.{file_type}")
    plt.savefig(output_path, format=file_type, bbox_inches='tight')
    plt.close(fig)
    print(f"MGA plot saved to {output_path}")

def plot_stacked_biomass_with_errorbars(data, export_dir="export/plots", file_name="biomass_stacked_errorbar", file_type="png"):
    # Filter for year 2050
    df = data[data["Year"] == 2050].copy()

    # Select relevant scenarios
    scenarios = ["Default", "Carbon Costs"]
    variant_suffix = "_710"
    sectors = df["Data Name"].unique()

    # Aggregate values per scenario and sector
    scenario_totals = {}
    for scen in scenarios + [s + variant_suffix for s in scenarios]:
        scenario_df = df[df["Folder"] == scen]
        sector_values = scenario_df.set_index("Data Name")["Values"].reindex(sectors, fill_value=0)
        scenario_totals[scen] = sector_values / 1e6  # Convert to TWh

    # Calculate total biomass and error bars
    total_biomass = {s: scenario_totals[s].sum() for s in scenarios}
    total_variants = {s: scenario_totals[s + variant_suffix].sum() for s in scenarios}
    diffs = {s: total_variants[s] - total_biomass[s] for s in scenarios}

    # Plotting
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(12, 8))

    bottom = np.zeros(len(scenarios))
    for sector in sectors:
        heights = [scenario_totals[s][sector] for s in scenarios]
        ax.bar(x, heights, bottom=bottom, label=sector)
        bottom += heights

    # Error bars
    y = [total_biomass[s] for s in scenarios]
    yerr = np.array([
        [abs(diffs[s]) if diffs[s] < 0 else 0 for s in scenarios],  # lower
        [diffs[s] if diffs[s] > 0 else 0 for s in scenarios]        # upper
    ])
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=6, linewidth=1.5)

    # Customise plot
    ax.set_xticks(x)
    ax.set_xticklabels(["Default", "Carbon Costs"])
    ax.set_ylabel("Total Biomass Use (TWh)")
    ax.set_title("Biomass Use by Sector in 2050")
    ax.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, f"{file_name}.{file_type}")

    if file_path.endswith(".pgf"):
        configure_for_pgf()

    plt.savefig(file_path)
    plt.close()

    print(f"Plot saved to {file_path}")

def main(custom_order=["Default", "Carbon Costs"], file_type="png", export_dir="export/plots", data_folder="export"):

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
    plot_biomass_use(data, "Biomass Use", "", "TWh", "biomass_supply", export_dir=export_dir,labels=False)
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

    data = load_csv("oil_production.csv",folder_path=data_folder)
    plot_bar_with_shares(
        data,
        "Liquid Fuel Production in 2050",
        "",
        "TWh",
        "oil_production",
        custom_order,
        width=10,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("electricity_generation_share.csv",folder_path=data_folder)
    plot_bar_with_shares(
        data,
        "Electricity Generation in 2050",
        "",
        "TWh",
        "electricity_generation_share",
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
    #     "electricity_generation_share_diff",
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

    # data = load_csv("industrial_energy.csv",folder_path=data_folder)
    # plot_bar_with_totals(
    #     data,
    #     "Industrial Heat Supply in 2050",
    #     "",
    #     "TWh",
    #     "industrial_energy",
    #     custom_order,
    #     axis2_ticks=500,
    #     include_total=False,
    #     export_dir=export_dir,
    #     file_type=file_type,
    # )

    # data = load_csv("heating_energy.csv",folder_path=data_folder)
    # plot_bar_with_totals(
    #     data,
    #     "Heating Energy Supply in 2050",
    #     "",
    #     "TWh",
    #     "heating_energy",
    #     custom_order,
    #     axis2_ticks=1000,
    #     include_total=False,
    #     export_dir=export_dir,
    #     file_type=file_type,
    # )

    data = load_csv("primary_energy.csv",folder_path=data_folder)
    plot_bar_with_shares(
        data,
        "Primary Energy Supply in 2050",
        "",
        "TWh",
        "primary_energy",
        custom_order,
        axis2_ticks=5000,
        width=10,
        threshold=0.0001,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("biomass_use_by_sector.csv",folder_path=data_folder)
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

    data = load_csv("shadow_price.csv",folder_path=data_folder)
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

    data = load_csv("hydrogen_production.csv",folder_path=data_folder)
    plot_bar_with_totals(
        data,
        "Hydrogen Production",
        "",
        "TWh",
        "hydrogen_production",
        custom_order,
        remove_letters=[1],
        axis2_ticks=500,
        include_total=True,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("heat_pumps.csv",folder_path=data_folder)
    plot_bar_with_totals(
        data,
        "Heat Pump Electricity Consumption",
        "",
        "TWh",
        "heat_pumps",
        custom_order,
        remove_letters=[0],
        axis2_ticks=500,
        include_total=True,
        export_dir=export_dir,
        file_type=file_type,
    )

    data = load_csv("gas_use.csv",folder_path=data_folder)
    plot_bar_with_totals(
        data,
        "(Bio)Gas Use",
        "",
        "TWh",
        "gas_use",
        custom_order,
        axis2_ticks=500,
        include_total=False,
        export_dir=export_dir,
        file_type=file_type,
    )

    supply_data = load_csv("biomass_supply.csv",folder_path=data_folder)
    usage_dict_default = get_usage_dict(supply_data, "Default")
    usage_dict_carbon_costs = get_usage_dict(supply_data, "Carbon Costs")

    data = load_csv("weighted_prices.csv",folder_path=data_folder)
    plot_feedstock_prices(
        data,
        "Weighted Feedstock Prices in 2050",
        "",
        "EUR/MWh",
        "weighted_feedstock_prices",
        export_dir=export_dir,
        file_type=file_type,
    )
    plot_costs_vs_prices(
        data,
        "Weighted Feedstock Prices vs. Costs in 2050",
        "Costs in Euro/MWh",
        "Prices in EUR/MWh",
        "prices_costs_default",
        scenario="Default",
        usage_dict=usage_dict_default,
        export_dir=export_dir,
        file_type=file_type,
    )
    plot_costs_vs_prices(
        data,
        "Weighted Feedstock Prices vs. Costs in 2050",
        "Costs in Euro/MWh",
        "Prices in EUR/MWh",
        "prices_costs_carbon_costs",
        scenario="Carbon Costs",
        usage_dict=usage_dict_carbon_costs,
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

    data = load_csv("biomass_use_by_sector.csv",folder_path=data_folder)
    plot_stacked_bar(
        data,
        "Biomass Use by Sector in 2050",
        "",
        "TWh",
        "biomass_use_by_sector",
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
        no_xticks=True,
    )

    capacity_factors = load_csv("capacity_factors.csv",folder_path=data_folder)

    create_gravitational_plot(
        "Gravitational Plot",
        "gravitational_plot",
        export_dir=export_dir,
        file_type=file_type,
        capacity_factors=capacity_factors,
    )

    data = load_csv("biomass_supply.csv",folder_path=data_folder)

    # for scenario in custom_order:
    #     create_gravitational_plot(
    #         f"Gravitational Plot ({scenario})",
    #         f"gravitational_plot_{scenario}",
    #         biomass_supply=data,
    #         scenario=scenario,
    #         export_dir=export_dir,
    #         file_type=file_type,
    #         capacity_factors=capacity_factors,
    #     )
    
    usage_threshold = [
        {
            "biomass_type": "grasses",
            "emission_cost": 265  # €/tonCO2
        },
        {
            "biomass_type": "fuelwood residues",
            "emission_cost": 869  # €/tonCO2
        }
    ]
    create_gravitational_plot(
        "Gravitational Plot (Carbon Costs)",
        "gravitational_plot_carbon_costs",
        export_dir=export_dir,
        file_type=file_type,
        capacity_factors=capacity_factors,
        biomass_supply=data,
        scenario="Carbon Costs",
        usage_threshold=usage_threshold,
    )

    usage_threshold = [
        {
            "biomass_type": "C&P_RW",
            "emission_cost": 0  # €/tonCO2
        },
        {
            "biomass_type": "woody crops",   
            "emission_cost": 0  # €/tonCO2
        }
    ]
    create_gravitational_plot(
        "Gravitational Plot (Default)",
        "gravitational_plot_default",
        export_dir=export_dir,
        file_type=file_type,
        capacity_factors=capacity_factors,
        biomass_supply=data,
        scenario="Default",
        usage_threshold=usage_threshold,
    )

def specific_plots():
    """
    Create specific plots for the project.
    """
    data = load_csv("biomass_supply.csv",folder_path="export/seq")
    capacity_factors = load_csv("capacity_factors.csv",folder_path="export")
    create_gravitational_plot(
        "Cost vs CO2 Emissions (Default)",
        "gravitational_plot_default",
        biomass_supply=data,
        scenario="Default",
        export_dir="export/plots",
        file_type="png",
        capacity_factors=capacity_factors,
        variant_plot=True,
    )
    create_gravitational_plot(
        "Cost vs CO2 Emissions (carbon costs)",
        "gravitational_plot_carbon_costs",
        biomass_supply=data,
        scenario="carbon_costs",
        export_dir="export/plots",
        file_type="png",
        capacity_factors=capacity_factors,
        variant_plot=True,
    )
    data = load_csv("biomass_use_by_sector.csv",folder_path="export/seq")
    plot_stacked_biomass_with_errorbars(
        data,
        export_dir="export/plots",
        file_name="biomass_stacked_errorbar",
        file_type="png"
    )


if __name__ == "__main__":

    file_type = "png"
    # file_type = "pgf"

    # custom_order = ["Default", "Carbon Costs", "Default 710", "Carbon Costs 710"]  
    # export_dir = "export/seq_plots"
    # data_folder = "export/seq"

    custom_order = ["Default", "Carbon Costs"]  # Adjusted for the current context
    export_dir = "export/plots"
    data_folder = "export"
    #main(custom_order=custom_order, file_type=file_type, export_dir=export_dir, data_folder=data_folder)

    specific_plots()

    ########## MGA Plots ###########
    # mga_data = load_csv("biomass_use_carbon_costs_710.csv",folder_path="export/mga")
    # plot_mga(
    #     mga_data,
    #     "mga_carbon_costs_710",
    #     title="Near Optimal Biomass Use (Scenario Carbon Costs 710)",
    #     export_dir="export/mga",
    #     file_type="png",
    #     unit="TWh",
    #     multiplier=1e-6,
    # )
    # mga_data = load_csv("biomass_use_carbon_costs.csv",folder_path="export/mga")
    # plot_mga(
    #     mga_data,
    #     "mga_carbon_costs",
    #     title="Near Optimal Biomass Use (Scenario Carbon Costs)",
    #     export_dir="export/mga",
    #     file_type="png",
    #     unit="TWh",
    #     multiplier=1e-6,
    # )
    # mga_data = load_csv("biomass_use_default_710.csv",folder_path="export/mga")
    # plot_mga(
    #     mga_data,
    #     "mga_default_710",
    #     title="Near Optimal Biomass Use (Scenario Default 710)",
    #     export_dir="export/mga",
    #     file_type="png",
    #     unit="TWh",
    #     multiplier=1e-6,
    # )
    # mga_data = load_csv("biomass_use_default.csv",folder_path="export/mga")
    # plot_mga(
    #     mga_data,
    #     "mga_default",
    #     title="Near Optimal Biomass Use (Scenario Default)",
    #     export_dir="export/mga",
    #     file_type="png",
    #     unit="TWh",
    #     multiplier=1e-6,
    # )






    ########### Sankey Diagrams ###########
    # co2_data = load_csv("co2_sankey.csv",folder_path="export")
    # plot_co2_sankey(
    #     co2_data,
    #     scenario="Default",
    #     multiplier=1e-6,
    #     output_dir="export/plots",
    #     unit_label="Mt CO2",
    # )
    # plot_co2_sankey(
    #     co2_data,
    #     scenario="Carbon Costs",
    #     multiplier=1e-6,
    #     output_dir="export/plots",
    #     unit_label="Mt CO2",
    # )
