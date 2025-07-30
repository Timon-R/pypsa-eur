# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# PLOTTING CONFIGURATION:
# All main plotting functions now accept fig_width, fig_height, fontsize, and title_fontsize parameters
# to allow consistent sizing and styling across all plots. Change the default values below or
# pass parameters when calling main(), specific_plots(), mga_plots(), or SA_plots().

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
from collections import OrderedDict
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch

import plotly.graph_objects as go
import plotly.express as px
from result_analysis import get_emission_factors

import matplot2tikz 

import warnings

# Default plotting parameters
DEFAULT_FIGURE_WIDTH = 10
DEFAULT_FIGURE_HEIGHT = 6
DEFAULT_FONTSIZE = 12
DEFAULT_TITLE_FONTSIZE = 16


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
    "residues from landscape care": "residues from landscape care", #consider naming it landscape management
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
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",   # Or any system font name if set in LaTeX
        "pgf.rcfonts": False,
    })
def configure_to_default():
    print("Configuring matplotlib to default...")    
    mpl.rcParams.update(mpl.rcParamsDefault)

def configure_for_tikz():
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",   # or xelatex/lualatex if you use those
        "text.usetex": True,           # so LaTeX typesets all labels
        "font.family": "serif",
        "pgf.rcfonts": False,         
    })

def carbon_flow_diagram(save_path: str | None = "carbon_flow_diagram.pdf",
                        show: bool = False,
                        close: bool = True,
                        fig_width=DEFAULT_FIGURE_WIDTH,
                        fig_height=DEFAULT_FIGURE_HEIGHT,
                        fontsize=DEFAULT_FONTSIZE,
                        title_fontsize=DEFAULT_TITLE_FONTSIZE):
    """
    Draw the linear vs circular carbon-flow schematic.

    Parameters
    ----------
    save_path : str | None   PDF/PNG file name (None → no file written)
    show      : bool         True → pop up an interactive window
    close     : bool         True → plt.close(fig) afterwards
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch

    # LaTeX & font setup
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=fontsize)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # ── Linear flow ───────────────────────────────────────────────────────
    ax.add_patch(Rectangle((0.08, 0.70), 0.28, 0.15, fill=False, lw=2))
    ax.add_patch(Rectangle((0.64, 0.70), 0.28, 0.15, fill=False, lw=2))

    ax.text(0.22, 0.775,  r'\shortstack{Cement / \\Fossil Fuels}',
            ha='center', va='center')
    ax.text(0.78, 0.775, r'\shortstack{Sequestered\\Carbon}',
            ha='center', va='center')

    ax.add_patch(FancyArrowPatch((0.36, 0.775), (0.64, 0.775),
                                 arrowstyle='-|>', mutation_scale=18, lw=2))
    ax.text(0.50, 0.805, r'CCS', ha='center', va='bottom')
    ax.text(0.50, 0.92,  r'\textbf{Linear Carbon Flow}',
            ha='center', va='center', fontsize=title_fontsize)

    # ── Circular flow ────────────────────────────────────────────────────
    ax.add_patch(Ellipse((0.22, 0.35), 0.28, 0.17, fill=False, lw=2))
    ax.add_patch(Ellipse((0.78, 0.35), 0.28, 0.17, fill=False, lw=2))

    ax.text(0.22, 0.35, r'\shortstack{Liquid Fuels}',
            ha='center', va='center')
    ax.text(0.78, 0.35, r'Atmosphere',
            ha='center', va='center')

    ax.add_patch(FancyArrowPatch((0.36, 0.35), (0.64, 0.35),
                                 arrowstyle='-|>', mutation_scale=18, lw=2))
    for start, end in [((0.78, 0.26), (0.78, 0.15)),
                       ((0.78, 0.15), (0.22, 0.15)),
                       ((0.22, 0.15), (0.22, 0.26))]:
        ax.add_patch(FancyArrowPatch(start, end,
                                     arrowstyle='-|>', mutation_scale=18, lw=2))

    ax.text(0.50, 0.50, r'\textbf{Circular Carbon Flow}',
            ha='center', va='center', fontsize=title_fontsize)
    ax.text(0.50, 0.07,
            r'Biomass to liquid / DAC \& Fischer--Tropsch / Electrobiofuels',
            ha='center', va='center')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig

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
            "default": "Default",
            "default_710": "Default 710",
            "optimal": "Carbon Stock Changes",
            "carbon_costs": "Carbon Stock Changes",
            "carbon_costs_710": "Carbon Stock Changes 710",
            "default_710_optimal": "Default 710",
            "710_optimal": "Carbon Stock Changes 710",
            "cscs": "Carbon Stock Changes",
            "cscs_710": "Carbon Stock Changes 710",
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
    include_solar_hsat=True,
    fig_width=DEFAULT_FIGURE_WIDTH,
    fig_height=DEFAULT_FIGURE_HEIGHT,
    fontsize=DEFAULT_FONTSIZE,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
):

    file_path = f"{file_name}.{file_type}"
    mpl.rcParams.update({
        "text.usetex": False,         # plain Matplotlib text engine
        "font.family":   "serif",
        "font.serif":    ["CMU Serif", "Latin Modern Roman",
                        "Computer Modern Roman", "Times"],  # fall-backs
        "mathtext.fontset": "cm",     # Computer Modern for $math$
        "figure.dpi":    300,
        "font.size": fontsize,
    })

    if biomass_supply is not None and scenario is not None:
        biomass = biomass_supply[biomass_supply["Folder"] == scenario]
        # remove the 1 from the data_name
        biomass.loc[:, "Data Name"] = biomass["Data Name"].str.replace(
            "1", ""
        )
    if variant_plot:
        biomass_variant = biomass_supply[biomass_supply["Folder"] == f"{scenario} 710"]

    # Extract data for plotting
    biomass_types = list(emission_factors.keys())
    emissions = [emission_factors[bt] for bt in biomass_types]
    costs = [biomass_costs[bt] for bt in biomass_types]
    potentials = [biomass_potentials_TWh[bt] for bt in biomass_types]

    # Normalize potentials for circle sizes
    max_potential = max(potentials)
    sizes = [max_potential * (p / max_potential) for p in potentials]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    dig_biomass_color = "blue"
    solid_biomass_color = "green"

    # Draw the plot first to get the limits
    for i, bt in enumerate(biomass_types):
        if bt in ["manure", "sludge"]:
            color = dig_biomass_color
        else:
            color = solid_biomass_color
        plt.scatter(
            float(costs[i]),
            float(emissions[i]),
            s=float(sizes[i]),
            alpha=1,
            facecolors="none",
            edgecolors=color,
            linewidth=1,
        )
        location = float(emissions[i]) + 2 * float(sizes[i]) / max_potential * 0.015 + 0.01
        if (
            bt == "secondary forestry residues" or bt == "sludge" or "import" in bt or bt == "fuelwoodRW"
        ):  # below the point
            location = float(emissions[i]) - 2 * float(sizes[i]) / max_potential * 0.015 - 0.015
        plt.text(
            float(costs[i]),
            location,
            new_names_dict[bt],
            fontsize=fontsize,
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
            "solar-hsat": 24.4,
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
                cf = scenario_cf[scenario_cf["Data Name"] == tech]["Values"].values[0].item()
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
                if not include_solar_hsat:
                    if tech == "solar-hsat":
                        continue
                cf = row["Values"]
                # Calculate emissions per MWh: ton/MW / (CF * 8760 hours/year) = ton/MWh
                emissions_per_mwh = renewable_ef_per_mw[tech] / (cf * 8760)
                print(f"Emissions for {tech}: {emissions_per_mwh:.4f} tonCO2/MWh")
                
                # Plot as a cross with different color
                plt.scatter(
                    lcoe[tech],
                    emissions_per_mwh,
                    marker='x',
                    color='orange',
                    s=80,
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
                    fontsize=fontsize,
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
                float(data["cost"]),
                float(data["emission"]),
                marker='x',  # square marker to differentiate
                color='black',
                s=80,
                label="_nolegend_"
            )
            
            # Add text label
            y_location = float(data["emission"]) + 0.012
            if fuel == "oil":
                y_location = float(data["emission"]) - 0.003
            x_location = float(data["cost"])
            if fuel == "oil": # right from the point
                x_location = float(data["cost"]) + 1.3
            plt.text(
                x_location,
                y_location,
                fuel,
                fontsize=fontsize,
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
                    biomass_variant[biomass_variant["Data Name"] == bt]["Values"].values[0].item()
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
                circle_radius = np.sqrt(float(sizes[i]) / np.pi)*0.95
                width = circle_radius * 0.09  # Scale factor for visual appearance
                height = width * adjustment_factor
                if usage >= 99.5:  # Special case for (nearly) 100% usage
                    # Draw a filled ellipse instead of a wedge
                    ellipse = mpatches.Ellipse(
                        (float(costs[i]), float(emissions[i])),
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
                        float(costs[i]), float(emissions[i]), width, height, theta1, theta2
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
                biomass[biomass["Data Name"] == bt]["Values"].values[0].item()
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
            circle_radius = np.sqrt(float(sizes[i]) / np.pi)*0.95 #for some reason the circles are too big so they need to be scaled down
            width = circle_radius * 0.09  # Scale factor for visual appearance
            height = width * adjustment_factor

            if usage >= 99.5:  # Special case for (nearly) 100% usage
                # Draw a filled ellipse instead of a wedge
                ellipse = mpatches.Ellipse(
                    (float(costs[i]), float(emissions[i])),
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
                    float(costs[i]), float(emissions[i]), width, height, theta1, theta2
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

    color_legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='green',
            markeredgecolor='green', label='Solid biomass', markersize=10, linewidth=0),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='lightgreen',
            markeredgecolor='lightgreen', label='Additional solid biomass use\nwith high co2 seq. potential', markersize=10, linewidth=0),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='blue',
            markeredgecolor='blue', label='Digestible biomass', markersize=10, linewidth=0),
        Line2D([0], [0], marker='x', color='none', markerfacecolor='black',
            markeredgecolor='black', label='Fossil fuels', markersize=10, linewidth=0),
        Line2D([0], [0], marker='x', color='none', markerfacecolor='orange',
            markeredgecolor='orange', label='Renewable energy', markersize=10, linewidth=0),
    ]
    if scenario is None or "Default" in scenario:
        color_legend_elements = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='green',
                markeredgecolor='green', label='Solid biomass', markersize=10, linewidth=0),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='blue',
                markeredgecolor='blue', label='Digestible biomass', markersize=10, linewidth=0),
            Line2D([0], [0], marker='x', color='none', markerfacecolor='black',
                markeredgecolor='black', label='Fossil fuels', markersize=10, linewidth=0),
            Line2D([0], [0], marker='x', color='none', markerfacecolor='orange',
                markeredgecolor='orange', label='Renewable energy', markersize=10, linewidth=0),
        ]
    else:
        color_legend_elements = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='green',
                markeredgecolor='green', label='Solid biomass', markersize=10, linewidth=0),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='lightgreen',
                markeredgecolor='lightgreen', label='Additional solid biomass use\nwith high co2 seq. potential', markersize=10, linewidth=0),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='blue',
                markeredgecolor='blue', label='Digestible biomass', markersize=10, linewidth=0),
            Line2D([0], [0], marker='x', color='none', markerfacecolor='black',
                markeredgecolor='black', label='Fossil fuels', markersize=10, linewidth=0),
            Line2D([0], [0], marker='x', color='none', markerfacecolor='orange',
                markeredgecolor='orange', label='Renewable energy', markersize=10, linewidth=0),
        ]


    if biomass_supply is None:
        color_legend_elements = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='green',
                markeredgecolor='green', label='Solid biomass', markersize=10, linewidth=0),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='blue',
                markeredgecolor='blue', label='Digestible biomass', markersize=10, linewidth=0),
            Line2D([0], [0], marker='x', color='none', markerfacecolor='black',
                markeredgecolor='black', label='Fossil fuels', markersize=10, linewidth=0),
            Line2D([0], [0], marker='x', color='none', markerfacecolor='orange',
                markeredgecolor='orange', label='Renewable energy', markersize=10, linewidth=0),
        ]

    legend2 = ax.legend(
        handles=color_legend_elements,
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
    error_bar_amount=None,
    fig_width=DEFAULT_FIGURE_WIDTH,
    fig_height=DEFAULT_FIGURE_HEIGHT,
    fontsize=DEFAULT_FONTSIZE,
    title_fontsize=DEFAULT_TITLE_FONTSIZE):

    file_path = f"{file_name}.{file_type}"
    # rename data_name column to Data Name
    df.rename(columns={"data_name": "Data Name"}, inplace=True)
    if custom_order is not None:
        df = reorder_data(df, custom_order)

    if remove_last_letters != 0:
        df["Data Name"] = df["Data Name"].str[:-remove_last_letters]

    df[column] = df[column] * multiplier
    for row in df.iterrows():
        if threshold is not None:
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

    # Generate a general palette (e.g., "viridis") with the number of unique data names
    unique_data_names = df["Data Name"].unique()
    palette = sns.color_palette("tab10", n_colors=len(unique_data_names))
    color_dict = {data_name: palette[i] for i, data_name in enumerate(unique_data_names)}


    # Create the bar plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Separate positive and negative bottoms for proper stacking
    bottom_pos = np.zeros(len(pivot_df.columns))
    bottom_neg = np.zeros(len(pivot_df.columns))

    specific_order = [
        "C&P_RW",
        "secondary forestry residues",
        "sawdust",
        "fuelwoodRW",
        "grasses",
        "solar-hsat",
        "fuelwood residues"  # This will be plotted last
    ]

    # Filter and reorder the pivot_df index based on the specific order
    stack_order = [layer for layer in specific_order if layer in pivot_df.index]

    bottom_pos = np.zeros(len(pivot_df.columns))
    bottom_neg = np.zeros(len(pivot_df.columns))

    for data_name in stack_order:
        values = pivot_df.loc[data_name].values

        pos_values = np.maximum(values, 0)
        neg_values = np.minimum(values, 0)

        # positive part
        if np.any(pos_values > 0):
            ax.bar(
                x, pos_values, bar_width,
                bottom=bottom_pos,
                label=new_names_dict.get(data_name, data_name),
                color = color_dict.get(data_name, "gray") 
            )
            bottom_pos += pos_values

        # negative part
        if np.any(neg_values < 0):
            ax.bar(
                x, neg_values, bar_width,
                bottom=bottom_neg,
                label=None if np.any(pos_values > 0) else data_name,
                color = color_dict.get(data_name, "gray") 
            )
            bottom_neg += neg_values
        
    if isinstance(error_bar_amount, (int, float)):
        tops = np.where(bottom_pos != 0, bottom_pos, bottom_neg)

        diff  = error_bar_amount*multiplier - tops          # signed difference to reference
        lower = np.where(diff < 0, -diff, 0.0)   # error below the top
        upper = np.where(diff > 0,  diff, 0.0)   # error above the top

        ax.errorbar(
            x, tops, yerr=[lower, upper],
            fmt='none', ecolor='black', elinewidth=1.5, capsize=6,
            label=r'Difference to high seq. pot. variant',
        )

    # Customise the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right")
    
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.7)

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
    fig_width=DEFAULT_FIGURE_WIDTH,
    fig_height=DEFAULT_FIGURE_HEIGHT,
    fontsize=DEFAULT_FONTSIZE,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
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
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bar_width = 0.35  # Width of the bars
    x = np.arange(len(df_plot.index))  # Positions for each Folder

    # Plot the stacked bars for carbon data
    bottom = np.zeros(len(df_plot.index))
    stacked_containers = []
    for column in [
        "Biogenic Carbon Sequestered",
        "Biogenic Carbon Utilised",
        "Biogenic Carbon Not Captured",
    ]:
        container = ax.bar(
            x - bar_width / 2, df_plot[column], bar_width, label=column, bottom=bottom
        )
        stacked_containers.append(container) 
        bottom += df_plot[column]

    if upstream_data is not None:
        # Plot the upstream emissions bars next to the stacked bars
        ax.bar(
            x + bar_width / 2,
            df_plot["Upstream Emissions"],
            bar_width,
            label="Land Carbon Stock Changes of Biomass",
            color="grey",
            alpha=0.6,
        )

    # Add labels and title
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=45, ha="right")

    # Add legend
    ax.legend(title="Legend", fontsize=fontsize)

    # Add percentages on the stacked bars
    for i in range(len(df_plot.index)):  # For each scenario
        for j, container in enumerate(stacked_containers):  # For each component
            bar = container[i]  # Get the bar for this scenario and component
            height = bar.get_height()
            if height == 0:  # Skip zero-height bars
                continue
                
            # Sum all components for this scenario
            total = sum(c[i].get_height() for c in stacked_containers)
            if total > 0:  # Avoid division by zero
                percentage = height / total * 100
                color = "black" if percentage < 4 else "white"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"{percentage:.1f}%",
                    ha="center", va="center",
                    fontsize=fontsize, color=color
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


def plot_costs_vs_prices(df, title, x_label, y_label, file_name, scenario, usage_dict,export_dir="export/plots",file_type="png",include_co2_costs=False, add_legend=True,
                         fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
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
    scenario_df = df[(df["folder"] == scenario)]

    # Filter for biomass types with available costs
    biomass_types = [
        "sludge",
        "manure",
        "residues from landscape care",
        "agricultural waste",
        "grasses",
        "woody crops",
        "fuelwood residues",
        "fuelwoodRW",
        "secondary forestry residues",
        "sawdust",
        "C&P_RW",
        "solid biomass import",
    ]

    biomass_df = scenario_df[
        (scenario_df["data_name"].isin(biomass_types))
        & (scenario_df["costs"].notnull())
    ]

    # Create export directory
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)

    # Set color palette
    palette = sns.color_palette("tab20", n_colors=len(biomass_types))
    color_mapping = {
        biomass: palette[i % len(palette)] for i, biomass in enumerate(biomass_types)
    }

    # Plotting
    if add_legend:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig, ax = plt.subplots(figsize=(fig_width*0.7, fig_height))
    ax.set_aspect("equal")  # Ensure equal scaling for both axes

    # Add diagonal line from (0, 0) without adding to the legend
    max_limit = 110
    #(max(biomass_df["costs"].max(), biomass_df["values"].max()) * 1.1)  # Add 10% headroom
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
        subset = biomass_df[biomass_df["data_name"] == biomass]
        if not subset.empty:
            for _, row in subset.iterrows():
                usage = usage_dict.get(biomass, 0)  # Default to 0 if not provided
                color = color_mapping[biomass]

                if include_co2_costs and "default" not in scenario.lower():
                    # If CO2 costs are included, adjust the costs
                    row["costs"] += row.get("CO2 costs", 0)


                if usage == 0:
                    # Empty circle (just outline)
                    ax.scatter(
                        row["costs"],
                        row["values"],
                        s=100,
                        facecolors="none",
                        edgecolors=color,
                        label=new_names_dict[biomass],
                    )
                elif usage >= 99:
                    # Fully filled circle
                    ax.scatter(
                        row["costs"],
                        row["values"],
                        s=100,
                        color=color,
                        label=new_names_dict[biomass],
                        alpha=0.8,
                    )
                else:
                    # Partially filled circle
                    theta1 = 90
                    theta2 = 90 - 360 * (usage / 100)
                    wedge = mpatches.Wedge(
                        (row["costs"], row["values"]),
                        1.6,
                        theta2,
                        theta1,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.8,
                    )
                    ax.add_patch(wedge)
                    ax.scatter(
                        row["costs"],
                        row["values"],
                        s=100,
                        facecolors="none",
                        edgecolors=color,
                        label=new_names_dict[biomass],
                        alpha=0.8,
                    )


                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="None",
                        markersize=10,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        label=new_names_dict[biomass],
                    )
                )
                # Add to legend handles
                # if usage == 0:
                #     legend_handles.append(
                #         mlines.Line2D(
                #             [],
                #             [],
                #             marker="o",
                #             linestyle="None",
                #             markersize=10,
                #             markerfacecolor="none",
                #             markeredgecolor=color,
                #             label=new_names_dict[biomass],
                #         )
                #     )
                # elif usage >= 99:
                #     legend_handles.append(
                #         mlines.Line2D(
                #             [],
                #             [],
                #             marker="o",
                #             linestyle="None",
                #             markersize=10,
                #             markerfacecolor=color,
                #             markeredgecolor=color,
                #             label=new_names_dict[biomass],
                #         )
                #     )
                # else:
                #     legend_handles.append(
                #         mpatches.Wedge(
                #             (0, 0),
                #             1,
                #             theta2,
                #             theta1,
                #             facecolor=color,
                #             edgecolor=color,
                #             alpha=0.8,
                #             label=new_names_dict[biomass],
                #         )
                #     )

    # Add legend before the diagonal line
    by_label = {handle.get_label(): handle for handle in legend_handles}
    if add_legend:
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

    return fig


def plot_costs_vs_prices_combined(df, usage_dict_default, usage_dict_carbon_costs, 
                                  export_dir="export/plots", file_type="png",
                                  fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, 
                                  fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
    """
    Plot both Default and Carbon Stock Changes scenarios side by side with a shared legend.
    
    Parameters
    ----------
    df : DataFrame
        The dataframe containing price and cost data.
    usage_dict_default : dict
        Dictionary mapping biomass types to their usage percentage for Default scenario.
    usage_dict_carbon_costs : dict  
        Dictionary mapping biomass types to their usage percentage for Carbon Stock Changes scenario.
    export_dir : str
        Directory to save the plot.
    file_type : str
        File type for saving the plot.
    fig_width, fig_height : float
        Figure dimensions.
    fontsize, title_fontsize : float
        Font sizes for text and titles.
    """
    
    # Filter for biomass types with available costs
    biomass_types = [
        "sludge",
        "manure", 
        "residues from landscape care",
        "agricultural waste",
        "grasses",
        "woody crops", 
        "fuelwood residues",
        "fuelwoodRW",
        "secondary forestry residues",
        "sawdust",
        "C&P_RW",
        "solid biomass import",
    ]
    
    # Set color palette
    palette = sns.color_palette("tab20", n_colors=len(biomass_types))
    color_mapping = {
        biomass: palette[i % len(palette)] for i, biomass in enumerate(biomass_types)
    }
    
    # Create figure with two subplots side by side with optimized spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height/1.2))
    plt.subplots_adjust(wspace=0.2)  # Optimal space between subplots
    
    # Add main title to the figure
    fig.suptitle("Weighted Feedstock Prices vs. Costs", fontsize=title_fontsize, y=0.88)
    
    scenarios = ["Default", "Carbon Stock Changes"]
    usage_dicts = [usage_dict_default, usage_dict_carbon_costs]
    axes = [ax1, ax2]
    
    legend_handles = []
    
    for scenario, usage_dict, ax in zip(scenarios, usage_dicts, axes):
        # Filter data for the specified scenario
        scenario_df = df[(df["folder"] == scenario)]
        biomass_df = scenario_df[
            (scenario_df["data_name"].isin(biomass_types))
            & (scenario_df["costs"].notnull())
        ]
        
        ax.set_aspect("equal")  # Ensure equal scaling for both axes
        
        # Add diagonal line from (0, 0) without adding to the legend
        max_limit = 110
        ax.plot(
            [0, max_limit],
            [0, max_limit],
            color="grey",
            linestyle="--",
            zorder=0,
            label="_nolegend_",
        )
        
        for biomass in biomass_types:
            subset = biomass_df[biomass_df["data_name"] == biomass]
            if not subset.empty:
                for _, row in subset.iterrows():
                    usage = usage_dict.get(biomass, 0)  # Default to 0 if not provided
                    color = color_mapping[biomass]
                    
                    if usage == 0:
                        # Empty circle (just outline)
                        ax.scatter(
                            row["costs"],
                            row["values"],
                            s=100,
                            facecolors="none",
                            edgecolors=color,
                            alpha=0.8,
                        )
                    elif usage >= 99:
                        # Fully filled circle
                        ax.scatter(
                            row["costs"],
                            row["values"],
                            s=100,
                            color=color,
                            alpha=0.8,
                        )
                    else:
                        # Partially filled circle
                        theta1 = 90
                        theta2 = 90 - 360 * (usage / 100)
                        wedge = mpatches.Wedge(
                            (row["costs"], row["values"]),
                            1.6,
                            theta2,
                            theta1,
                            facecolor=color,
                            edgecolor=color,
                            alpha=0.8,
                        )
                        ax.add_patch(wedge)
                        ax.scatter(
                            row["costs"],
                            row["values"],
                            s=100,
                            facecolors="none",
                            edgecolors=color,
                            alpha=0.8,
                        )
                    
                    # Add to legend handles (only once, for the first subplot)
                    if ax == ax1:
                        legend_handles.append(
                            mlines.Line2D(
                                [],
                                [],
                                marker="o",
                                linestyle="None",
                                markersize=10,
                                markerfacecolor=color,
                                markeredgecolor=color,
                                label=new_names_dict[biomass],
                            )
                        )
        
        # Set plot properties
        ax.set_xlim(0, max_limit)
        ax.set_ylim(0, max_limit)
        ax.set_xlabel("Costs in Euro/MWh", fontsize=fontsize)
        
        # Only add y-label and y-ticks to the left plot
        if ax == ax1:  # Left plot
            ax.set_ylabel("Prices in EUR/MWh", fontsize=fontsize)
        else:  # Right plot
            ax.set_ylabel("")  # No y-label
            ax.set_yticklabels([])  # No y-tick labels
            
        # Set subplot titles (scenarios)
        ax.set_title(f"{scenario}", fontsize=title_fontsize, loc="center")
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    # Add shared legend positioned to minimize space waste
    by_label = {handle.get_label(): handle for handle in legend_handles}
    fig.legend(
        by_label.values(),
        by_label.keys(),
        title="Biomass Types",
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),  # Position legend closer to the plots
        borderaxespad=0,
        fontsize=fontsize,
        title_fontsize=fontsize,
    )
    
    # Adjust layout for minimal space waste
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, top=0.82)  # Minimize both right margin and top spacing
    
    # Save the plot
    file_path = f"prices_costs_combined.{file_type}"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, file_path)
    
    if file_path.endswith(".pgf"):
        configure_for_pgf()
    
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0.3, dpi=300)
    plt.close()
    
    print(f"Combined costs vs prices plot saved to {file_path}")
    
    return fig


def plot_feedstock_prices(df, title, x_label, y_label, file_name, custom_order=None, export_dir="export/plots",file_type="png",
                          fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
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
    df = df[~df["data_name"].isin(["solid biomass"])]

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
    folders = df["folder"].unique()
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

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Adjust position for each scenario within each feedstock
    width = 0.3  # Bar width for spacing
    x = np.arange(len(feedstocks))  # Base positions

    for idx, scenario in enumerate(folders):
        scenario_df = df[df["folder"] == scenario]

        # Aggregate prices
        aggregated_data = {
            "solid biomass": scenario_df.loc[
                scenario_df["data_name"].isin(solid_biomass), "values"
            ].values,
            "digestible biomass": scenario_df.loc[
                scenario_df["data_name"].isin(digestible_biomass), "values"
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
    fig_width=DEFAULT_FIGURE_WIDTH,
    fig_height=DEFAULT_FIGURE_HEIGHT,
    fontsize=DEFAULT_FONTSIZE,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
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
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
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
            fontsize=fontsize,
            color="black",
        )

        # Share difference above or below the bars
        delta_text = f"Δ {share_delta * 100:.0f}%"
        ax.text(
            x[i],
            value / 1e6 + 0.08 * max(value_diff) / 1e6,
            delta_text,
            ha="center",
            fontsize=fontsize,
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
    fig_width=DEFAULT_FIGURE_WIDTH,
    fig_height=DEFAULT_FIGURE_HEIGHT,
    fontsize=DEFAULT_FONTSIZE,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
):
    file_path = f"{file_name}.{file_type}"
    # Pivot the DataFrame for plotting

    plt.rcParams.update({"font.size": fontsize})
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
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
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
                fontsize=fontsize/(len(custom_order)/2),
            )

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))  # Add 10% to the y-axis limit

    # Customise the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=title_fontsize)
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


def plot_shares(df, title, x_label, y_label, file_name, custom_order=None, export_dir="export/plots",file_type="png",
                fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
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
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

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
                ].values[0].item()  # Get value of share
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
                fontsize=fontsize//2,  # Half the normal fontsize for small labels
                rotation=90,
            )  # Added legend below each bar

    # Set X-Axis labels to the years
    ax.set_xticks(x_positions + ((num_folders - 1) * (bar_width + group_spacing) / 2))
    ax.set_xticklabels(unique_years)

    # Add labels and title
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(title, fontsize=title_fontsize)

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


def plot_costs(df, title, x_label, y_label, file_name, export_dir="export/plots",file_type="png",
               fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
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
    plt.figure(figsize=(fig_width, fig_height))

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
        ax.bar_label(container, fmt="%.0f", fontsize=fontsize, padding=5)

    # Add labels and title
    plt.title(title, fontsize=title_fontsize)
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
    data, title, x_label, y_label, file_name, custom_order=None, color_palette="viridis", export_dir="export/plots", file_type="png",
    fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE
):
    """
    Plot data
    """
    file_path = f"{file_name}.{file_type}"
    if custom_order is not None:
        data = reorder_data(data, custom_order)
    plt.rcParams.update({"font.size": fontsize})
    # Recreate the bar chart with the reordered folders
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(
        data=data, x="Year", y="Values", hue="Folder", palette=color_palette, orient="v"
    )
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))

    # Add values as labels above the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=fontsize, padding=5)

    # Add labels and title
    plt.title(title, fontsize=title_fontsize)
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


def plot_biomass_use(df, title, x_label, y_label, file_name, year=2050,export_dir="export/plots",file_type="png", labels=True,
                     fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
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
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bar_width = 0.8
    gap = 0.6  # Adjust this value to change the space between bars
    x_coords = np.arange(len(df_pivot)) * (bar_width + gap)

    for index, row in df_pivot.iterrows():
        biomass_type = row["Data Name"]
        value_a = row["Default"]
        value_b = row["Carbon Stock Changes"]
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
                fontsize=fontsize//2,  # Smaller fontsize for detailed annotations
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
    plt.legend(handles=[potential_patch, a_patch, b_patch], fontsize=fontsize)

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.3))  # Add 10% to the y-axis limit

    # Customize plot
    ax.set_xticks(x_coords)
    # new_labels = [f"{name}\n{emission_factors.get(name, 'N/A')} | {emission_factors.get(name, 'N/A') / 0.0036 if emission_factors.get(name, 'N/A') != 'N/A' else 'N/A'} \n (ton/MWh | g/MJ)" for name in df_pivot['Data Name']]
    ax.set_xticklabels(df_pivot["Data Name"], rotation=45, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=title_fontsize)

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


def plot_efs(export_dir="export/plots",file_type="png",
             fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
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
    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
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
            fontsize=fontsize,
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
        fontsize=fontsize,
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
        fontsize=fontsize,
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
        fontsize=fontsize,
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
    ax.set_title("Emission Factors and Costs for Different Feedstocks", fontsize=title_fontsize)

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

def plot_efs_for_presentation(export_dir="export/plots",file_type="png",
                              fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):

    # font size
    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bar_width = 0.8
    emission_factors = emission_factors_new_names

    # Calculate the average emission factors for the new entries
    to_remove_stemwood = ["chips and pellets", "stemwood", "secondary forestry residues", "sawdust"]
    stemwood_based_biomass = sum(emission_factors[x] for x in to_remove_stemwood) / len(to_remove_stemwood)

    to_remove_herbaceous = ["grasses", "woody crops"]
    woody_and_herbaceous_crops = sum(emission_factors[x] for x in to_remove_herbaceous) / len(to_remove_herbaceous)

    # Create a new dictionary with the new entries first
    emission_factors = {        
        "woody and herbaceous crops": woody_and_herbaceous_crops,
        "crop residues": emission_factors["crop residues"],
        "stemwood-based biomass": stemwood_based_biomass,
        "logging residues": emission_factors["logging residues"],
        **{k: v for k, v in emission_factors.items() if k not in to_remove_stemwood + to_remove_herbaceous +["logging residues","crop residues"]},
    }

    gap = 0.6  # Adjust this value to change the space between bars
    x_coords = np.arange(len(emission_factors)) * (bar_width + gap)

    for index, (biomass, ef) in enumerate(emission_factors.items()):
        ax.bar(x_coords[index], ef, width=bar_width, label=biomass, color="LightGreen")

    # Add gas emission factor
    x_coords = np.append(x_coords, [x_coords[-1] + (bar_width + gap)])
    ax.bar(x_coords[-1], 0.198, width=bar_width, label="natural gas", color="Grey")

    # Add oil emission factor
    x_coords = np.append(x_coords, [x_coords[-1] + (bar_width + gap)])
    ax.bar(x_coords[-1], 0.2571, width=bar_width, label="oil", color="Brown")

    # Add coal emission factor
    x_coords = np.append(x_coords, [x_coords[-1] + (bar_width + gap)])
    ax.bar(x_coords[-1], 0.3361, width=bar_width, label="coal", color="Black")

    ax.set_xticks(x_coords)
    # ax.set_xticklabels(emission_factors.keys(), rotation=45, ha="right")
    ax.set_xticklabels(
        list(emission_factors.keys()) + ["natural gas", "oil", "coal"],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("")
    ax.set_ylabel("tonCO2/MWh")
    ax.set_title("Carbon Stock Changes and Emission Factors for Different Feedstocks", fontsize=title_fontsize)

    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks() / 0.0036)
    ax2.set_ylabel("g/MJ")

    secondary_locator = MultipleLocator(10)  # Adjust this value as needed
    ax2.yaxis.set_major_locator(secondary_locator)

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.3))

    plt.tight_layout()

    file_path = f"emission_factors_presentation.{file_type}"
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
    fig_width=DEFAULT_FIGURE_WIDTH,
    fig_height=DEFAULT_FIGURE_HEIGHT,
    fontsize=DEFAULT_FONTSIZE,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
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
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
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
                fontsize=fontsize,
            )

    # Adjust plot limits to add space for text
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + abs(ylim[1] * 0.1))  # Add 10% to the y-axis limit

    # Customise the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right", fontsize=fontsize)
    ax.legend(title="Scenario", fontsize=fontsize)

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

def plot_co2(df, scenario, file_name, export_dir="export/plots",file_type="png", unit = "ton", multiplier=1,
             fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):

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
    color_links_by_technology=True,
    include_legend=True,
    fixed = True,
    fig_width=DEFAULT_FIGURE_WIDTH,
    fig_height=DEFAULT_FIGURE_HEIGHT,
    fontsize=DEFAULT_FONTSIZE,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
):
    """
    Generate a Sankey diagram from a DataFrame of carbon flows for a given scenario.

    Parameters:
        df (pd.DataFrame): Input data with [folder, data_name, values, from_sink, to_sink].
        scenario (str): Scenario to plot (filters 'folder').
        multiplier (float): Multiplier for value scaling.
        output_dir (str): Output directory for the saved plot.
        unit_label (str): Unit string shown in hover tooltips.
        color_links_by_technology (bool): Whether to color flows by technology.
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

    # Sink type node coloring
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
        elif "utilization" in n or "utilised" in n or "utilized" in n or ("waste" in n and scenario == "Default"):
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
        if color_links_by_technology:
            link_colors.append(tech_colors[row['data_name']])
        else:
            link_colors.append("rgba(150,150,150,0.4)")  # uniform grey if not colored


    # Node throughput for hover info
    node_in = {i: 0 for i in range(len(nodes))}
    node_out = {i: 0 for i in range(len(nodes))}
    for s, t, v in zip(sources, targets, values):
        node_out[s] += v
        node_in[t] += v

    throughput = [max(node_in[i], node_out[i]) for i in range(len(nodes))]

    if fixed:
        # Define node category positions (x-coordinates)
        position_categories = {
            "Other": 0.25,  # Default position for other nodes
            "CO2 Captured": 0.8,  # Capture nodes
            "CO2 Storage": 0.99,  # Sequester/sink nodes
            "Atmosphere": 0.6,  # Atmosphere nodes
            "CO2 Utilization": 0.99  # Utilization nodes
        }
        
        # Map each node to a category based on name
        node_categories = {}
        for node in nodes:
            n = node.lower()
            node_categories[node] = classify_sink(node)
        
        # Create node_x and node_y lists
        node_x = [position_categories[node_categories[node]] for node in nodes]
        
        y_positions = {}
        for category in position_categories:
            nodes_in_category = [n for n in nodes if node_categories[n] == category]
            if category == "Other":
                y_dict = {
                    "liquid fuels": 0.8,
                    "gas": 0.55,
                    "municipal solid waste": 0.05,
                    "crude oil": 0.4,
                    "HVC": 0.45,
                    "indirect emissions from biomass": 0.3,
                    "indirect emissions from renewables": 0.35,
                    "cement": 0.2,
                }
                for node in nodes_in_category:
                    y_positions[node] = y_dict.get(node, 0.2)
            elif category == "CO2 Captured":
                for node in nodes_in_category:
                    y_positions[node] = 0.4
            elif category == "CO2 Storage":
                for node in nodes_in_category:
                    y_positions[node] = 0.3
            elif category == "Atmosphere":
                for node in nodes_in_category:
                    y_positions[node] = 0.6
            elif category == "CO2 Utilization":
                for node in nodes_in_category:
                    y_positions[node] = 0.7
        node_y = [y_positions[node] for node in nodes]

        fig = go.Figure(go.Sankey(
            arrangement="fixed",
            node=dict(
                label=[f"{n}<br><span style='font-size:10px'>({throughput[node_index[n]]:,.0f} {unit_label})</span>" for n in nodes],
                x=node_x,
                y=node_y,
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
    else:
        # Original code for non-fixed arrangement
        fig = go.Figure(go.Sankey(
            arrangement="perpendicular",
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
        if color_links_by_technology:
            for tech, color in tech_colors.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color, symbol="square"),
                    name=tech,
                    legendgroup="tech",
                    legendgrouptitle_text="Technology"
                ))
        for cat, color in sink_type_colors.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="circle"),
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

def make_cost_diffs(df, name_col="Folder", cost_col="Values"):
    s = df.set_index(name_col)[cost_col]
    return {
        "cscs":        (s["cscs_bm0"]            - s["cscs"])              / s["cscs"],
        "cscs_710":    (s["cscs_710_bm0"]        - s["cscs_710"])          / s["cscs_710"],
        "default":     (s["default_bm0"]    - s["default"])      / s["default"],
        "default_710": (s["default_710_bm0"]- s["default_710"])  / s["default_710"],
    }

def plot_mga(df, file_name, title="Near Optimal Biomass Use", export_dir='export/plots',
             file_type='png', unit='MWh', multiplier=1, y_range=(0, 4300),
             allow_incomplete=True, zero_lower_pct=None,
             include_fossil=False, fossil_df=None, fossil_unit='MWh', fossil_multiplier=1,
             fossil_y_range=None, fossil_breakdown=False,
             fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT,
             fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
    """
    Plot the near‑optimal solution space for biomass use under various cost deviations.
    Optionally include fossil fuel data on a secondary y-axis.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with MGA results. Expected columns:
            - 'Folder'     Scenario name (e.g. 'optimal', 'max_0.05', 'min_0.05', ...)
            - 'Year'       (optional) year or time period (ignored here)
            - 'Data Name'  (optional) metric name (should contain 'biomass')
            - 'Values'     Numeric values of the metric (biomass usage)
        The frame must include an 'optimal' scenario and matching 'max_X'/'min_X' pairs
        for each cost deviation X.
    file_name : str
        Base name for the saved plot file (without extension).
    export_dir : str, default 'export/plots'
        Directory in which to save the plot.
    file_type : str, default 'png'
        File format for saving (e.g. 'png', 'pdf').
    unit : str, default 'MWh'
        Unit label for biomass values (y‑axis).
    multiplier : float, default 1
        Factor for scaling the biomass values (useful for unit conversion).
    y_range : tuple or None, default (0, 4300)
        Limits for the y‑axis.  If None, let matplotlib choose.
    allow_incomplete : bool, default True
        Whether to plot cost deviations even if either the min or max value is missing.
    zero_lower_pct : float | list[float] | None, default None
        One or several cost‑deviation percentages at which the lower bound is forced to
        zero. Can be passed as fractions (e.g., 0.04 for 4%) or percentages (e.g., 4.0).
        All min values at this percentage and higher will be set to zero.
    include_fossil : bool, default False
        Whether to include fossil fuel data on a secondary y-axis.
    fossil_df : pandas.DataFrame or None, default None
        DataFrame with fossil fuel MGA results. Same structure as df.
    fossil_unit : str, default 'MWh'
        Unit label for fossil fuel values (secondary y-axis).
    fossil_multiplier : float, default 1
        Factor for scaling the fossil fuel values.
    fossil_y_range : tuple or None, default None
        Limits for the fossil fuel y-axis. If None, let matplotlib choose.
    fossil_breakdown : bool, default False
        If True, show gas and oil individually on secondary axis.
        If False, show total fossil fuel supply.
    """
    os.makedirs(export_dir, exist_ok=True)

    # Filter to biomass rows if Data Name column exists
    if "Data Name" in df.columns:
        mask = df["Data Name"].str.contains("biomass", case=False, na=False)
        df_plot = df[mask].copy() if mask.any() else df.copy()
    else:
        df_plot = df.copy()

    # Initialize data containers
    min_vals, max_vals = {}, {}
    original_min, original_max = set(), set()
    optimal_value = None

    # Parse scenario data
    for _, row in df_plot.iterrows():
        scenario = str(row["Folder"]).lower()
        val = float(row["Values"]) * multiplier

        if "optimal" in scenario:
            optimal_value = val
            min_vals[0] = max_vals[0] = val
            original_min.add(0)
            original_max.add(0)
            continue

        # Parse min/max scenarios
        for kind in ("min_", "max_"):
            if kind in scenario:
                dev = scenario.split(kind, 1)[1]
                try:
                    frac = float(dev)
                except ValueError:
                    frac = float(dev.strip("%")) / 100.0
                pct = frac * 100
                
                if kind == "min_":
                    min_vals[pct] = val
                    original_min.add(pct)
                else:
                    max_vals[pct] = val
                    original_max.add(pct)
                break

    if optimal_value is None:
        raise ValueError("Missing 'optimal' scenario in DataFrame.")

    # Apply zero lower bounds if requested
    if zero_lower_pct is not None:
        zero_pcts = [float(zero_lower_pct)] if isinstance(zero_lower_pct, (int, float)) else [float(p) for p in zero_lower_pct]
        
        for pct in zero_pcts:
            # Convert fraction to percentage if needed
            if pct <= 1.0:
                pct = pct * 100.0
            
            # Set this percentage and all higher percentages to 0 for min_vals
            for existing_pct in list(min_vals.keys()):
                if existing_pct >= pct:
                    min_vals[existing_pct] = 0.0
            
            # Add the specific percentage if it doesn't exist
            min_vals[pct] = 0.0
            original_min.add(pct)

            print(f"INFO: Applied zero lower bound for {pct:.1f}% cost deviation and all higher deviations in scenario '{file_name}'.")

    # Interpolation function
    def interpolate(target, originals, target_type="values"):
        """Interpolate missing values in target dictionary."""
        all_devs = sorted(set(min_vals) | set(max_vals))
        
        for dev in all_devs:
            if dev in target:
                continue
                
            # Find bounding values
            lower_bounds = [d for d in target.keys() if d < dev]
            upper_bounds = [d for d in target.keys() if d > dev]
            
            lower = max(lower_bounds) if lower_bounds else None
            upper = min(upper_bounds) if upper_bounds else None
            
            # Handle missing bounds
            if lower is None:
                # No lower bound: extrapolate from 0 with constant value
                lower = 0.0
                if lower not in target:
                    target[lower] = 0.0
                target[dev] = target[lower]  # Use constant extrapolation
                print(f"Warning: No lower bound for {dev:.1f}% in {target_type}; extrapolating with constant value from 0%")
                continue
                
            if upper is None:
                # No upper bound: extrapolate with constant value from highest available point
                if not target:
                    raise ValueError("No data points available for interpolation")
                upper = max(target.keys())
                target[dev] = target[upper]  # Use constant extrapolation
                print(f"Warning: No upper bound for {dev:.1f}% in {target_type}; extrapolating with constant value from {upper:.1f}%")
                continue
            
            # Perform interpolation/extrapolation
            if lower == upper:
                target[dev] = target[lower]
            else:
                r = (dev - lower) / (upper - lower)
                target[dev] = target[lower] + r * (target[upper] - target[lower])

    # Perform interpolation
    interpolate(min_vals, original_min, "min values")
    interpolate(max_vals, original_max, "max values")

    # Process fossil fuel data if provided
    fossil_min_vals, fossil_max_vals = {}, {}
    fossil_original_min, fossil_original_max = set(), set()
    fossil_optimal_value = None
    
    # For breakdown: separate gas and oil data
    gas_min_vals, gas_max_vals = {}, {}
    oil_min_vals, oil_max_vals = {}, {}
    gas_original_min, gas_original_max = set(), set()
    oil_original_min, oil_original_max = set(), set()
    gas_optimal_value, oil_optimal_value = None, None
    
    if include_fossil and fossil_df is not None:
        # Filter fossil fuel data
        if "Data Name" in fossil_df.columns:
            if fossil_breakdown:
                # Filter for gas and oil separately using exact matches
                gas_df_plot = fossil_df[fossil_df["Data Name"] == "gas"].copy()
                oil_df_plot = fossil_df[fossil_df["Data Name"] == "oil primary"].copy()
                # If no exact matches, try partial matches
                if gas_df_plot.empty:
                    gas_mask = fossil_df["Data Name"].str.contains("gas", case=False, na=False)
                    gas_df_plot = fossil_df[gas_mask].copy() if gas_mask.any() else None
                if oil_df_plot.empty:
                    oil_mask = fossil_df["Data Name"].str.contains("oil", case=False, na=False)
                    oil_df_plot = fossil_df[oil_mask].copy() if oil_mask.any() else None
            else:
                # Use total fossil fuel data - sum gas and oil if breakdown data available
                gas_mask = fossil_df["Data Name"] == "gas"
                oil_mask = fossil_df["Data Name"] == "oil primary"
                if gas_mask.any() and oil_mask.any():
                    # Sum gas and oil values for each scenario
                    gas_data = fossil_df[gas_mask].copy()
                    oil_data = fossil_df[oil_mask].copy()
                    # Create combined dataframe
                    fossil_df_plot = gas_data.copy()
                    fossil_df_plot["Values"] = gas_data["Values"].values + oil_data["Values"].values
                    fossil_df_plot["Data Name"] = "total fossil fuel"
                else:
                    # Fallback to any fossil data
                    fossil_mask = fossil_df["Data Name"].str.contains("fossil", case=False, na=False)
                    fossil_df_plot = fossil_df[fossil_mask].copy() if fossil_mask.any() else fossil_df.copy()
        else:
            if fossil_breakdown:
                # If no Data Name column, assume the dataframe contains the right data
                gas_df_plot = fossil_df.copy()
                oil_df_plot = fossil_df.copy()  # User should provide separate data
            else:
                fossil_df_plot = fossil_df.copy()

        if fossil_breakdown and gas_df_plot is not None and oil_df_plot is not None and not gas_df_plot.empty and not oil_df_plot.empty:
            # Process gas data
            for _, row in gas_df_plot.iterrows():
                scenario = str(row["Folder"]).lower()
                val = float(row["Values"]) * fossil_multiplier

                if "optimal" in scenario:
                    gas_optimal_value = val
                    gas_min_vals[0] = gas_max_vals[0] = val
                    gas_original_min.add(0)
                    gas_original_max.add(0)
                    continue

                for kind in ("min_", "max_"):
                    if kind in scenario:
                        dev = scenario.split(kind, 1)[1]
                        try:
                            frac = float(dev)
                        except ValueError:
                            frac = float(dev.strip("%")) / 100.0
                        pct = frac * 100
                        
                        if kind == "min_":
                            gas_min_vals[pct] = val
                            gas_original_min.add(pct)
                        else:
                            gas_max_vals[pct] = val
                            gas_original_max.add(pct)
                        break

            # Process oil data
            for _, row in oil_df_plot.iterrows():
                scenario = str(row["Folder"]).lower()
                val = float(row["Values"]) * fossil_multiplier

                if "optimal" in scenario:
                    oil_optimal_value = val
                    oil_min_vals[0] = oil_max_vals[0] = val
                    oil_original_min.add(0)
                    oil_original_max.add(0)
                    continue

                for kind in ("min_", "max_"):
                    if kind in scenario:
                        dev = scenario.split(kind, 1)[1]
                        try:
                            frac = float(dev)
                        except ValueError:
                            frac = float(dev.strip("%")) / 100.0
                        pct = frac * 100
                        
                        if kind == "min_":
                            oil_min_vals[pct] = val
                            oil_original_min.add(pct)
                        else:
                            oil_max_vals[pct] = val
                            oil_original_max.add(pct)
                        break

            # Interpolate gas and oil data
            if gas_min_vals or gas_max_vals:
                interpolate(gas_min_vals, gas_original_min, "gas min values")
                interpolate(gas_max_vals, gas_original_max, "gas max values")
            if oil_min_vals or oil_max_vals:
                interpolate(oil_min_vals, oil_original_min, "oil min values")
                interpolate(oil_max_vals, oil_original_max, "oil max values")
        
        elif not fossil_breakdown:
            # Process total fossil fuel data (original logic)
            for _, row in fossil_df_plot.iterrows():
                scenario = str(row["Folder"]).lower()
                val = float(row["Values"]) * fossil_multiplier

                if "optimal" in scenario:
                    fossil_optimal_value = val
                    fossil_min_vals[0] = fossil_max_vals[0] = val
                    fossil_original_min.add(0)
                    fossil_original_max.add(0)
                    continue

                for kind in ("min_", "max_"):
                    if kind in scenario:
                        dev = scenario.split(kind, 1)[1]
                        try:
                            frac = float(dev)
                        except ValueError:
                            frac = float(dev.strip("%")) / 100.0
                        pct = frac * 100
                        
                        if kind == "min_":
                            fossil_min_vals[pct] = val
                            fossil_original_min.add(pct)
                        else:
                            fossil_max_vals[pct] = val
                            fossil_original_max.add(pct)
                        break

            # Interpolate fossil fuel data
            if fossil_min_vals or fossil_max_vals:
                interpolate(fossil_min_vals, fossil_original_min, "fossil min values")
                interpolate(fossil_max_vals, fossil_original_max, "fossil max values")

    # Prepare plotting data
    cost_devs = sorted(set(min_vals) | set(max_vals))
    x_vals, y_min, y_max = [], [], []
    
    for d in cost_devs:
        if allow_incomplete or (d in min_vals and d in max_vals):
            x_vals.append(d)
            y_min.append(min_vals.get(d, 0))
            y_max.append(max_vals.get(d, 0))

    # Create plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot biomass solution space
    ax.fill_between(x_vals, y_min, y_max, color="grey", alpha=0.3,
                    label="Near optimal solution space (biomass)")
    
    # Plot biomass min/max lines
    ax.plot(x_vals, y_min, color="tab:blue", linewidth=2, label="Min biomass use")
    ax.plot(x_vals, y_max, color="tab:orange", linewidth=2, label="Max biomass use")
    
    # Mark original (non-interpolated) biomass points
    original_min_x = [d for d in x_vals if d in original_min]
    original_max_x = [d for d in x_vals if d in original_max]
    
    if original_min_x:
        ax.scatter(original_min_x, [min_vals[d] for d in original_min_x],
                   color="tab:blue", marker="o", s=50, zorder=5)
    if original_max_x:
        ax.scatter(original_max_x, [max_vals[d] for d in original_max_x],
                   color="tab:orange", marker="o", s=50, zorder=5)
    
    # Mark optimal biomass solution
    ax.plot([0], [optimal_value], color="black", marker="o", markersize=10,
            linestyle="none", label="Cost optimal solution (biomass)", zorder=6)

    # Add fossil fuel data on secondary axis if requested
    ax2 = None
    if include_fossil and fossil_df is not None:
        ax2 = ax.twinx()
        
        if fossil_breakdown and ((gas_min_vals or gas_max_vals) or (oil_min_vals or oil_max_vals)):
            # Plot gas and oil separately using consistent blue/orange colors
            if gas_min_vals or gas_max_vals:
                gas_cost_devs = sorted(set(gas_min_vals) | set(gas_max_vals))
                gas_x_vals, gas_y_min, gas_y_max = [], [], []
                
                for d in gas_cost_devs:
                    if allow_incomplete or (d in gas_min_vals and d in gas_max_vals):
                        gas_x_vals.append(d)
                        gas_y_min.append(gas_min_vals.get(d, 0))
                        gas_y_max.append(gas_max_vals.get(d, 0))
                
                # Plot gas lines with blue/orange colors and dotted lines
                ax2.plot(gas_x_vals, gas_y_min, color="tab:blue", linewidth=2, linestyle="dotted", 
                        label="Gas use at min biomass", alpha=0.8)
                ax2.plot(gas_x_vals, gas_y_max, color="tab:orange", linewidth=2, linestyle="dotted", 
                        label="Gas use at max biomass", alpha=0.8)
                
                # Mark original gas points with square markers
                gas_original_min_x = [d for d in gas_x_vals if d in gas_original_min]
                gas_original_max_x = [d for d in gas_x_vals if d in gas_original_max]
                
                if gas_original_min_x:
                    ax2.scatter(gas_original_min_x, [gas_min_vals[d] for d in gas_original_min_x],
                               color="tab:blue", marker="s", s=50, zorder=5, alpha=0.8)
                if gas_original_max_x:
                    ax2.scatter(gas_original_max_x, [gas_max_vals[d] for d in gas_original_max_x],
                               color="tab:orange", marker="s", s=50, zorder=5, alpha=0.8)
                
                # Mark optimal gas solution with square marker
                if gas_optimal_value is not None:
                    ax2.plot([0], [gas_optimal_value], color="black", marker="s", markersize=10,
                            linestyle="none", label="Cost optimal solution (gas)", zorder=6, alpha=0.8)
            
            if oil_min_vals or oil_max_vals:
                oil_cost_devs = sorted(set(oil_min_vals) | set(oil_max_vals))
                oil_x_vals, oil_y_min, oil_y_max = [], [], []
                
                for d in oil_cost_devs:
                    if allow_incomplete or (d in oil_min_vals and d in oil_max_vals):
                        oil_x_vals.append(d)
                        oil_y_min.append(oil_min_vals.get(d, 0))
                        oil_y_max.append(oil_max_vals.get(d, 0))
                
                # Plot oil lines with blue/orange colors and dash-dot lines
                ax2.plot(oil_x_vals, oil_y_min, color="tab:blue", linewidth=2, linestyle="dashdot", 
                        label="Oil use at min biomass", alpha=0.8)
                ax2.plot(oil_x_vals, oil_y_max, color="tab:orange", linewidth=2, linestyle="dashdot", 
                        label="Oil use at max biomass", alpha=0.8)
                
                # Mark original oil points with triangle markers
                oil_original_min_x = [d for d in oil_x_vals if d in oil_original_min]
                oil_original_max_x = [d for d in oil_x_vals if d in oil_original_max]
                
                if oil_original_min_x:
                    ax2.scatter(oil_original_min_x, [oil_min_vals[d] for d in oil_original_min_x],
                               color="tab:blue", marker="^", s=50, zorder=5, alpha=0.8)
                if oil_original_max_x:
                    ax2.scatter(oil_original_max_x, [oil_max_vals[d] for d in oil_original_max_x],
                               color="tab:orange", marker="^", s=50, zorder=5, alpha=0.8)
                
                # Mark optimal oil solution with triangle marker
                if oil_optimal_value is not None:
                    ax2.plot([0], [oil_optimal_value], color="black", marker="^", markersize=10,
                            linestyle="none", label="Cost optimal solution (oil)", zorder=6, alpha=0.8)
        
        elif not fossil_breakdown and (fossil_min_vals or fossil_max_vals):
            # Plot total fossil fuels (original logic)
            fossil_cost_devs = sorted(set(fossil_min_vals) | set(fossil_max_vals))
            fossil_x_vals, fossil_y_min, fossil_y_max = [], [], []
            
            for d in fossil_cost_devs:
                if allow_incomplete or (d in fossil_min_vals and d in fossil_max_vals):
                    fossil_x_vals.append(d)
                    fossil_y_min.append(fossil_min_vals.get(d, 0))
                    fossil_y_max.append(fossil_max_vals.get(d, 0))
            
            # Plot fossil fuel lines (corresponding to biomass boundaries, not feasible space)
            # Use similar colors to biomass but distinguish as fossil fuel data
            ax2.plot(fossil_x_vals, fossil_y_min, color="tab:blue", linewidth=2, linestyle="dotted", 
                    label="Fossil fuel use at min biomass", alpha=0.8)
            ax2.plot(fossil_x_vals, fossil_y_max, color="tab:orange", linewidth=2, linestyle="dotted", 
                    label="Fossil fuel use at max biomass", alpha=0.8)
            
            # Mark original fossil fuel points with same markers as corresponding biomass data
            fossil_original_min_x = [d for d in fossil_x_vals if d in fossil_original_min]
            fossil_original_max_x = [d for d in fossil_x_vals if d in fossil_original_max]
            
            if fossil_original_min_x:
                ax2.scatter(fossil_original_min_x, [fossil_min_vals[d] for d in fossil_original_min_x],
                           color="tab:blue", marker="o", s=50, zorder=5, alpha=0.8)
            if fossil_original_max_x:
                ax2.scatter(fossil_original_max_x, [fossil_max_vals[d] for d in fossil_original_max_x],
                           color="tab:orange", marker="o", s=50, zorder=5, alpha=0.8)
            
            # Mark optimal fossil fuel solution
            if fossil_optimal_value is not None:
                ax2.plot([0], [fossil_optimal_value], color="black", marker="o", markersize=10,
                        linestyle="none", label="Cost optimal solution (fossil)", zorder=6, alpha=0.8)
        
        # Customize secondary axis
        if fossil_breakdown:
            ax2.set_ylabel(f"Gas & Oil use ({fossil_unit})", color="black")
        else:
            ax2.set_ylabel(f"Fossil fuel use ({fossil_unit})", color="black")
        ax2.tick_params(axis='y', labelcolor="black")
        if fossil_y_range:
            ax2.set_ylim(fossil_y_range)

    # Customize plot
    ax.set_xlabel("ε (%)")
    ax.set_ylabel(f"Biomass use ({unit})")
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Set x-axis ticks with smart spacing to avoid crowded labels
    ax.set_xticks(x_vals)
    
    # Create labels with minimum 1% spacing
    min_spacing = 1.0  # Minimum spacing between labels in percentage points
    tick_labels = []
    last_labeled = None
    
    for i, x in enumerate(x_vals):
        # Always label the first point (0) and ensure minimum spacing
        if x == 0 or last_labeled is None or abs(x - last_labeled) >= min_spacing:
            label = f"{x:.0f}" if x == int(x) else f"{x:.0f}"
            tick_labels.append(label)
            last_labeled = x
        else:
            tick_labels.append("")  # Empty label for points too close together
    
    # Always label the last point if it wasn't already labeled
    if x_vals and last_labeled != x_vals[-1]:
        last_x = x_vals[-1]
        tick_labels[-1] = f"{last_x:.0f}" if last_x == int(last_x) else f"{last_x:.0f}"
    
    ax.set_xticklabels(tick_labels)
    
    if y_range:
        ax.set_ylim(y_range)

    # Order legend consistently
    if include_fossil and ax2 is not None:
        # Combined legend for both axes
        if fossil_breakdown:
            legend_order = [
                "Cost optimal solution (biomass)", "Max biomass use", "Min biomass use", 
                "Near optimal solution space (biomass)",
                "Cost optimal solution (gas)", "Gas use at max biomass", "Gas use at min biomass",
                "Cost optimal solution (oil)", "Oil use at max biomass", "Oil use at min biomass"
            ]
        else:
            legend_order = [
                "Cost optimal solution (biomass)", "Max biomass use", "Min biomass use", 
                "Near optimal solution space (biomass)",
                "Cost optimal solution (fossil)", "Fossil fuel use at max biomass", "Fossil fuel use at min biomass"
            ]
        
        # Get handles and labels from both axes
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        
        all_handles = handles1 + handles2
        all_labels = labels1 + labels2
        
        ordered_handles = []
        ordered_labels = []
        
        for desired_label in legend_order:
            for handle, label in zip(all_handles, all_labels):
                if label == desired_label:
                    ordered_handles.append(handle)
                    ordered_labels.append(label)
                    break
        
        ax.legend(ordered_handles, ordered_labels, loc="best", frameon=True)
    else:
        # Original single-axis legend
        legend_order = [
            "Cost optimal solution (biomass)", "Max biomass use",
            "Min biomass use", "Near optimal solution space (biomass)",
        ]
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles = []
        ordered_labels = []
        
        for desired_label in legend_order:
            for handle, label in zip(handles, labels):
                if label == desired_label:
                    ordered_handles.append(handle)
                    ordered_labels.append(label)
                    break
        
        ax.legend(ordered_handles, ordered_labels, loc="best", frameon=True)

    # Save plot
    out_path = os.path.join(export_dir, f"{file_name}.{file_type}")
    plt.savefig(out_path, format=file_type, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    print(f"MGA plot saved to {out_path}")
    return fig

def plot_stacked_biomass_with_errorbars(
    data,                                # pandas.DataFrame
    export_dir="export/plots",
    file_name="biomass_stacked_errorbar",
    file_type="png",                     # "png", "pgf", or "tex"
    errorbars=True,
    fig_width=DEFAULT_FIGURE_WIDTH,
    fig_height=DEFAULT_FIGURE_HEIGHT,
    fontsize=DEFAULT_FONTSIZE,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
):
    # ── 1. aggregate 2050 data ───────────────────────────────────────
    df = data.loc[data["Year"] == 2050].copy()
    base = ["Default", "Carbon Stock Changes"]
    variant_suffix = " 710"
    sectors = df["Data Name"].unique()

    baseline_data = {}
    high_variant_data = {}

    for scen in base:
        baseline_data[scen] = (
            df.query("Folder == @scen")
              .set_index("Data Name")["Values"]
              .reindex(sectors, fill_value=0)
              .div(1e6)
              .to_dict()
        )
        high_scen = scen + variant_suffix
        high_variant_data[scen] = (
            df.query("Folder == @high_scen")
              .set_index("Data Name")["Values"]
              .reindex(sectors, fill_value=0)
              .div(1e6)
              .to_dict()
        )
        # Add total for error bar calc
        high_variant_data[scen]["TOTAL"] = sum(high_variant_data[scen].values())

    # ── 2. plot ───────────────────────────────────────────────────────
    categories = base
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    width = 0.8

    bottom = np.zeros_like(x, dtype=float)
    for sector in sectors:
        sector_values = np.array([baseline_data[cat][sector] for cat in categories], dtype=float)
        ax.bar(x, sector_values, width, bottom=bottom, label=sector)
        bottom += sector_values

    if errorbars:
        total_baseline = bottom
        diff_high = np.array([
            high_variant_data[cat]["TOTAL"] - total_baseline[i]
            for i, cat in enumerate(categories)
        ], dtype=float)

        # Ensure error bars are non-negative: split into upper and lower
        lower = np.where(diff_high < 0, -diff_high, 0)
        upper = np.where(diff_high > 0, diff_high, 0)
        y_err = [lower, upper]

        err = ax.errorbar(
            x, total_baseline, yerr=y_err, fmt='none', ecolor='black',
            elinewidth=1.5, capsize=6, label='Difference to high seq. pot. variant.'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Total Biomass Use (TWh)')
    ax.set_title('Biomass Use by Sector', fontsize=title_fontsize)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Legend deduplication
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()

    os.makedirs(export_dir, exist_ok=True)
    base_path = os.path.join(export_dir, file_name)

    if file_type == "png":
        fig.savefig(f"{base_path}.png", dpi=300)
    elif file_type == "pgf":
        fig.savefig(f"{base_path}.pgf")
    elif file_type == "tex":
        tikz_code = matplot2tikz.get_tikz_code()
        legend_label = "Difference to high seq. pot. variant."
        occurrences = [pos for pos in range(len(tikz_code)) if tikz_code.startswith(legend_label, pos)]
        if len(occurrences) > 1:
            tikz_code = tikz_code.replace(f"\\addlegendentry{{{legend_label}}}", "", len(occurrences) - 1)
        with open(f"{base_path}.tex", "w", encoding="utf-8") as f:
            f.write(tikz_code)

    plt.close(fig)

def plot_technology_barplot_with_errorbars(
    data: pd.DataFrame,
    export_dir: str = "export/plots",
    file_name: str = "primary_energy_errorbars",
    file_type: str = "png",
    color_error: bool = False,
    error_bars: bool = True,
    fig_width=DEFAULT_FIGURE_WIDTH,
    fig_height=DEFAULT_FIGURE_HEIGHT,
    fontsize=DEFAULT_FONTSIZE,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
):
    """
    Grouped bar chart of energy (TWh) by technology (x-axis) for two scenarios
    ('default', 'carbon_costs'), with one-sided error bars that show the
    difference to the matching _710 variant (positive = up, negative = down).

    Parameters
    ----------
    data : DataFrame
        Needs columns ['Year', 'Folder', 'Data Name', 'Values'].
    export_dir : str
        Folder where the file is saved.
    file_name : str
        Base name of the output file.
    file_type : str
        'png', 'pdf', etc.
    color_error : bool
        If True, error bars are green (positive) or red (negative);
        otherwise they are black.
    """
    os.makedirs(export_dir, exist_ok=True)

    # --- prepare data -------------------------------------------------------
    df = data.query("Year == 2050").copy()
    df["Values"] /= 1e6  # → TWh

    scenarios = ["Default", "Carbon Stock Changes"]
    variants = [f"{s} 710" for s in scenarios]

    # wide table: rows = tech, cols = scenario / variant
    wide = df.pivot(index="Data Name", columns="Folder", values="Values")

    # check required columns
    missing = [c for c in scenarios + variants if c not in wide.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    techs = wide.index.tolist()
    n_tech = len(techs)

    # base heights and deltas (variant – base) as Series dictionaries
    base = {s: wide[s].fillna(0) for s in scenarios}
    delta = {s: wide[f"{s} 710"].fillna(0) - base[s] for s in scenarios}

    # --- plotting -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(n_tech)
    bar_w = 0.35
    colors = ["#1f77b4","#ff7f0e"]  # two scenarios

    for j, s in enumerate(scenarios):
        xpos = x + (j - 0.5) * bar_w
        heights = base[s].values
        ax.bar(xpos, heights, width=bar_w, color=colors[j],
               label=s.replace("_", " ").title(), zorder=3)

        # one-sided error bars
        errs = delta[s].values
        lowers = [abs(e) if e < 0 else 0 for e in errs]
        uppers = [e if e > 0 else 0 for e in errs]
        if error_bars:
            if color_error:
                ecols = ["green" if e > 0 else "red" if e < 0 else "black"
                        for e in errs]
            else:
                ecols = ["black"] * n_tech

            for i, (xi, y, lo, up, ec) in enumerate(zip(xpos, heights, lowers, uppers, ecols)):
                # Add label only for the very first errorbar (j=0, i=0)
                err_label = "Difference to high seq. pot. variant" if (j == 1 and i == 0) else None
                ax.errorbar(xi, y,
                            yerr=np.array([[lo], [up]]),
                            fmt="none", ecolor=ec,
                            capsize=4, elinewidth=1.8, zorder=4,
                            label=err_label)
                   
    # axes, legend, layout
    ax.set_xticks(x)
    ax.set_xticklabels(techs, rotation=45, ha="right")
    ax.set_ylabel("Energy (TWh)")
    ax.set_title("Primary Energy", fontsize=title_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.legend(title="Scenario", frameon=True)

    plt.tight_layout()

    # save
    path = os.path.join(export_dir, f"{file_name}.{file_type}")
    if path.endswith(".pgf"):
        configure_for_pgf()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Primary energy plot with error bars saved to {path}")

def plot_morris_mu_star(
    df: pd.DataFrame,
    file_name: str,
    unit: str,
    title: str = "Morris μ*",
    export_dir: str = "export/GSA",
    threshold: float = 0.0,
    fig_width=DEFAULT_FIGURE_WIDTH,
    fig_height=DEFAULT_FIGURE_HEIGHT,
    fontsize=DEFAULT_FONTSIZE,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
):
    plot_labels = {
        "seq_potential": "CO2 sequestration potential",
        "ef_wind": "CSCs from wind",
        "ef_solar": "CSCs from solar PV",
        "ef_solar-hsat": "CSCs from solar HSAT",
        "nuclear_costs": "Nuclear capital cost",
        "ef_manure": "CSCs from manure",
        "ef_crop_residues": "CSCs from crop residues",
        "ef_logging_residues": "CSCs from logging residues",
        "ef_stemwood": "CSCs from stemwood",
        "ef_grasses": "CSCs from grasses",
        "ef_sawdust": "CSCs from sawdust",
        "ef_secondary_forestry_residues": "CSCs from secondary forestry residues",
        "ef_woody_crops": "CSCs from woody crops",
        "ef_chips_and_pellets": "CSCs from chips & pellets",
        "ef_biomass_import": "CSCs from imported biomass",
        "biomass_costs": "Biomass feedstock costs",
    }
    # 1 · normalise column names ------------------------------------------------
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "parameter"})

    needed = ["parameter", "mu_star", "mu_star_conf"]
    if missing := [c for c in needed if c not in df.columns]:
        raise ValueError(f"DataFrame lacks required columns: {missing}")

    df = df[needed].copy()
    df[["mu_star", "mu_star_conf"]] = df[["mu_star", "mu_star_conf"]].astype(float)

    # filter by threshold
    df = df[df["mu_star"] >= threshold].sort_values("mu_star", ascending=False)
    if df.empty:
        raise ValueError("No rows exceed the given threshold; nothing to plot.")

    # 2 · asymmetric error bars (truncate at zero) ------------------------------
    mu_star  = df["mu_star"].to_numpy()
    half_ci  = df["mu_star_conf"].to_numpy()
    upper    = half_ci
    lower    = np.minimum(half_ci, mu_star)
    xerr     = np.vstack([lower, upper])          # shape (2, n)

    # 3 · horizontal bar plot ---------------------------------------------------
    y_pos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.barh(
        y_pos, mu_star,
        xerr=xerr, capsize=6,
        color="steelblue", linewidth=0, ecolor="black",
    )
    ax.invert_yaxis() 

    ax.set_yticks(y_pos)
    ax.set_yticklabels([plot_labels[param] for param in df["parameter"]])
    ax.set_xlabel(r"μ*({})".format(unit))
    ax.set_title(title, fontsize=title_fontsize)
    ax.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()

    # 4 · save ------------------------------------------------------------------
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, f"{file_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"μ* plot saved → {out_path}")

def main(custom_order=["Default", "Carbon Stock Changes"], file_type="png", export_dir="export/plots", data_folder="export",
         fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):

    capacity_factors = load_csv("capacity_factors.csv",folder_path=data_folder)

    create_gravitational_plot(
        "Costs vs. Emissions/CSCs",
        "gravitational_plot",
        export_dir=export_dir,
        file_type=file_type,
        capacity_factors=capacity_factors,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )

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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )

    data = load_csv("biomass_supply.csv",folder_path=data_folder)
    plot_biomass_use(data, "Biomass Use", "", "TWh", "biomass_supply", export_dir=export_dir,labels=False,
                     fig_width=fig_width, fig_height=fig_height, fontsize=fontsize, title_fontsize=title_fontsize)
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )

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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )

    data = load_csv("weighted_prices.csv",folder_path=data_folder)
    plot_feedstock_prices(
        data,
        "Weighted Feedstock Prices",
        "",
        "EUR/MWh",
        "weighted_feedstock_prices",
        export_dir=export_dir,
        file_type=file_type,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
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
    
    # usage_threshold = [
    #     {
    #         "biomass_type": "grasses",
    #         "emission_cost": 265  # €/tonCO2
    #     },
    #     {
    #         "biomass_type": "fuelwood residues",
    #         "emission_cost": 869  # €/tonCO2
    #     }
    # ]
    # create_gravitational_plot(
    #     "Gravitational Plot (Carbon Stock Changes)",
    #     "gravitational_plot_carbon_costs",
    #     export_dir=export_dir,
    #     file_type=file_type,
    #     capacity_factors=capacity_factors,
    #     biomass_supply=data,
    #     scenario="Carbon Stock Changes",
    #     usage_threshold=usage_threshold,
    # )

    # usage_threshold = [
    #     {
    #         "biomass_type": "C&P_RW",
    #         "emission_cost": 0  # €/tonCO2
    #     },
    #     {
    #         "biomass_type": "woody crops",   
    #         "emission_cost": 0  # €/tonCO2
    #     }
    # ]
    # create_gravitational_plot(
    #     "Gravitational Plot (Default)",
    #     "gravitational_plot_default",
    #     export_dir=export_dir,
    #     file_type=file_type,
    #     capacity_factors=capacity_factors,
    #     biomass_supply=data,
    #     scenario="Default",
    #     usage_threshold=usage_threshold,
    # )

def specific_plots(folder_path="export/main", export_path= "export/plots", file_type="png",
                   fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
    """
    Create specific plots for the project.
    """
    data = load_csv("biomass_supply.csv",folder_path=folder_path)
    capacity_factors = load_csv("capacity_factors.csv",folder_path=folder_path)
    create_gravitational_plot(
        "Cost vs Emissions/CSCs (Default)",
        "gravitational_plot_default",
        biomass_supply=data,
        scenario="Default",
        export_dir=export_path,
        file_type="png",
        capacity_factors=capacity_factors,
        variant_plot=True,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    create_gravitational_plot(
        "Cost vs Emissions/CSCs (Carbon Stock Changes)",
        "gravitational_plot_carbon_costs",
        biomass_supply=data,
        scenario="Carbon Stock Changes",
        export_dir=export_path,
        file_type="png",
        capacity_factors=capacity_factors,
        variant_plot=True,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    bm_data = load_csv("biomass_use_by_sector.csv",folder_path=folder_path)
    plot_stacked_biomass_with_errorbars(
        bm_data,
        export_dir=export_path,
        file_name="biomass_stacked_errorbar",
        file_type="png",
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    data = load_csv("primary_energy.csv",folder_path=folder_path)
    plot_technology_barplot_with_errorbars(
        data,
        export_dir=export_path,
        file_name="primary_energy_errorbars",
        file_type="png",
        color_error= False,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    plot_technology_barplot_with_errorbars(
        data,
        export_dir=export_path,
        file_name="primary_energy_no_errorbars",
        file_type="png",
        color_error= False,
        error_bars=False,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    # plot_stacked_biomass_with_errorbars(
    #     bm_data,
    #     export_dir=export_path,
    #     file_name="biomass_stacked_errorbar",
    #     file_type="tex",
    # )
    carbon_flow_diagram(save_path="export/plots/carbon_flow_diagram.png",
                         fig_width=fig_width, fig_height=fig_height, fontsize=fontsize, title_fontsize=title_fontsize)
    data = load_csv("supply_difference.csv",folder_path=folder_path)
    data = data[data["data_name"] != "total"]
    data_variants = load_csv("supply_difference_variants.csv",folder_path=folder_path)
    total_em_diff_variants = (
    data_variants.loc[data_variants["data_name"].str.strip().str.lower() == "total", "emission_difference"]
      .iloc[0].item()
    )
    plot_stacked_bar(
        data,
        "Avoided Carbon Stock Changes",
        "",
        "Mt_CO2",
        "emission_difference",
        multiplier=1e-6,
        column="emission_difference",
        columns="year",
        index="Data Name",
        threshold=1,
        threshold_column="emission_difference",
        export_dir=export_path,
        file_type=file_type,
        no_xticks=True,
        error_bar_amount= total_em_diff_variants,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    data = load_csv("cost_difference.csv",folder_path=folder_path)
    plot_costs(
        data,
        "Extra Costs Due to Carbon Stock Changes (larger 1 B€)",
        "",
        "Cost (Billion EUR)",
        "cost_difference",
        export_dir=export_path,
        file_type=file_type,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    data = load_csv("weighted_prices.csv",folder_path=folder_path)
    supply_data = load_csv("biomass_supply.csv",folder_path=folder_path)
    usage_dict_default = get_usage_dict(supply_data, "Default")
    usage_dict_carbon_costs = get_usage_dict(supply_data, "Carbon Stock Changes")
    
    # Create individual plots as before
    plot_costs_vs_prices(
        data,
        "Weighted Feedstock Prices vs. Costs",
        "Costs in Euro/MWh",
        "Prices in EUR/MWh",
        "prices_costs_default",
        scenario="Default",
        usage_dict=usage_dict_default,
        export_dir=export_path,
        file_type=file_type,
        add_legend=False,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    plot_costs_vs_prices(
        data,
        "Weighted Feedstock Prices vs. Costs",
        "Costs in Euro/MWh",
        "Prices in EUR/MWh",
        "prices_costs_carbon_costs",
        scenario="Carbon Stock Changes",
        usage_dict=usage_dict_carbon_costs,
        export_dir=export_path,
        file_type=file_type,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    
    # Create combined costs vs prices plot with shared legend
    plot_costs_vs_prices_combined(
        data,
        usage_dict_default=usage_dict_default,
        usage_dict_carbon_costs=usage_dict_carbon_costs,
        export_dir=export_path,
        file_type=file_type,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )


def mga_plots(include_fossils=False, fossil_breakdown=False, fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
    costs = load_csv("total_costs.csv",folder_path="export/mga",rename_scenarios=False)
    cost_diffs = make_cost_diffs(costs)

    # Load fossil fuel data if requested
    fossil_carbon_costs_710 = None
    fossil_carbon_costs = None
    fossil_default_710 = None
    fossil_default = None
    
    if include_fossils:
        try:
            # Load fossil fuel data (same files used for both breakdown and total modes)
            fossil_carbon_costs_710 = load_csv("fossil_fuel_supply_carbon_costs_710.csv", folder_path="export/mga", rename_scenarios=False)
            fossil_carbon_costs = load_csv("fossil_fuel_supply_carbon_costs.csv", folder_path="export/mga", rename_scenarios=False)
            fossil_default_710 = load_csv("fossil_fuel_supply_default_710.csv", folder_path="export/mga", rename_scenarios=False)
            fossil_default = load_csv("fossil_fuel_supply_default.csv", folder_path="export/mga", rename_scenarios=False)
        except Exception as e:
            print(f"Warning: Could not load fossil fuel data: {e}")
            include_fossils = False

    mga_data = load_csv("biomass_use_carbon_costs_710.csv",folder_path="export/mga",rename_scenarios=False)
    plot_mga(
        mga_data,
        "mga_carbon_costs_710",
        title="Near Optimal Biomass Use (Scenario Carbon Stock Changes 710)",
        export_dir="export/mga",
        file_type="png",
        unit="TWh",
        multiplier=1e-6,
        zero_lower_pct=cost_diffs["cscs_710"],
        include_fossil=include_fossils,
        fossil_df=fossil_carbon_costs_710,
        fossil_unit="TWh",
        fossil_multiplier=1e-6,
        fossil_breakdown=fossil_breakdown,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    mga_data = load_csv("biomass_use_carbon_costs.csv",folder_path="export/mga",rename_scenarios=False)
    plot_mga(
        mga_data,
        "mga_carbon_costs",
        title="Near Optimal Biomass Use (Scenario Carbon Stock Changes)",
        export_dir="export/mga",
        file_type="png",
        unit="TWh",
        multiplier=1e-6,
        zero_lower_pct=cost_diffs["cscs"],
        include_fossil=include_fossils,
        fossil_df=fossil_carbon_costs,
        fossil_unit="TWh",
        fossil_multiplier=1e-6,
        fossil_breakdown=fossil_breakdown,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    mga_data = load_csv("biomass_use_default_710.csv",folder_path="export/mga",rename_scenarios=False)
    plot_mga(
        mga_data,
        "mga_default_710",
        title="Near Optimal Biomass Use (Scenario Default 710)",
        export_dir="export/mga",
        file_type="png",
        unit="TWh",
        multiplier=1e-6,
        zero_lower_pct=cost_diffs["default_710"],
        include_fossil=include_fossils,
        fossil_df=fossil_default_710,
        fossil_unit="TWh",
        fossil_multiplier=1e-6,
        fossil_breakdown=fossil_breakdown,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    mga_data = load_csv("biomass_use_default.csv",folder_path="export/mga",rename_scenarios=False)
    plot_mga(
        mga_data,
        "mga_default",
        title="Near Optimal Biomass Use (Scenario Default)",
        export_dir="export/mga",
        file_type="png",
        unit="TWh",
        multiplier=1e-6,
        zero_lower_pct=cost_diffs["default"],
        include_fossil=include_fossils,
        fossil_df=fossil_default,
        fossil_unit="TWh",
        fossil_multiplier=1e-6,
        fossil_breakdown=fossil_breakdown,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )

def SA_plots(export_dir="export/GSA", data_folder="GSA/SA_results", fig_width=DEFAULT_FIGURE_WIDTH, fig_height=DEFAULT_FIGURE_HEIGHT, fontsize=DEFAULT_FONTSIZE, title_fontsize=DEFAULT_TITLE_FONTSIZE):
    """
    Create plots for the sensitivity analysis.
    """
    data = load_csv("biomass_use_SA_results.csv",folder_path=data_folder)
    plot_morris_mu_star(
        df=data,
        file_name="biomass_use_mu_star",
        unit="TWh",
        title="Morris μ* for Biomass Use",
        export_dir=export_dir,
        threshold=5,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    data = load_csv("wind_electricity_production_SA_results.csv",folder_path=data_folder)
    plot_morris_mu_star(
        df=data,
        file_name="wind_electricity_production_mu_star",
        unit="TWh",
        title="Morris μ* for Wind Electricity Production",
        export_dir=export_dir,
        threshold=5,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    data = load_csv("solar_electricity_production_SA_results.csv",folder_path=data_folder)
    plot_morris_mu_star(
        df=data,
        file_name="solar_electricity_production_mu_star",  
        unit="TWh",
        title="Morris μ* for Solar Electricity Production",
        export_dir=export_dir,
        threshold=5,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    data = load_csv("nuclear_electricity_production_SA_results.csv",folder_path=data_folder)
    plot_morris_mu_star(
        df=data,
        file_name="nuclear_electricity_production_mu_star",
        unit="TWh",
        title="Morris μ* for Nuclear Electricity Production",
        export_dir=export_dir,
        threshold=5,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    data = load_csv("oil_use_SA_results.csv",folder_path=data_folder)
    plot_morris_mu_star(
        df=data,
        file_name="oil_use_mu_star",
        unit="TWh",
        title="Morris μ* for Oil Use",
        export_dir=export_dir,
        threshold=5,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    data = load_csv("gas_use_SA_results.csv",folder_path=data_folder)
    plot_morris_mu_star(
        df=data,
        file_name="gas_use_mu_star",
        unit="TWh",
        title="Morris μ* for Gas Use",
        export_dir=export_dir,
        threshold=5,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    data = load_csv("system_costs_SA_results.csv",folder_path=data_folder)
    plot_morris_mu_star(
        df=data,
        file_name="system_costs_mu_star",
        unit="Billion EUR",
        title="Morris μ* for System Costs",
        export_dir=export_dir,
        threshold=1,
        fig_width=fig_width,
        fig_height=fig_height,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )

if __name__ == "__main__":

    file_type = "png"
    # file_type = "pgf"
    mpl.rcParams.update({
        "text.usetex": False,         # plain Matplotlib text engine
        "font.family":   "serif",
        "font.serif":    ["CMU Serif", "Latin Modern Roman",
                        "Computer Modern Roman", "Times"],  # fall-backs
        "mathtext.fontset": "cm",     # Computer Modern for $math$
        "figure.dpi":    300,
        "font.size": 12,
    })

    custom_order = ["Default", "Carbon Stock Changes", "Default 710", "Carbon Stock Changes 710"]  
    export_dir = "export/main_plots"
    data_folder = "export/main"
    
    # Configure plot dimensions and font sizes
    fig_width = 12  # Change this to adjust all plot widths
    fig_height = 8  # Change this to adjust all plot heights
    fontsize = 14   # Change this to adjust general font size
    title_fontsize = 18  # Change this to adjust title font size

    #specific_plots(fig_width=fig_width, fig_height=fig_height, fontsize=fontsize, title_fontsize=title_fontsize)
    #main(custom_order=custom_order, file_type=file_type, export_dir=export_dir, data_folder=data_folder, 
    #     fig_width=fig_width, fig_height=fig_height, fontsize=fontsize, title_fontsize=title_fontsize)
    # plot_efs(export_dir=export_dir)
    #plot_efs_for_presentation(export_dir=export_dir, file_type=file_type)

    mga_plots(include_fossils=True, fossil_breakdown=True, fig_width=fig_width, fig_height=fig_height, fontsize=fontsize, title_fontsize=title_fontsize)

    #SA_plots(fig_width=fig_width, fig_height=fig_height, fontsize=fontsize, title_fontsize=title_fontsize)

    ########### Sankey Diagrams ###########
    # co2_data = load_csv("co2_sankey.csv",folder_path="export/seq")
    # plot_co2_sankey(
    #     co2_data,
    #     scenario="Default",
    #     multiplier=1e-6,
    #     output_dir="export/plots",
    #     unit_label="Mt CO2",
    # )
    # plot_co2_sankey(
    #     co2_data,
    #     scenario="Carbon Stock Changes",
    #     multiplier=1e-6,
    #     output_dir="export/plots",
    #     unit_label="Mt CO2",
    # )




