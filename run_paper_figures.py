"""
Driver to regenerate the results/main-dependent paper figures only.
Mirrors the relevant parts of plotting.py __main__, skipping MGA / GSA(mu*) /
EF / Sankey blocks (those need result sets not present in results/main).

Writes all figures into a single folder: export/main_plots
"""
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "Computer Modern Roman", "Times"],
    "mathtext.fontset": "cm",
    "figure.dpi": 300,
    "font.size": 12,
})

import plotting as P

# carbon_flow_diagram is a static schematic (no results data) and forces
# text.usetex=True, which needs LaTeX font packages not present here and would
# break all subsequent data-driven plots. Skip it; the paper PNG is unchanged.
P.carbon_flow_diagram = lambda *a, **k: None

file_type = "png"
custom_order = ["Default", "Carbon Stock Changes", "Default 710", "Carbon Stock Changes 710"]
export_dir = "export/main_plots"
data_folder = "export/main"

fig_width = 10
fig_height = 6
fontsize = 14
title_fontsize = 18

# 1) Project-specific figures (gravitational_plot_default/_carbon_costs,
#    biomass_stacked, primary_energy, emission_difference, cost_difference,
#    prices_costs_combined). Redirect export_path to the single folder.
P.specific_plots(
    folder_path=data_folder,
    export_path=export_dir,
    file_type=file_type,
    fig_width=fig_width,
    fig_height=fig_height,
    fontsize=fontsize,
    title_fontsize=title_fontsize,
)

# 2) main() -> combined gravitational_plot + auxiliary plots.
P.main(
    custom_order=custom_order,
    file_type=file_type,
    export_dir=export_dir,
    data_folder=data_folder,
    fig_width=fig_width,
    fig_height=fig_height,
    fontsize=fontsize,
    title_fontsize=title_fontsize,
)

# 3) Clean emission-factor bar chart (Methods input figure). Depends only on the
#    config emission factors, so it runs without result sets.
P.plot_efs_clean(export_dir=export_dir, file_type=file_type)

print("DONE")
