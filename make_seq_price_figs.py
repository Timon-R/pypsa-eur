"""Regenerate the CSCs biomass-price vs sequestration-potential figures.

Reads feedstock weighted prices and feedstock use directly from the solved-run
summary CSVs (results/seq_pot for 150-225, results/main for the 250 baseline and
710), so it does not depend on any export step or on the .nc networks.

Produces two figures in export/seq:
  - biomass_prices_vs_sequestration_potential_cscs_selected      (lines + markers;
    feedstock points with negligible use are omitted via the usage threshold)
  - biomass_prices_vs_sequestration_potential_cscs_selected_sized (marker area
    proportional to absolute feedstock use, TWh)
"""
import os
import sys
import types

import pandas as pd

# plotting.py imports pypsa (via result_analysis) and matplot2tikz at module
# load; this figure routine uses neither, so stub them if absent to avoid
# pulling in the full solver stack.
for _name in ("pypsa", "matplot2tikz"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = types.ModuleType(_name)

import plotting as P  # noqa: E402

# seq potential [MtCO2/yr] -> results folder. 250 and 710 come from results/main.
SCENARIOS = {
    150: ("results/seq_pot/cscs_150", "cscs_150"),
    175: ("results/seq_pot/cscs_175", "cscs_175"),
    200: ("results/seq_pot/cscs_200", "cscs_200"),
    225: ("results/seq_pot/cscs_225", "cscs_225"),
    250: ("results/main/cscs", "cscs"),
    710: ("results/main/cscs_710", "cscs_710"),
}
FEEDSTOCKS = [
    "residues from landscape care",
    "fuelwood residues",
    "manure",
    "agricultural waste",
]
BIOMASS_BUSES = {"solid biomass", "biogas"}


def read_prices(path):
    df = pd.read_csv(os.path.join(path, "csvs", "weighted_prices.csv"),
                     header=None, names=["name", "value"])
    s = df.set_index("name")["value"]
    return {fe: float(s[fe]) for fe in FEEDSTOCKS if fe in s.index}


def read_use_mwh(path):
    df = pd.read_csv(os.path.join(path, "csvs", "energy_balance.csv"),
                     skiprows=[0, 1], header=0)
    df.columns = ["component", "carrier", "bus_carrier", "value"]
    out = {}
    for fe in FEEDSTOCKS:
        m = df[(df.component == "Link") & (df.carrier == fe)
               & (df.bus_carrier.isin(BIOMASS_BUSES)) & (df.value > 0)]
        out[fe] = float(m["value"].sum())  # MWh supplied to the biomass buses
    return out


def main():
    price_rows, use_rows = [], []
    for seq_pot, (path, folder) in SCENARIOS.items():
        prices = read_prices(path)
        uses = read_use_mwh(path)
        for fe in FEEDSTOCKS:
            if fe in prices:
                price_rows.append({"Folder": folder, "Data Name": fe,
                                   "Values": prices[fe]})
            use_rows.append({"Folder": folder, "Data Name": fe,
                             "Values": uses[fe]})

    price_df = pd.DataFrame(price_rows)
    use_df = pd.DataFrame(use_rows)

    # Feedstock potential = the largest annual draw observed across all runs
    # (capacity-bound feedstocks sit at their potential in the unconstrained
    # default runs); expressed in TWh, replicated per scenario.
    use_twh = use_df.assign(twh=use_df["Values"] * 1e-6)
    potential_twh = use_twh.groupby("Data Name")["twh"].max()
    pot_rows = [{"scenario": folder, "carrier": fe,
                 "weight_TWh": float(potential_twh[fe])}
                for folder in (f for _, f in SCENARIOS.values()) for fe in FEEDSTOCKS]
    potentials_df = pd.DataFrame(pot_rows)

    common = dict(
        df=price_df,
        biomass_use_df=use_df,
        biomass_potentials_df=potentials_df,
        export_dir="export/seq",
        file_type="png",
    )

    # 1) Standard figure (negligible-use points omitted by usage threshold).
    P.plot_selected_biomass_prices_vs_sequestration_potential(
        file_name="biomass_prices_vs_sequestration_potential_cscs_selected",
        **common,
    )
    # 2) Marker size proportional to absolute feedstock use.
    P.plot_selected_biomass_prices_vs_sequestration_potential(
        size_by_use=True,
        size_metric="use_twh",
        title="Biomass Marginal Prices vs Sequestration Potential (CSCs)\n(marker size = feedstock use)",
        file_name="biomass_prices_vs_sequestration_potential_cscs_selected_sized",
        **common,
    )

    # Diagnostic table.
    tbl = use_twh.pivot(index="Folder", columns="Data Name", values="twh")
    order = [f for _, f in SCENARIOS.values()]
    print("\nFeedstock use [TWh] by scenario:")
    print(tbl.reindex(order)[FEEDSTOCKS].round(1).to_string())
    print("\nFeedstock potential [TWh]:")
    print(potential_twh[FEEDSTOCKS].round(1).to_string())


if __name__ == "__main__":
    main()
