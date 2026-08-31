"""Regenerate the use-vs-sequestration-potential figures in export/seq.
Reconstructs the data the (un-committed) original driver fed to
plotting.plot_biomass_use_vs_sequestration_potential /
plot_fossil_gas_use_vs_sequestration_potential, reading energy_balance.csv
directly (same source as make_seq_price_figs.py). pypsa/matplot2tikz stubbed."""
import os, sys, types
ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(ROOT); sys.path.insert(0, ROOT)
for m in ("pypsa", "matplot2tikz"):
    if m not in sys.modules:
        try: __import__(m)
        except ImportError: sys.modules[m] = types.ModuleType(m)
import matplotlib as mpl; mpl.use("Agg")
import pandas as pd
import plotting as P

BIO_BUSES = {"solid biomass", "biogas"}

# label -> result path. Label must satisfy the plotting regex (cscs_<n> / cscs / cscs_710).
CSCS = {
    "cscs_150": "results/seq_pot/cscs_150", "cscs_175": "results/seq_pot/cscs_175",
    "cscs_200": "results/seq_pot/cscs_200", "cscs_225": "results/seq_pot/cscs_225",
    "cscs":     "results/main/cscs",        "cscs_710": "results/main/cscs_710",
}
DEFAULT = {
    "cscs_150": "results/seq_pot/default_150", "cscs_175": "results/seq_pot/default_175",
    "cscs_200": "results/seq_pot/default_200", "cscs_225": "results/seq_pot/default_225",
    "cscs":     "results/seq_pot/default_250", "cscs_710": "results/main/default_710",
}

def read_eb(path):
    df = pd.read_csv(os.path.join(path, "csvs", "energy_balance.csv"),
                     skiprows=[0, 1], header=0)
    df.columns = ["component", "carrier", "bus_carrier", "value"]
    return df

def total_biomass_use_mwh(df):
    m = df[(df.component == "Link") & (df.bus_carrier.isin(BIO_BUSES)) & (df.value > 0)]
    return float(m["value"].sum())

def fossil_gas_use_mwh(df):
    m = df[(df.carrier == "gas") & (df.bus_carrier == "gas") & (df.value > 0)]
    return float(m["value"].sum())

def build_df(scenarios):
    rows = []
    for label, path in scenarios.items():
        df = read_eb(path)
        rows.append({"Folder": label, "Data Name": "biomass", "Values": total_biomass_use_mwh(df)})
        rows.append({"Folder": label, "Data Name": "gas",     "Values": fossil_gas_use_mwh(df)})
    return pd.DataFrame(rows)

for variant, scen in (("cscs", CSCS), ("default", DEFAULT)):
    d = build_df(scen)
    bio = d[d["Data Name"]=="biomass"].assign(TWh=lambda x: x["Values"]*1e-6)
    gas = d[d["Data Name"]=="gas"].assign(TWh=lambda x: x["Values"]*1e-6)
    print(f"\n### {variant} ###")
    print("biomass use [TWh]:"); print(bio[["Folder","TWh"]].to_string(index=False))
    print("fossil gas use [TWh]:"); print(gas[["Folder","TWh"]].round(3).to_string(index=False))
    P.plot_biomass_use_vs_sequestration_potential(
        df=d, file_name=f"biomass_use_vs_sequestration_potential_{variant}",
        title=f"Total Biomass Use vs Sequestration Potential ({'CSCs' if variant=='cscs' else 'Default'})",
        export_dir="export/seq", file_type="png")
    P.plot_fossil_gas_use_vs_sequestration_potential(
        df=d, file_name=f"fossil_gas_use_vs_sequestration_potential_{variant}",
        title=f"Fossil Gas Use vs Sequestration Potential ({'CSCs' if variant=='cscs' else 'Default'})",
        export_dir="export/seq", file_type="png")
print("\nSEQ USE FIGS DONE")
