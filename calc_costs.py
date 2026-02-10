

"""
calc_costs.py — investment cost calculation for technologies with carbon capture (CC)

Implements the cost formulation described in the Nature Energy paper's Methods
section (equations (8)–(11)) for post-combustion capture on biomass-related
processes. Parameters (boiler efficiencies, ε_s, etc.) should be provided per
technology based on the paper's Supplementary Information. Investment cost
numbers are read from `resources/costs_2040.csv`.

Key equations (variable names in parentheses):
  α = 1 / (1 + ε_s * e_th_cc / η_th)
  η_new = α * η_old
  η_steam = (1 - α) * η_th
  C_I,new per MW(main) = C_I,old * (η_old/η_new) + C_I,th * (η_steam/η_new) + C_I,cc * ε_s
See paper for definitions. Units this module expects are documented below.

Author: (you)

If executed **without arguments**, an interactive wizard will prompt for the
required inputs (CSV path, base technology, η_old, ε_s, boiler type, etc.), so
you can simply press ▶️ in VS Code and provide values when asked.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


def _suggest_techs(df: pd.DataFrame, needle: str, limit: int = 15) -> list[str]:
    techs = df["technology"].unique().tolist()
    n = needle.casefold()
    hits = [t for t in techs if n in t.casefold()]
    return hits[:limit]


def save_results_to_csv(result: 'CCReturn', inputs: 'CCInputs', filename: str) -> None:
    """Save the CC calculation results to a CSV file.
    
    Parameters
    ----------
    result : CCReturn
        The calculation results
    inputs : CCInputs
        The input parameters used for calculation
    filename : str
        Path to save the CSV file
    """
    # Create a comprehensive results dictionary
    data = {
        'Parameter': [
            'Base Technology',
            'η_old (base efficiency)',
            'ε_s (CO2 intensity) [tCO2/MWh_out]',
            'Boiler Type',
            'e_th_cc (heat for capture) [MWh_th/tCO2]',
            'η_th override',
            'Capture Unit Technology',
            '',  # Empty row
            'α (alpha)',
            'η_new (new efficiency)',
            'η_steam (steam efficiency)',
            '',  # Empty row
            'C_I,new [EUR/MW_out]',
            'Base Process Scaled [EUR/MW_out]',
            'Boiler Scaled [EUR/MW_out]',
            'CC Unit [EUR/MW_out]'
        ],
        'Value': [
            inputs.base_tech,
            inputs.eta_old,
            inputs.eps_s,
            inputs.boiler_type,
            inputs.e_th_cc,
            inputs.eta_th_override or 'Default',
            inputs.capture_unit_tech or 'Auto',
            '',  # Empty row
            f'{result.alpha:.6f}',
            f'{result.eta_new:.6f}',
            f'{result.eta_steam:.6f}',
            '',  # Empty row
            f'{result.ci_new_eur_per_mw:,.2f}',
            f'{result.breakdown["base_scaled"]:,.2f}',
            f'{result.breakdown["boiler_scaled"]:,.2f}',
            f'{result.breakdown["cc_unit"]:,.2f}'
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def interactive_main() -> None:
    print("\n— CC investment cost wizard —")
    # Defaults that work from repo root
    default_csv = "resources/costs_2040.csv"
    csv_path = input(f"Path to costs CSV [{default_csv}]: ").strip() or default_csv

    if not os.path.exists(csv_path):
        print(f"CSV not found at '{csv_path}'. Please check the path.")
        return

    costs_df = load_investment_costs(csv_path)

    # Choose base technology
    while True:
        base_tech = input("Base technology name (as in CSV): ").strip()
        try:
            _get_row(costs_df, base_tech)
            break
        except KeyError:
            sug = _suggest_techs(costs_df, base_tech)
            if sug:
                print("Not found. Did you mean one of:")
                for s in sug:
                    print("  -", s)
            else:
                print("Not found. Try a different substring or check the CSV.")

    # Efficiencies and emissions intensity
    def _ask_float(prompt: str, default: float | None = None) -> float:
        while True:
            s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
            if not s and default is not None:
                return float(default)
            try:
                return float(s)
            except ValueError:
                print("Please enter a number.")

    eta_old = _ask_float("η_old (efficiency to main product, per unit)")

    eps_s_raw = _ask_float("ε_s (CO2 intensity)")
    basis = input("Is ε_s given per 'MWh_out' or 'MWh_in'? [MWh_out]: ").strip().lower() or "mwh_out"
    if basis.startswith("mwh_in") or basis.startswith("mwh_th"):
        if eta_old <= 0:
            print("η_old must be > 0 to convert from per MWh_in. Aborting.")
            return
        eps_s = eps_s_raw / eta_old
        print(f"Converted ε_s to per MWh_out using η_old: {eps_s:.6g} tCO2/MWh_out")
    else:
        eps_s = eps_s_raw

    boiler = (input("Boiler for capture steam ['biomass' or 'gas'] [biomass]: ").strip() or "biomass").lower()

    e_th_cc = _ask_float("Heat for solvent regeneration e_th_cc [MWh_th/tCO2]", default=E_TH_CC)

    eta_th_in = input("Override η_th boiler efficiency? Leave blank to use defaults (0.89 biomass / 0.93 gas): ").strip()
    eta_th_override = float(eta_th_in) if eta_th_in else None

    cc_label = input("CSV label for capture train CAPEX (€/tCO2/h) [leave blank to auto]: ").strip() or None

    inputs = CCInputs(
        base_tech=base_tech,
        eta_old=eta_old,
        eps_s=eps_s,
        boiler_type=boiler,
        e_th_cc=e_th_cc,
        eta_th_override=eta_th_override,
        capture_unit_tech=cc_label,
    )

    try:
        out = compute_cc_investment(costs_df, inputs)
    except Exception as e:
        print("Calculation failed:", e)
        return

    print("\n— Result —")
    print(f"C_I,new [EUR/MW_out] = {out.ci_new_eur_per_mw:,.2f}")
    print("Breakdown:")
    for k, v in out.breakdown.items():
        print(f"  {k:>13s}: {v:,.2f}")
    print(f"alpha    = {out.alpha:.6f}")
    print(f"eta_new  = {out.eta_new:.6f}")
    print(f"eta_steam= {out.eta_steam:.6f}")
    
    # Ask user if they want to save results to CSV
    save_csv = input("\nSave results to CSV file? [y/N]: ").strip().lower()
    if save_csv.startswith('y'):
        default_filename = f"cc_results_{base_tech.replace(' ', '_').lower()}.csv"
        filename = input(f"CSV filename [{default_filename}]: ").strip() or default_filename
        
        try:
            save_results_to_csv(out, inputs, filename)
            print(f"Results saved to: {filename}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")


# ---- Constants from the paper's Methods ----
 # Heat required for solvent regeneration for post-combustion capture (MWh_th / tCO2)
E_TH_CC = 0.66  # MWh_th per tCO2 (at ~100 °C)

# Default capture-train investment (€/ (tCO2/h)) from Supplementary Table S3
DEFAULT_CAPTURE_COST = 2_400_000.0

# Technology label used in costs_2040.csv for generic capture train CAPEX (€/ (tCO2/h)).
# The supplementary uses a constant 2.4 M€/ (tCO2/h) for 2040 — the CSV should carry this.
CC_UNIT_TECH = "CO2 liquefaction"  # fallback if specific CC line is missing
CC_CHPSURROGATE_TECH = "CO2 capture in CHP"

# Boiler technology labels in costs_2040.csv (€/kW_th)
BOILER_TECH_SOLID = "Solid biomass boiler steam"
BOILER_TECH_GAS = "Gas boiler steam"


@dataclass
class CCInputs:
    """Inputs required to compute CC-adjusted investment cost for a base process.

    Attributes
    ----------
    base_tech : str
        Technology name in costs_2040.csv representing the base (without CC) plant.
        Examples: "Biomass CHP", "Gas boiler steam", "Direct firing biomass",
        "Biofuel", etc.
    eta_old : float
        Base-process efficiency to main product (η_old, per unit of energy in).
        Must be taken from the Supplementary (Table S4 / Extended Data Table 1).
    eps_s : float
        CO2 in the process output stream per unit of main product (ε_s).
        Units: tCO2 / MWh_out. See Supplementary Tables S1–S2.
    boiler_type : str
        Either "biomass" or "gas". Per Methods: use a gas steam boiler for biogas
        processes; otherwise a solid-biomass steam boiler. This choice sets η_th and
        selects the correct boiler CAPEX line from the CSV.
    e_th_cc : float
        Heat demand for CO2 solvent regeneration (MWh_th / tCO2). Default 0.66.
    eta_th_override : Optional[float]
        If provided, overrides the default boiler efficiency (η_th) implied by
        boiler_type. Otherwise uses 0.89 for solid biomass and 0.93 for gas per S2.
    capture_unit_tech : Optional[str]
        Technology label in CSV to take the capture train CAPEX (€/ (tCO2/h)).
        Defaults to using the CHP capture line if present, else falls back to
        CO2 liquefaction (rare) which you should override in practice.
    capture_unit_cost_eur_per_tco2h : Optional[float]
        If provided, use this numeric CAPEX (€/ (tCO2/h)) for the capture train
        instead of reading from the CSV.
    """

    base_tech: str
    eta_old: float
    eps_s: float
    boiler_type: str = "biomass"
    e_th_cc: float = E_TH_CC
    eta_th_override: Optional[float] = None
    capture_unit_tech: Optional[str] = None
    capture_unit_cost_eur_per_tco2h: Optional[float] = None


@dataclass
class CCReturn:
    """Result and breakdown for CC-adjusted investment cost per MW of main output."""
    ci_new_eur_per_mw: float
    breakdown: Dict[str, float]
    eta_new: float
    eta_steam: float
    alpha: float


# ---- CSV loading and unit handling -------------------------------------------------

UNIT_PER_KW_PATTERN = re.compile(r"EUR/(kW(\w*)?)")
UNIT_PER_MW_PATTERN = re.compile(r"EUR/(MW(\w*)?)")
UNIT_PER_TCO2H_PATTERN = re.compile(r"EUR/\(tCO2/h\)")


def load_investment_costs(csv_path: str) -> pd.DataFrame:
    """Load the costs CSV and filter to investment entries.

    Returns a DataFrame with at least: technology, unit, value.
    """
    df = pd.read_csv(csv_path)
    # Keep only investment parameter rows
    df = df[df["parameter"].str.lower() == "investment"].copy()
    # Normalise whitespace in technology labels
    df["technology"] = df["technology"].str.strip()
    df["unit"] = df["unit"].str.strip()
    return df


def _get_row(df: pd.DataFrame, tech: str) -> Tuple[float, str]:
    rows = df[df["technology"].str.casefold() == tech.casefold()]
    if rows.empty:
        raise KeyError(f"Technology not found in CSV: '{tech}'")
    # If multiple, take the first occurrence
    r = rows.iloc[0]
    return float(r["value"]), str(r["unit"])  # type: ignore


def value_to_eur_per_mw(value: float, unit: str) -> float:
    """Convert CSV investment value to EUR per MW of *main-product* capacity.

    Accepts any of the following unit shapes in the CSV:
    - EUR/kW, EUR/kWel, EUR/kW_th, EUR/kWCH4, EUR/kWFT, ... → multiply by 1000
    - EUR/MW, EUR/MWel, EUR/MW_th, ... → keep as-is
    For capture trains (EUR/tCO2/h), use value_to_eur_per_tco2h().
    """
    if UNIT_PER_KW_PATTERN.fullmatch(unit):
        return value * 1000.0
    if UNIT_PER_MW_PATTERN.fullmatch(unit):
        return value
    raise ValueError(
        f"Unsupported unit for per-MW conversion: '{unit}'. Expected EUR/kW* or EUR/MW*."
    )



# ---- Unit canonicaliser for CC units ----
def _canon_unit(u: str) -> str:
    """Canonicalise a unit string to tolerate variants like 'EUR/t_CO2/h' vs 'EUR/(tCO2/h)'."""
    s = str(u)
    s = s.replace('€', 'EUR')
    s = s.replace('₂', '2')  # Unicode subscript 2 → '2'
    s = re.sub(r"\s+", "", s)  # remove spaces
    s = s.replace('_', '').replace('-', '')  # drop underscores and hyphens
    s = s.replace('(', '').replace(')', '')  # drop parentheses
    return s

def value_to_eur_per_tco2h(value: float, unit: str) -> float:
    canon = _canon_unit(unit).upper()
    if canon == 'EUR/TCO2/H':
        return value
    raise ValueError(
        f"Unsupported CC unit: '{unit}' (normalised: '{canon}'). Expected something like 'EUR/(tCO2/h)' or 'EUR/t_CO2/h'."
    )


# ---- Core formulas from the Methods ------------------------------------------------

def compute_alpha(eps_s: float, eta_th: float, e_th_cc: float = E_TH_CC) -> float:
    """Equation (8): α = 1 / (1 + ε_s * e_th_cc / η_th).

    Parameters
    ----------
    eps_s : float
        CO2 in the exhaust per MWh_out (output energy) [tCO2/MWh_out].
    eta_th : float
        Boiler efficiency to steam at ~100 °C (per unit). Typical: 0.89 (solid biomass), 0.93 (gas).
    e_th_cc : float
        Heat demand for solvent regeneration [MWh_th / tCO2]. Default 0.66.
    """
    denom = 1.0 + (eps_s * e_th_cc / eta_th)
    return 1.0 / denom


def compute_eta_new(alpha: float, eta_old: float) -> float:
    """Equation (10): η_new = α * η_old."""
    return alpha * eta_old


def compute_eta_steam(alpha: float, eta_th: float) -> float:
    """Equation (9): η_steam = (1 − α) * η_th."""
    return (1.0 - alpha) * eta_th


# ---- Top-level calculation ---------------------------------------------------------

def compute_cc_investment(
    costs_df: pd.DataFrame,
    inputs: CCInputs,
) -> CCReturn:
    """Compute CC-adjusted CAPEX per MW_out and provide a breakdown.

    The result implements Eq. (11):
      C_I,new = C_I,old * (η_old/η_new) + C_I,th * (η_steam/η_new) + C_I,cc * ε_s

    Assumptions
    -----------
    * Boiler type → η_th defaults: 0.89 (solid biomass), 0.93 (gas), unless overridden.
    * Capture train CAPEX is read from `CO2 capture in CHP` entry (€/ (tCO2/h)).
      Override via `inputs.capture_unit_tech` if you prefer a different line.
    * All per-MW values refer to the main product (electricity, heat, CH4, FT fuel, ...).

    Returns
    -------
    CCReturn
        Includes total cost per MW_out and a dict with the three Eq. (11) terms.
    """
    # Base-process CAPEX (per MW of main product)
    val, unit = _get_row(costs_df, inputs.base_tech)
    ci_old = value_to_eur_per_mw(val, unit)

    # Boiler CAPEX (per MW_th)
    boiler_tech = BOILER_TECH_GAS if inputs.boiler_type.lower().startswith("gas") else BOILER_TECH_SOLID
    b_val, b_unit = _get_row(costs_df, boiler_tech)
    ci_th = value_to_eur_per_mw(b_val, b_unit)  # treat per MW_th as per MW capacity

    # Capture train CAPEX (€/ (tCO2/h))
    if inputs.capture_unit_cost_eur_per_tco2h is not None:
        ci_cc = inputs.capture_unit_cost_eur_per_tco2h
    else:
        cc_label = inputs.capture_unit_tech or CC_CHPSURROGATE_TECH
        try:
            cc_val, cc_unit = _get_row(costs_df, cc_label)
            ci_cc = value_to_eur_per_tco2h(cc_val, cc_unit)
        except KeyError:
            # Prefer the Supplementary default rather than falling back to CO2 liquefaction
            ci_cc = DEFAULT_CAPTURE_COST

    # Boiler efficiency η_th
    if inputs.eta_th_override is not None:
        eta_th = inputs.eta_th_override
    else:
        eta_th = 0.93 if boiler_tech == BOILER_TECH_GAS else 0.89

    # Eq. (8)–(10)
    alpha = compute_alpha(inputs.eps_s, eta_th, inputs.e_th_cc)
    eta_new = compute_eta_new(alpha, inputs.eta_old)
    eta_steam = compute_eta_steam(alpha, eta_th)

    if eta_new <= 0:
        raise ValueError("Computed η_new ≤ 0. Check ε_s, η_th, and e_th_cc inputs.")

    # Eq. (11) — all terms are per MW_out
    term_base = ci_old * (inputs.eta_old / eta_new)
    term_boiler = ci_th * (eta_steam / eta_new)
    term_cc = ci_cc * inputs.eps_s  # note: scales with the process CO2 flow ε_s

    ci_new = term_base + term_boiler + term_cc

    return CCReturn(
        ci_new_eur_per_mw=ci_new,
        breakdown={
            "base_scaled": term_base,
            "boiler_scaled": term_boiler,
            "cc_unit": term_cc,
        },
        eta_new=eta_new,
        eta_steam=eta_steam,
        alpha=alpha,
    )


# ---- Presets runner ---------------------------------------------------------------

from typing import List

def run_presets() -> None:
    """Run a predefined list of CC calculations and print nicely formatted tables.

    Uses units consistent with the Supplementary: ε_s in tCO2/MWh_out. Boiler
    efficiencies default to 0.89 (solid biomass) and 0.93 (gas). Capture CAPEX
    is taken as 2.4 M€/ (tCO2/h) unless overridden.
    """
    csv_path = "resources/costs_2040.csv"
    df = load_investment_costs(csv_path)

    # Helper to compute ε_s from a per-input figure
    def per_out_from_per_in(eps_s_in: float, eta_old: float) -> float:
        if eta_old <= 0:
            raise ValueError("eta_old must be > 0 to convert ε_s from per MWh_in to per MWh_out")
        return eps_s_in / eta_old

    PRESETS: List[dict] = [
        {
            "label": "Gas CHP with CC",
            "base_tech": "central gas CHP",
            "eta_old": 0.42,
            "eps_s": 0.198,  # tCO2/MWh_out (natural gas)
            "boiler": "gas",
        },
        {
            "label": "Biomass CHP with CC",
            "base_tech": "Biomass CHP",
            "eta_old": 0.27,
            "eps_s": 0.37,   # tCO2/MWh_out (solid biomass)
            "boiler": "biomass",
        },
        {
            "label": "Steam Methane Reforming with CC",
            "base_tech": "Steam Methane Reforming",
            "eta_old": 0.76,  # base SMR -> H2
            "eps_s": 0.143,   # back-calibrated to yield eta_new ≈ 0.69
            "boiler": "gas",
        },
        {
            "label": "BtL (biofuel) with CC",
            "base_tech": "BtL",
            "eta_old": 0.4167,
            # ε_s from SI Table S1: 0.2458 tCO2/MWh_in → convert to per MWh_out
            "eps_s": per_out_from_per_in(0.2458, 0.4167),
            "boiler": "biomass",
        },
        {
            "label": "Direct firing gas with CC",
            "base_tech": "Direct firing gas",
            "eta_old": 1.0,
            "eps_s": 0.198,
            "boiler": "gas",
        },
        {
            "label": "Direct firing biomass with CC",
            "base_tech": "Direct firing solid fuels",
            "eta_old": 1.0,
            "eps_s": 0.37,
            "boiler": "biomass",
        },
        {
            "label": "Solid biomass boiler steam CC",
            "base_tech": "Solid biomass boiler steam",
            "eta_old": 0.89,
            "eps_s": 0.37,
            "boiler": "biomass",
        },
        {
            "label": "Biogas with CC",
            "base_tech": "Biogas",
            "eta_old": 1.0,
            "eps_s": 0.198,
            "boiler": "gas",
        },
        {
            "label": "Waste incineration with CC",
            "base_tech": "waste CHP",
            "eta_old": 0.21,
            # Proxy ε_s as solid-biomass; replace if you have MSW-specific figure
            "eps_s": 0.37,
            "boiler": "biomass",
        },
    ]

    rows_results = []
    rows_inputs = []

    for p in PRESETS:
        out = compute_cc_investment(
            df,
            CCInputs(
                base_tech=p["base_tech"],
                eta_old=p["eta_old"],
                eps_s=p["eps_s"],
                boiler_type=p["boiler"],
                e_th_cc=E_TH_CC,
                eta_th_override=None,
                capture_unit_tech=None,
                capture_unit_cost_eur_per_tco2h=DEFAULT_CAPTURE_COST,
            ),
        )

        # Determine η_th used
        eta_th_used = 0.93 if p["boiler"].lower().startswith("gas") else 0.89

        rows_results.append({
            "Technology": p["label"],
            "η_new": out.eta_new,
            "CAPEX [EUR/kW_out]": out.ci_new_eur_per_mw / 1000.0,
        })

        rows_inputs.append({
            "Technology": p["label"],
            "Base tech": p["base_tech"],
            "η_old": p["eta_old"],
            "ε_s [tCO2/MWh_out]": p["eps_s"],
            "Boiler": p["boiler"],
            "η_th": eta_th_used,
            "e_th,cc [MWh_th/tCO2]": E_TH_CC,
            "C_I,cc [EUR/(tCO2/h)]": DEFAULT_CAPTURE_COST,
            "alpha": out.alpha,
        })

    # Print nicely
    def fmt_row(r):
        return f"{r['Technology']:<32s}  η_new={r['η_new']:.3f}   CAPEX={r['CAPEX [EUR/kW_out]']:,.0f} EUR/kW_out"

    print("\n— Carbon capture presets (2040) —")
    for r in rows_results:
        print(fmt_row(r))

    print("\nCO2 capture equipment (CHP): 2,400,000 EUR/(tCO2/h)")

    # Ask user if they want to save CSV files
    save_csv = input("\nSave results to CSV files? [y/N]: ").strip().lower()
    if save_csv.startswith('y'):
        pd.DataFrame(rows_results).to_csv("cc_presets_results.csv", index=False)
        pd.DataFrame(rows_inputs).to_csv("cc_presets_inputs.csv", index=False)
        print("Results saved to: cc_presets_results.csv and cc_presets_inputs.csv")

    # Also print a compact table of inputs
    print("\n— Inputs used —")
    hdr = "Technology                          η_old   ε_s[tCO2/MWh_out]  Boiler   η_th   alpha"
    print(hdr)
    for r in rows_inputs:
        print(f"{r['Technology']:<32s}  {r['η_old']:.3f}   {r['ε_s [tCO2/MWh_out]']:.3f}           {r['Boiler']:<7s} {r['η_th']:.2f}  {r['alpha']:.3f}")


# ---- Convenience helpers -----------------------------------------------------------

def compute_cc_for_named_process(
    csv_path: str,
    base_tech: str,
    eta_old: float,
    eps_s: float,
    boiler_type: str = "biomass",
    e_th_cc: float = E_TH_CC,
    eta_th_override: Optional[float] = None,
    capture_unit_tech: Optional[str] = None,
) -> CCReturn:
    """One-shot convenience function loading the CSV for you.

    Example
    -------
    >>> res = compute_cc_for_named_process(
    ...     csv_path="resources/costs_2040.csv",
    ...     base_tech="Biomass CHP",
    ...     eta_old=0.27,           # from Supplementary Table S4
    ...     eps_s=0.37,             # tCO2/MWh_main (example; ensure correct basis)
    ...     boiler_type="biomass",
    ... )
    >>> res.ci_new_eur_per_mw
    1234567.8
    >>> res.breakdown
    {"base_scaled": ..., "boiler_scaled": ..., "cc_unit": ...}
    """
    costs_df = load_investment_costs(csv_path)
    inputs = CCInputs(
        base_tech=base_tech,
        eta_old=eta_old,
        eps_s=eps_s,
        boiler_type=boiler_type,
        e_th_cc=e_th_cc,
        eta_th_override=eta_th_override,
        capture_unit_tech=capture_unit_tech,
    )
    return compute_cc_investment(costs_df, inputs)


if __name__ == "__main__":
    # If run without arguments (e.g. VS Code ▶️), ask user what they want to do
    if len(sys.argv) == 1:
        print("\n— CC investment cost calculator —")
        run_presets_choice = input("Run predefined presets? [Y/n]: ").strip().lower()
        
        if run_presets_choice in ['', 'y', 'yes']:
            run_presets()
        else:
            interactive_main()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Compute CC-adjusted CAPEX per MW of main output.")
    parser.add_argument("--presets", action="store_true", help="Run predefined preset calculations")
    parser.add_argument("--csv", default="resources/costs_2040.csv", help="Path to costs_2040.csv")
    parser.add_argument("--tech", help="Base technology name in CSV (without CC)")
    parser.add_argument("--eta_old", type=float, help="η_old (base efficiency to main product)")
    parser.add_argument("--eps_s", type=float, help="ε_s [tCO2/MWh_out]")
    parser.add_argument("--boiler", choices=["biomass", "gas"], default="biomass", help="Boiler fuel for CC steam")
    parser.add_argument("--e_th_cc", type=float, default=E_TH_CC, help="Heat demand for capture [MWh_th/tCO2]")
    parser.add_argument("--eta_th", type=float, default=math.nan, help="Override η_th (boiler efficiency)")
    parser.add_argument("--cc_label", default="", help="CSV label for CC train (default: CO2 capture in CHP)")
    parser.add_argument("--save", type=str, help="Save results to CSV file (provide filename)")

    args = parser.parse_args()
    
    # Handle presets mode
    if args.presets:
        run_presets()
        sys.exit(0)
    
    # Require tech, eta_old, eps_s for individual calculations
    if not all([args.tech, args.eta_old is not None, args.eps_s is not None]):
        parser.error("--tech, --eta_old, and --eps_s are required for individual calculations (or use --presets)")
    
    eta_th_override = None if math.isnan(args.eta_th) else args.eta_th
    cc_label = args.cc_label or None

    df = load_investment_costs(args.csv)
    inputs = CCInputs(
        base_tech=args.tech,
        eta_old=args.eta_old,
        eps_s=args.eps_s,
        boiler_type=args.boiler,
        e_th_cc=args.e_th_cc,
        eta_th_override=eta_th_override,
        capture_unit_tech=cc_label,
    )
    out = compute_cc_investment(df, inputs)

    print(f"C_I,new [EUR/MW_out] = {out.ci_new_eur_per_mw:,.2f}")
    print("Breakdown:")
    for k, v in out.breakdown.items():
        print(f"  {k:>13s}: {v:,.2f}")
    print(f"alpha    = {out.alpha:.6f}")
    print(f"eta_new  = {out.eta_new:.6f}")
    print(f"eta_steam= {out.eta_steam:.6f}")
    
    # Save to CSV if requested
    if args.save:
        try:
            save_results_to_csv(out, inputs, args.save)
            print(f"\nResults saved to: {args.save}")
        except Exception as e:
            print(f"\nFailed to save CSV: {e}")