# scenario_generator.py
from __future__ import annotations
from pathlib import Path
import yaml, re

_BANNER = "###### {} ######"
_FOCUS  = {"solver": {"options": "gurobi-numeric-focus"}}

_BIOMASS_EF = {
    "agricultural waste": 0,
    "fuelwood residues": 0,
    "secondary forestry residues": 0,
    "sawdust": 0,
    "residues from landscape care": 0,
    "grasses": 0,
    "woody crops": 0,
    "fuelwoodRW": 0,
    "manure": 0,
    "sludge": 0,
    "C&P_RW": 0,
}

_DEFAULT_SECTOR = {
    "renewable_emissions": {"enable": False},
    "solid_biomass_import": {"upstream_emissions_factor": 0},
}

def _dump(name: str, mapping: dict) -> str:
    return yaml.safe_dump({name: mapping}, sort_keys=False).rstrip()

# -- helper to identify family & extras ---------------------------------
def _extras_for(tag: str) -> dict:
    if tag == "default_710_":
        return {"biomass": {"emission_factors": _BIOMASS_EF},
                "sector": {**_DEFAULT_SECTOR,
                           "co2_sequestration_potential": {2050: 710}}}
    if tag == "default_":
        return {"biomass": {"emission_factors": _BIOMASS_EF},
                "sector": _DEFAULT_SECTOR}
    if tag == "710_":
        return {"sector": {"co2_sequestration_potential": {2050: 710}}}
    return {}

# -- main generator ------------------------------------------------------
def generate_scenarios(
    slacks: list[float] | tuple[float, ...],
    numeric_focus: list[str] | tuple[str, ...] = (),
    *,
    special_cases: dict[str, bool] | None = None,
) -> str:
    focus_set = set(numeric_focus)
    out: list[str] = []

    def add_block(title: str, tag: str):
        out.append(_BANNER.format(title))
        extras = _extras_for(tag)
        for sense in ("min", "max"):
            for s in slacks:
                name = f"{tag}{sense}_{s}" if tag else f"{sense}_{s}"
                mapping = {"solving": {"mga": {"enable": True,
                                               "sense": sense,
                                               "slack": float(s)}}} | extras
                if name in focus_set:
                    mapping["solving"] |= _FOCUS
                out.append(_dump(name, mapping))
        opt_name = f"{tag}optimal" if tag else "optimal"
        opt_map  = {"solving": {"mga": {"enable": False}}} | extras
        if opt_name in focus_set:
            opt_map["solving"] |= _FOCUS
        out.append(_dump(opt_name, opt_map))

        # ---------- SPECIAL CASES -------------------------------------------
    if special_cases:
        out.append(_BANNER.format("SPECIAL CASES"))
        for name, need_focus in special_cases.items():
            # recognise family, sense, slack, optimal
            tag = next((p for p in ("default_710_", "default_", "710_") if name.startswith(p)), "")
            body = name[len(tag):]
            extras = _extras_for(tag)

            mm = re.fullmatch(r"(min|max)_(\d+(?:\.\d+)?)", body)
            if mm:
                sense, slack = mm.groups()
                mapping = {"solving": {"mga": {"enable": True,
                                               "sense": sense,
                                               "slack": float(slack)}}} | extras
            elif body == "optimal":
                mapping = {"solving": {"mga": {"enable": False}}} | extras
            else:
                raise ValueError(f"Cannot parse custom scenario name: {name}")

            if need_focus:
                mapping["solving"] |= _FOCUS
            out.append(_dump(name, mapping))

    add_block("carbon cost", "")
    add_block("710 carbon costs", "710_")
    add_block("default", "default_")
    add_block("710 default", "default_710_")

    return "\n\n".join(out) + "\n"

if __name__ == "__main__":
    
    slacks = [0.025, 0.05, 0.1, 0.15]
    numeric_focus = ["default_min_0.15","min_0.05","710_min_0.025"]
    special_cases = {
        "default_710_min_0.12": False,
        "default_min_0.12": False,
        "default_min_0.13": True,
        "min_0.03": True,
    }

    yaml_text = generate_scenarios(slacks,numeric_focus,special_cases=special_cases)
    Path("config/mga_scenarios.yaml").write_text(yaml_text, encoding="utf-8")
    print(yaml_text)