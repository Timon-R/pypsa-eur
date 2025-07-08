from __future__ import annotations
from pathlib import Path
import copy, yaml, re

_BANNER = "###### {} ######"
_FOCUS  = {"solver": {"options": "gurobi-numeric-focus"}}

_BIOMASS_EF = {
    "agricultural waste": 0, "fuelwood residues": 0,
    "secondary forestry residues": 0, "sawdust": 0,
    "residues from landscape care": 0, "grasses": 0,
    "woody crops": 0, "fuelwoodRW": 0, "manure": 0,
    "sludge": 0, "C&P_RW": 0,
}
_DEFAULT_SECTOR = {
    "renewable_emissions": {"enable": False},
    "solid_biomass_import": {"upstream_emissions_factor": 0},
}

def _dump(name: str, mapping: dict) -> str:
    """YAML dump preserving key order, without trailing newline."""
    return yaml.safe_dump({name: mapping}, sort_keys=False).rstrip()

# ------------------------------------------------------------------ helpers
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

def _add_bm0(name_tag: str, base_extras: dict, focus_set: set[str],
             out: list[str]) -> None:
    """Insert the biomass-zero variant right after ‘optimal’."""
    base_tag        = name_tag.rstrip("_")
    bm0_name        = "bm0" if name_tag == "" else f"{base_tag}_bm0"
    mapping         = copy.deepcopy(base_extras)
    # ensure sector exists and set biomass False
    mapping.setdefault("sector", {})
    mapping["sector"]["biomass"] = False
    if bm0_name in focus_set:
        mapping |= _FOCUS
    out.append(_dump(bm0_name, mapping))

# ---------------------------------------------------------------- generator
def generate_scenarios(
    slacks: list[float] | tuple[float, ...],
    numeric_focus: list[str] | tuple[str, ...] = (),
    *,
    special_cases: dict[str, bool] | None = None,
    include_bm0: bool = True,
) -> str:
    """Return complete scenario YAML text."""
    focus_set = set(numeric_focus)
    out: list[str] = []

    def add_block(title: str, tag: str, *, skip_bigger_slacks: bool = False) -> None:
        out.append(_BANNER.format(title))
        extras = _extras_for(tag)

        # --- optimal
        opt_name = f"{tag}optimal" if tag else "optimal"
        opt_map  = {"solving": {"mga": {"enable": False}}} | extras
        if opt_name in focus_set:
            opt_map |= _FOCUS
        out.append(_dump(opt_name, opt_map))

        # --- biomass-zero variant
        if include_bm0:
            _add_bm0(tag, extras, focus_set, out)

        # --- MGA variants
        for sense in ("min", "max"):
            for s in slacks:
                if skip_bigger_slacks and sense == "min" and s > 0.1 and "default" not in tag:
                    continue
                name = f"{tag}{sense}_{s}" if tag else f"{sense}_{s}"
                mapping = {"solving": {"mga": {"enable": True,
                                               "sense": sense,
                                               "slack": float(s)}}} | extras
                if name in focus_set:
                    mapping |= _FOCUS
                out.append(_dump(name, mapping))

    # main four families
    add_block("carbon cost",      "",           skip_bigger_slacks=True)
    add_block("710 carbon costs", "710_",       skip_bigger_slacks=True)
    add_block("default",          "default_")
    add_block("710 default",      "default_710_")

    # ------------------------------------------------ SPECIAL CASES
    if special_cases:
        out.append(_BANNER.format("SPECIAL CASES"))
        for name, need_focus in special_cases.items():
            tag  = next((p for p in ("default_710_", "default_", "710_")
                         if name.startswith(p)), "")
            body = name[len(tag):]
            extras = _extras_for(tag)
            m     = re.fullmatch(r"(min|max)_(\d+(?:\.\d+)?)", body)
            if m:
                sense, slack = m.groups()
                mapping = {"solving": {"mga": {"enable": True,
                                               "sense": sense,
                                               "slack": float(slack)}}} | extras
            elif body == "optimal":
                mapping = {"solving": {"mga": {"enable": False}}} | extras
            elif body == "bm0":
                mapping = copy.deepcopy(extras)
                mapping.setdefault("sector", {})
                mapping["sector"]["biomass"] = False
            else:
                raise ValueError(f"Cannot parse custom scenario name '{name}'")

            if need_focus:
                mapping |= _FOCUS
            out.append(_dump(name, mapping))

    return "\n\n".join(out) + "\n"

# ------------------------------------------------------------------- usage
if __name__ == "__main__":

    slacks         = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
    numeric_focus  = []          # e.g. ["default_min_0.15", "bm_0"]
    special_cases  = {}          # e.g. {"min_0.07": True}
    add_bm0_cases  = True        # toggle biomass-zero scenarios here

    yaml_text = generate_scenarios(slacks,
                                   numeric_focus,
                                   special_cases=special_cases,
                                   include_bm0=add_bm0_cases)

    Path("config/mga_scenarios.yaml").write_text(yaml_text, encoding="utf-8")
    print(yaml_text)