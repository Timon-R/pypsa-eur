# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
This script is used to generate the scenarios for the Global Sensitivity Analysis (GSA) using the SALib library.
The script reads the parameters from the GSA.yaml file and generates the scenarios for the GSA.
The scenarios are saved in the config/GSA_runs.yaml file.
"""

from pathlib import Path

import yaml


def get_GSA():
    fn = Path("config/GSA.yaml")
    if fn.exists():
        with fn.open() as f:
            return yaml.safe_load(f)
    return {}


def extract_results():
    resultsfolder = Path("results")
    gsa_config = get_GSA()
    result_variables = gsa_config.get("results", [])
    for variable in result_variables:
        type = variable.get("type")
        if type == "csv":
            for folder in resultsfolder.iterdir():
                if folder.is_dir():
                    for file in folder.iterdir():
                        if file.is_file():
                            pass
        elif type == "network":
            raise NotImplementedError("Network results not yet implemented.")
        else:
            raise ValueError(f"Unknown result type {type}. Has to be csv or network.")
