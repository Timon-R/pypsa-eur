# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Generates scenario YAML files from a sample file

Arguments:
---------
sample_file : str
    File containing the sample values

output_dir : str
    Directory to save the scenario YAML files

Usage
-----
To run the script on the command line, type::

    python expand_sample.py path/to/sample.txt path/to/output_dir

The sample file should be formatted as follows::

    value1,value2,value3,...
"""

import logging
import os
import sys
from logging import getLogger

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    handlers=[logging.StreamHandler(sys.stdout)],  # Write logs to stdout
)
logger = getLogger(__name__)


def generate_scenarios(sample, output_dir: str, parameters: dict):
    try:
        os.makedirs(output_dir, exist_ok=True)

        for i, row in enumerate(sample):
            scenario = {"parameters": {}}
            for j, (param_name, param_details) in enumerate(parameters.items()):
                value = float(row[j])
                adjustments = param_details["config_file_location"]["adjustments"]
                type = "factor" if "factor" in adjustments["sector"] else "absolute"
                component = list(adjustments["sector"][type].keys())[0]
                attribute = list(adjustments["sector"][type][component].keys())[0]
                scenario["parameters"].setdefault(param_name, {"adjustments": {}})
                scenario["parameters"][param_name]["adjustments"].setdefault(
                    "sector", {}
                )
                scenario["parameters"][param_name]["adjustments"]["sector"].setdefault(
                    type, {}
                )
                scenario["parameters"][param_name]["adjustments"]["sector"][
                    type
                ].setdefault(component, {})
                scenario["parameters"][param_name]["adjustments"]["sector"][type][
                    component
                ][attribute] = value

            scenario_file = os.path.join(output_dir, f"model_{i}/sample_{i}.yaml")
            with open(scenario_file, "w") as f:
                yaml.dump(scenario, f)
            logger.info(f"Scenario saved to {scenario_file}")
    except Exception as e:
        logger.error(f"Error generating scenarios yaml files (expand_sample.py): {e}")
        raise e


if __name__ == "__main__":
    # {input} {params.parameters} {output} > {log} 2>&1
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_sector_network",
            opts="",
            clusters="38",
            ll="vopt",
            sector_opts="",
            planning_horizons="2030",
        )

    sample_file = snakemake.input.sample_file
    parameters = snakemake.params.config["parameters"]
    # output_files = snakemake.output.output
    output_dir = snakemake.params.output_dir

    morris_sample = np.loadtxt(sample_file, delimiter=",")
    generate_scenarios(morris_sample, output_dir, parameters)
