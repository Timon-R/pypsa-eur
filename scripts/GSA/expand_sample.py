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
import sys
from logging import getLogger

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    handlers=[logging.StreamHandler(sys.stdout)],  # Write logs to stdout
)
logger = getLogger(__name__)


def set_nested_value(d, keys, value):
    """Recursively set a value in a nested dictionary."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def extract_keys(d, keys):
    for k, v in d.items():
        keys.append(k)
        if isinstance(v, dict):
            extract_keys(v, keys)


def generate_scenarios(sample, output_file: str, parameters: dict):
    try:
        scenarios = {}
        for i, row in enumerate(sample):
            scenario = {}
            for j, (param_name, param_details) in enumerate(parameters.items()):
                value = float(row[j])
                config_file_location = param_details["config_file_location"]
                keys = []
                extract_keys(config_file_location, keys)
                set_nested_value(scenario.setdefault(param_name, {}), keys, value)
            scenarios[f"modelrun_{i}"] = scenario

        yaml_content = yaml.dump(scenarios, default_flow_style=False)

        with open(output_file, "w") as f:
            f.write(yaml_content.replace("\nmodelrun_", "\n\nmodelrun_"))
        logger.info(f"Scenarios saved to {output_file}")
    except Exception as e:
        logger.error(f"Error generating scenarios yaml file (expand_sample.py): {e}")
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
    output_file = snakemake.output.output

    morris_sample = np.loadtxt(sample_file, delimiter=",")
    generate_scenarios(morris_sample, output_file, parameters)
