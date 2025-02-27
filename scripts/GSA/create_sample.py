# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Generates an unscaled sample from a list of parameters

Arguments:
---------
path_to_parameters : str
    File containing the parameters to generate a sample for

sample_file : str
    File path to save sample to

replicates : int
    The number of model runs to generate

Usage
-----
To run the script on the command line, type::

    python create_sample.py path/to/parameters.csv path/to/save.txt 10

The ``parameters.csv`` CSV file should be formatted as follows::

    name,group,indexes,min_value,max_value,dist,interpolation_index,action
    CapitalCost,pvcapex,"GLOBAL,GCPSOUT0N",500,1900,unif,YEAR,interpolate
    DiscountRate,discountrate,"GLOBAL,GCIELEX0N",0.05,0.20,unif,None,fixed

"""

import logging
import sys
from logging import getLogger

import numpy as np
from SALib.sample import morris

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    handlers=[logging.StreamHandler(sys.stdout)],  # Write logs to stdout
)
logger = getLogger(__name__)


def create_salib_problem(parameters: list) -> dict:
    """
    Creates SALib problem from scenario configuration.

    Arguments:
    ---------
    parameters: List
        List of dictionaries describing problem. Each dictionary must have
        'name', 'indexes', 'group' keys

    Returns:
    -------
    problem: dict
        SALib formatted problem dictionary

    Raises:
    ------
    ValueError
        If only one variable is given, OR
        If only one group is given
    """

    problem = {}
    problem["num_vars"] = len(
        parameters
    )  # this is the number of parameters (to be extracted of the configfile)
    if problem["num_vars"] <= 1:
        raise ValueError(
            f"Must define at least two variables in problem. User defined "
            f"{problem['num_vars']} variable(s)."
        )

    names = []
    bounds = []
    groups = []
    for param_name, param_details in parameters.items():
        names.append(param_name)  # Use the parameter name directly
        groups.append(param_details["groupname"])
        min_value = param_details["min"]
        max_value = param_details["max"]
        bounds.append([min_value, max_value])

    problem["names"] = names
    problem["bounds"] = bounds
    problem["groups"] = groups
    num_groups = len(set(groups))
    if num_groups <= 1:
        raise ValueError(
            f"Must define at least two groups in problem. User defined "
            f"{num_groups} group(s)."
        )
    return problem


def main(parameters: dict, sample_file: str, replicates: int):
    problem = create_salib_problem(
        parameters
    )  # this needs to be changed to sth new as the parameters are in a different format

    sample = morris.sample(
        problem,
        N=100,
        optimal_trajectories=replicates,
        local_optimization=True,
        seed=42,
    )

    np.savetxt(sample_file, sample, delimiter=",")
    logger.info(f"Sample saved to {sample_file}")


if __name__ == "__main__":
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

    config = snakemake.params.config
    output_file_name = snakemake.output.output_file

    replicates = config["general"]["replicates"]
    parameters = config["parameters"]

    main(parameters, output_file_name, replicates)
