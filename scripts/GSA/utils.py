# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import os
from logging import getLogger
from pathlib import Path

import pandas as pd
import yaml

logger = getLogger(__name__)


def get_model_run_scenario_from_input_filepath(filename: str):
    """
    Parses filepath to extract useful bits

    "results/{{scenario}}/model_{modelrun}/data/{input_file}.csv"
    """
    filepath, name = os.path.split(filename)
    param = os.path.splitext(name)[0]
    scenario_path, model_run = os.path.split(filepath)
    resultsscenario, _ = os.path.split(scenario_path)
    scenario = os.path.split(resultsscenario)[1]
    return {
        "model_run": model_run,
        "scenario": scenario,
        "param": param,
        "filepath": filepath,
    }


def get_model_run_scenario_from_filepath(filepath: str) -> dict:
    """
    Parses filepath to extract useful bits.

    Input filepath is expected in the form of:
        "results/{scenario}/{modelrun}/results/{input_file}.csv"

    Parameters
    ----------
    file : str
        file path from root directory

    Returns
    -------
    Dict
        With model_run, scenario, OSeMOSYS parameter, and filepath directory
    """
    f = Path(filepath)
    parts = f.parts
    return {
        "model_run": parts[2],
        "scenario": parts[1],
        "param": f.stem,
        "filepath": str(f.parent),
    }


def read_results(input_filepath: str) -> pd.DataFrame:
    extension = os.path.splitext(input_filepath)[1]
    if extension == ".parquet":
        df = pd.read_parquet(input_filepath)
    elif extension == ".csv":
        df = pd.read_csv(input_filepath)
    elif extension == ".feather":
        df = pd.read_feather(input_filepath)
    return df


def write_results(df: pd.DataFrame, output_filepath: str, index=None) -> None:
    """
    Write out aggregated results to disk by scenario

    Arguments:
    ---------
    df: pd.DataFrame
        Dataframe to write out
    output_filepath: str
        Path to the output file
    index=None
        Whether to write out the index or not
    """
    extension = os.path.splitext(output_filepath)[1]
    if extension == ".parquet":
        df.to_parquet(output_filepath, index=index)
    elif extension == ".csv":
        df.to_csv(output_filepath, index=index)
    elif extension == ".feather":
        if index:
            df = df.reset_index()
        df.to_feather(output_filepath)


def parse_yaml(path: str) -> dict:
    """
    Parses a YAML file to a dictionary

    Parameters
    ----------
    path : str
        input path the yaml file

    Returns
    -------
    parsed_yaml : Dict
        parsed YAML file

    Raises
    ------
    YAMLError
        If the yaml file can't be loaded
    """
    with open(path) as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return parsed_yaml
