# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# maybe this can be done better like the other ones are loaded

PARAMETERS = gsa_config["parameters"]
GROUPS = set(param["groupname"] for param in PARAMETERS.values())
MODELRUNS = range((len(GROUPS) + 1) * gsa_config["general"]["replicates"])


rule create_sample:  #requires SALib to be installed (add that to env or create new one for this)
    input:
        gsa_config=config["run"]["GSA"]["file"],
    params:
        config=gsa_config,
    message:
        "Creating sample for GSA"
    output:
        output_file="GSA/morris_sample.txt",
    log:
        "logs/GSA/morris_sample.log",
    script:
        "../scripts/GSA/create_sample.py"


rule expand_sample:
    input:
        sample_file="GSA/morris_sample.txt",
    params:
        config=gsa_config,
        output_dir="GSA/modelruns",
    output:
        output=expand(
            "GSA/modelruns/model_{model_run}/sample_{model_run}.yaml",
            model_run=MODELRUNS,
        ),
    script:
        "../scripts/GSA/expand_sample.py"
