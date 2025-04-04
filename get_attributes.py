# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import pypsa

# n = pypsa.Network("results/biomass_emissions/networks/base_s_39___2050.nc")

# biomass_stores = n.stores[n.stores.carrier == "agricultural waste"]
# # print marginal costs
# # print(biomass_stores.marginal_cost)
# print(biomass_stores.e_initial.sum())

n_default = pypsa.Network("results/test_optimal_default/networks/base_s_2___2050.nc")
n_new = pypsa.Network("results/test_optimal/networks/base_s_2___2050.nc")


def print_carrier_attributes(
    carrier: str, n_default: pypsa.Network, n_new: pypsa.Network
) -> None:
    print(
        f"Default {carrier.capitalize()} Capacity:",
        n_default.generators.loc[
            n_default.generators.carrier == carrier, "p_nom_opt"
        ].sum(),
    )
    print(
        f"New {carrier.capitalize()} Capacity:",
        n_new.generators.loc[n_new.generators.carrier == carrier, "p_nom_opt"].sum(),
    )

    print(
        f"Default Marginal Cost ({carrier.capitalize()}):",
        n_default.generators.loc[
            n_default.generators.carrier == carrier, "marginal_cost"
        ].unique(),
    )
    print(
        f"New Marginal Cost ({carrier.capitalize()}):",
        n_new.generators.loc[
            n_new.generators.carrier == carrier, "marginal_cost"
        ].unique(),
    )

    print(
        f"Total {carrier.capitalize()} Generation (Default):",
        n_default.generators_t.p.loc[:, n_default.generators.carrier == carrier]
        .sum()
        .sum(),
    )
    print(
        f"Total {carrier.capitalize()} Generation (New):",
        n_new.generators_t.p.loc[:, n_new.generators.carrier == carrier].sum().sum(),
    )


print(f"Total Costs: {n_default.objective} -> {n_new.objective}")

print_carrier_attributes("solar", n_default, n_new)
print_carrier_attributes("onwind", n_default, n_new)
