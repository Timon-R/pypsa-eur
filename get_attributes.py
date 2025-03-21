# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import pypsa

n = pypsa.Network("results/biomass_emissions/networks/base_s_39___2050.nc")

biomass_stores = n.stores[n.stores.carrier == "agricultural waste"]
# print marginal costs
# print(biomass_stores.marginal_cost)
print(biomass_stores.e_initial.sum())
