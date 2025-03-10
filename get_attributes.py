# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import pypsa

n = pypsa.Network("results/test/networks/base_s_5___2050.nc")

biomass_stores = n.stores[n.stores.carrier == "agricultural waste"]
# print marginal costs
print(biomass_stores.marginal_cost)

sludge_stores = n.stores[n.stores.carrier == "sludge"]
# print marginal costs
print(sludge_stores.marginal_cost)
