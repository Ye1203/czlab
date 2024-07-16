# czlab
This document stores the code for calculating trajectory analysis, which is based on scanpy. paga_weight.py is adjusted based on sc.tl.paga() in scapy. This document stores the code for calculating trajectory analysis, which is based on scanpy. paga_weight.py is adjusted based on sc.tl.paga() in scapy. And on this basis, the difference in connectivities between different nodes is considered to draw the connectivities tree.

## How to use paga_weight.py
The version of package used in paga_weight
  -Python version: 3.11.5
  -Numpy version: 1.26.4
  -Scipy version: 1.13.1
  -Scanpy version: 1.10.1

after running sc.pp.neighbors

import paga_weight as pw
pw.paga(adata,root = start_cell,connectivities_threshold=0.5)

sc.pl.paga(adata, color = "connectivities_tree")

## Updated paga_distance.py
This is a new trajectory analysis algorithm, which is still under testing.
