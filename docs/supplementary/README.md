# Supplementary movies and interactive views

Movies and interactive visualizations accompanying *Observation feedback can conceal competing dynamics*.

[Open the interactive collection](https://brianzhu1999.github.io/PCE-APCE/supplementary/)

## Contents

The collection contains eight supplementary movies and five interactive views.

| Movie | Subject | Manuscript |
| --- | --- | --- |
| 1 | Uncertain wave, spring and heat equations | Extended Data Fig. 2 |
| 2 | Five nonlinear dynamical systems | Fig. 2 |
| 3 | Lorenz–96 | Fig. 3a–f |
| 4 | Kuramoto–Sivashinsky | Fig. 3g–l |
| 5 | Kolmogorov flow | Fig. 3m–p |
| 6 | Measured cylinder-wake flow | Fig. 4 |
| 7 | Three-dimensional acoustic field (MeshRIR) | Extended Data Fig. 3 |
| 8 | Acoustic-source tracking | Extended Data Fig. 4 |

Movie 7 offers isosurface and slice views. Interactive viewers cover observation blackout, cylinder wakes, three-dimensional acoustics, source tracking and candidate evidence.

## Offline viewing

The MeshRIR interactive page requires the adjacent `meshrir_isosurface_data/` directory. The cylinder-wake viewer requires `viv_piv_data/`. Other interactive pages are self-contained.

## Build the collection page

From this directory:

```bash
python code/build_collection_portal.py
```

The builder uses `catalog.json`, `code/collection_portal.css` and `code/collection_portal.js`.

## Data sources

- [Measured VIV–PIV wake dataset](https://doi.org/10.57745/HPA87O)
- [MeshRIR acoustic-field dataset](https://doi.org/10.5281/zenodo.5002817)
