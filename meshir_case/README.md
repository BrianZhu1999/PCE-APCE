# MeshRIR S1/S32 pilot

Independent experiment package for measured MeshRIR reconstruction, forward
prediction and source localization.  This directory is not part of the clean
manuscript package.

- S1-M3969: 3D field reconstruction and acoustic-time prediction.
- S32-M441: 2D source localization with source-holdout candidates.
- Backend: Python/PyTorch; pilot devices: `cuda:2` and `cuda:3` only.
- Native data are read lazily from the public NPY ZIP archives and cached in a
  derived, audited form; the ZIP files remain unchanged.

The pilot is a gate.  It does not edit the manuscript and does not launch the
formal matrix automatically.
