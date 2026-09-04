# Release scope

The repository is an independent code release rather than a mirror of the
working manuscript tree. It includes source modules, experiment runners,
figure-generation scripts, tests and protocol documentation.

The following remain outside the release by design:

- raw or private measurements, including the Baoding recordings;
- remote workstation result trees and audit archives;
- generated arrays, figures, videos, PDFs and LaTeX build products;
- manuscript source files and author metadata from the clean manuscript;
- discontinued-case archives and exploratory checkpoints;
- credentials, local configuration and machine-specific absolute paths.

Machine-specific paths in the included protocol matrices use repository-relative
examples under `data/`, `external/` and `results/`. Path placeholders such as
`<PUBLIC_DATA_ROOT>` and `<HILDA_RESULTS_ROOT>` in case configurations are
documentation markers. Supply corresponding local paths through configuration
files, command-line arguments or environment variables before running a case.
