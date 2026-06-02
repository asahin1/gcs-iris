# gcs-iris

Simple path planning around obstacles in 2D using Graphs of Convex Sets (GCS) framework. Convex regions are computed using IRIS algorithm.

## Requirements and Core Dependencies

- [Drake](https://drake.mit.edu) - For IRIS algorithm and polytope computations
- [gcsopt](https://github.com/TobiaMarcucci/gcsopt) - Python library for GCS
- A `Gurobi` license - This implementation uses `Gurobi` as the `CVXPY` solver

## Setup

1. Install Python dependencies
``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements
```

2. Point to your `Gurobi` license file
``` bash
export GRB_LICENSE_FILE=<path_to_license_file>
```

3. Run
``` bash
python main.py
```
