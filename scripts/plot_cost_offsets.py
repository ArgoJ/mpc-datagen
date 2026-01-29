import argparse
from pathlib import Path

from mpc_datagen.mpc_data import MPCDataset
from mpc_datagen import plots as mdg_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one-step descent check ΔV = V(n+1) - V(n) + l(x_n,u_n) from one dataset file."
    )
    parser.add_argument(
        "--dataset",
        default="/home/josua/programming_stuff/projects/mpc-datagen/data/double_integrator_none_N40_data",
        type=Path,
        help="Path to the HDF5 dataset file"
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Optional path to save the Plotly HTML output",
    )
    args = parser.parse_args()

    dataset = MPCDataset.load(args.dataset)
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {args.dataset}")

    mdg_plots.relaxed_dp_residual(
        dataset=dataset[:5],
        html_path=str(args.html) if args.html is not None else None,
    )


if __name__ == "__main__":
    main()
