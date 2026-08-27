from . import linalg
from . import render
from . import dataset

from .render import pretty_num, prettify_text
from .linalg import (
    weighted_quadratic_norm,
    as_vec,
    as_mat,
    sym,
    min_pd_eig,
    is_psd,
    is_pd,
    sqrt_psd,
    pbh_stabilizable,
    pbh_detectable,
    dare_residual,
    euler_step,
    rk4_step,
    discretize_and_linearize_euler,
    discretize_and_linearize_rk4,
    lin_c2d_euler,
    lin_c2d_rk4,
)
from .dataset import (
    MatchedStatePair,
    get_initial_states,
    extract_initial_states,
    find_matching_initial_states,
    match_initial_states,
    compute_entry_error,
    create_error_dataset,
    compute_error_dataset,
)

__all__ = [
    # Submodules
    "linalg",
    "render",
    "dataset",

    # Render helpers
    "pretty_num",
    "prettify_text",

    # Linalg helpers
    "weighted_quadratic_norm",
    "as_vec",
    "as_mat",
    "sym",
    "min_pd_eig",
    "is_psd",
    "is_pd",
    "sqrt_psd",
    "pbh_stabilizable",
    "pbh_detectable",
    "dare_residual",
    "euler_step",
    "rk4_step",
    "discretize_and_linearize_euler",
    "discretize_and_linearize_rk4",
    "lin_c2d_euler",
    "lin_c2d_rk4",

    # Dataset helpers
    "MatchedStatePair",
    "get_initial_states",
    "extract_initial_states",
    "find_matching_initial_states",
    "match_initial_states",
    "compute_entry_error",
    "create_error_dataset",
    "compute_error_dataset",
]

