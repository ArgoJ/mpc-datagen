# Generation
from .generation import (
    MPCDataGenerator,
    EpsBandConfig,
    UniqueBoundedSampler,
    SamplerBase,
    add_temp_folder
)

# Data structures
from .mpc_data import (
    MPCData,
    MPCDataset,
    MPCConfig,
    MPCMeta,
    MPCTrajectory,
    LinearLSCost,
    LinearSystem,
    Constraints,
)

# Dataset utilities
from .utils.dataset import (
    create_error_dataset,
    compute_error_dataset,
    find_matching_initial_states,
    match_initial_states,
    get_initial_states,
    extract_initial_states,
    MatchedStatePair,
)


# Verification
from .verification import (
    StabilityVerifier,
    StabilityReport,
    AsymptoticStabilityReport,
    AlphaViolationStats,
    GrüneHorizonReport,
    LyapunovDescentReport,
    VerificationRender,
)

# Region of Attraction (ROA) - Analytic & Empirical
from .roa import (
    ROAVerifier,
    AnalyticROAVerifier,
    AnalyticROAReport,
    AnalyticROARender,
    EmpiricalROAEstimator,
    EmpiricalROAReport,
    EmpiricalROARender,
    TrajectoryStatus,
    SampledPoint,
)


# Submodules
from . import adapters as mdg_adapters
from . import utils as mdg_utils
from . import plots as mdg_plt
from . import roa as mdg_roa

__all__ = [
    # Data structures
    "MPCData",
    "MPCDataset",
    "MPCConfig",
    "MPCMeta",
    "MPCTrajectory",
    "LinearLSCost",
    "LinearSystem",
    "Constraints",
    "EpsBandConfig",

    # Generation
    "MPCDataGenerator",
    "UniqueBoundedSampler",
    "SamplerBase",

    # Verification
    "StabilityVerifier",
    "StabilityReport",
    "AsymptoticStabilityReport",
    "AlphaViolationStats",
    "GrüneHorizonReport",
    "LyapunovDescentReport",
    "VerificationRender",

    # Region of Attraction (ROA)
    "ROAVerifier",
    "AnalyticROAVerifier",
    "AnalyticROAReport",
    "AnalyticROARender",
    "EmpiricalROAEstimator",
    "EmpiricalROAReport",
    "EmpiricalROARender",
    "TrajectoryStatus",
    "SampledPoint",

    # Submodules & Utilities
    "mdg_adapters",
    "mdg_utils",
    "mdg_plt",
    "mdg_roa",

    # Dataset utilities
    "create_error_dataset",
    "compute_error_dataset",
    "find_matching_initial_states",
    "match_initial_states",
    "get_initial_states",
    "extract_initial_states",
    "MatchedStatePair",

    # Helpers
    "add_temp_folder",
]


# Logger
from pkg_logger import setup_logger 
logger = setup_logger(__name__)