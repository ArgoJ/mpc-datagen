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
from . import adapters
from . import utils
from . import plots as mdg_plt
from . import roa as mdg_roa

# Backward-compatibility submodules
mdg_adapters = adapters
mdg_utils = utils
mdg_linalg = utils.linalg
mdg_extractor = adapters.acados

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
    "adapters",
    "mdg_adapters",
    "utils",
    "mdg_utils",
    "mdg_linalg",
    "mdg_plt",
    "mdg_extractor",
    "mdg_roa",

    # Helpers
    "add_temp_folder",
]


# Logger
from pkg_logger import setup_logger 
logger = setup_logger(__name__)