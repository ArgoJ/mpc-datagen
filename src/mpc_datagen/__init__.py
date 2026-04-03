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
    ROAVerifier,
    StabilityReport,
    AsymptoticStabilityReport,
    AlphaViolationStats,
    GrüneHorizonReport,
    LyapunovDescentReport,
    VerificationRender,

)


# Submodules
from . import linalg as mdg_linalg
from . import plots as mdg_plt
from . import extractor as mdg_extractor

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
    "ROAVerifier",
    "StabilityReport",
    "AsymptoticStabilityReport",
    "AlphaViolationStats",
    "GrüneHorizonReport",
    "LyapunovDescentReport",
    "VerificationRender",

    # Submodules
    "mdg_linalg",
    "mdg_plt",
    "mdg_extractor",

    # Helpers
    "add_temp_folder",
]


# Logger
from pkg_logger import setup_logger 
logger = setup_logger(__name__)