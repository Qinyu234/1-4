"""Diagnostics on Trajectory: elements, resonance, MEGNO, encounters, AMD."""

from fairy_orbit.observe.amd import AMDSeries, amd_of_elements, extract_amd_series
from fairy_orbit.observe.calibration import (
    CalibrationSeries,
    epsilon_at_orbit,
    measure_calibration,
    td_breaking,
    tetrahedron_shape_error,
)
from fairy_orbit.observe.diagnose import Diagnosis, diagnose
from fairy_orbit.observe.elements_series import ElementSeries, extract_element_series
from fairy_orbit.observe.encounters import EncounterEvent, find_encounters
from fairy_orbit.observe.interest import a_order_changed, interestingness
from fairy_orbit.observe.resonance import ResonanceSeries, resonance_angles
from fairy_orbit.observe.closure import (
    ClosureResult,
    ClosureSeries,
    E_r,
    E_v,
    best_closure,
    best_closure_by_Er,
    closure_for_perm,
    closure_series,
    kabsch_rotation,
    radial_order,
    shape_score,
    velocity_score,
)
from fairy_orbit.observe.error_base import (
    DEFAULT_ERROR_BASE,
    ExpErrorBase,
    closure_drift_summary,
    log_excess_error,
    normalize_error,
)
from fairy_orbit.observe.choreography_verify import (
    ChoreographyVerifyResult,
    verify_choreography_Tn,
    verify_seed_choreography,
)
from fairy_orbit.observe.peo import PEOFilterResult, evaluate_peo
from fairy_orbit.observe.rep_error import (
    CHANNELS,
    RepErrorSeries,
    RepErrorSnapshot,
    RepSigmas,
    apply_sigmas,
    compute_sigmas,
    rep_error_series,
    weighted_score,
)
from fairy_orbit.observe.search import (
    FREE_NAMES,
    BeamConfig,
    BeamSearchResult,
    SearchBounds,
    grid_beam_search,
    result_to_dict,
)
from fairy_orbit.observe.score import ScoreBreakdown, ScoreFloors, score_summary

__all__ = [
    "Diagnosis",
    "diagnose",
    "ElementSeries",
    "extract_element_series",
    "EncounterEvent",
    "find_encounters",
    "ResonanceSeries",
    "resonance_angles",
    "a_order_changed",
    "interestingness",
    "ScoreFloors",
    "ScoreBreakdown",
    "score_summary",
    "AMDSeries",
    "amd_of_elements",
    "extract_amd_series",
    "CalibrationSeries",
    "measure_calibration",
    "epsilon_at_orbit",
    "tetrahedron_shape_error",
    "td_breaking",
    "ExpErrorBase",
    "DEFAULT_ERROR_BASE",
    "normalize_error",
    "log_excess_error",
    "closure_drift_summary",
    "ClosureResult",
    "ClosureSeries",
    "kabsch_rotation",
    "E_r",
    "E_v",
    "shape_score",
    "velocity_score",
    "closure_for_perm",
    "best_closure",
    "best_closure_by_Er",
    "radial_order",
    "closure_series",
    "CHANNELS",
    "RepErrorSnapshot",
    "RepErrorSeries",
    "RepSigmas",
    "rep_error_series",
    "compute_sigmas",
    "apply_sigmas",
    "weighted_score",
    "FREE_NAMES",
    "SearchBounds",
    "BeamConfig",
    "BeamSearchResult",
    "grid_beam_search",
    "result_to_dict",
    "PEOFilterResult",
    "evaluate_peo",
    "ChoreographyVerifyResult",
    "verify_choreography_Tn",
    "verify_seed_choreography",
]
