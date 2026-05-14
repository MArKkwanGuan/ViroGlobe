from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Virus(IntEnum):
    DENGUE = 0
    ZIKA = 1
    YELLOW_FEVER = 2
    WEST_NILE = 3
    JAPANESE_ENCEPHALITIS = 4


class VectorSpecies(IntEnum):
    AE_AEGYPTI = 0
    AE_ALBOPICTUS = 1
    CULEX = 2
    ANOPHELES = 3
    OTHER = 4


N_VIRUS = 5
N_SPECIES = 5
GRID_RESOLUTION_DEG = 0.05
DEFAULT_CLIMATE_WINDOW_DAYS = 7
MAX_CLIMATE_WINDOW_DAYS = 365

VIRUS_NAMES = [
    "dengue",
    "zika",
    "yellow_fever",
    "west_nile",
    "japanese_encephalitis",
]

SPECIES_NAMES = [
    "ae_aegypti",
    "ae_albopictus",
    "culex",
    "anopheles",
    "other",
]

SUPPORTED_STATIC_VARIABLES = ("lat", "lon", "elevation", "population", "landcover")
CLIMATE_VARIABLE_ALIASES = {
    "temperature": ("temperature", "t2m", "tas"),
    "precipitation": ("precipitation", "precip", "pr"),
    "humidity": ("humidity", "relative_humidity", "rh"),
}

LCCS_WATER = {80, 210}
LCCS_SNOW = {70, 220}

LCCS_TO_LANDTYPE = {
    10: 5,
    20: 8,
    30: 9,
    40: 4,
    50: 1,
    60: 10,
    70: 12,
    80: 11,
    90: 3,
    95: 3,
    100: 10,
    110: 9,
    120: 8,
    130: 8,
    140: 10,
    150: 10,
    160: 3,
    170: 3,
    180: 3,
    190: 1,
    200: 10,
    210: 11,
    220: 12,
}


@dataclass(frozen=True)
class SpeciesParam:
    t_min: float
    t_opt: float
    t_max: float
    bite_rate_25c: float
    mortality_base: float
    fertility: float
    maturation_rate: float
    carrying_scale: float


SPECIES_PARAMS = [
    SpeciesParam(13.0, 30.0, 40.0, 0.50, 0.030, 0.22, 0.12, 1.00),
    SpeciesParam(11.0, 28.0, 38.0, 0.40, 0.035, 0.18, 0.10, 0.85),
    SpeciesParam(10.0, 27.0, 37.0, 0.30, 0.033, 0.30, 0.11, 1.20),
    SpeciesParam(16.0, 29.0, 38.0, 0.35, 0.040, 0.24, 0.09, 1.00),
    SpeciesParam(15.0, 28.0, 36.0, 0.25, 0.045, 0.15, 0.08, 0.60),
]

VECTOR_COMPETENCE = np.array(
    [
        [0.90, 0.80, 0.85, 0.05, 0.02],
        [0.60, 0.65, 0.40, 0.10, 0.05],
        [0.01, 0.01, 0.01, 0.80, 0.70],
        [0.01, 0.01, 0.01, 0.05, 0.15],
        [0.05, 0.05, 0.05, 0.15, 0.10],
    ],
    dtype=np.float32,
)

EIP_THERMAL_CONST = np.array(
    [
        [105.0, 110.0, 120.0, 130.0, 140.0],
        [115.0, 120.0, 130.0, 140.0, 150.0],
        [150.0, 150.0, 150.0, 100.0, 110.0],
        [150.0, 150.0, 150.0, 140.0, 120.0],
        [140.0, 140.0, 140.0, 130.0, 130.0],
    ],
    dtype=np.float32,
)

EIP_THRESHOLD_C = np.full((N_SPECIES, N_VIRUS), 12.0, dtype=np.float32)

INCUBATION_DAYS = np.array([6.0, 6.0, 5.0, 7.0, 8.0], dtype=np.float32)
INFECTIOUS_DAYS = np.array([6.0, 7.0, 6.0, 8.0, 8.0], dtype=np.float32)
SEVERE_DAYS = np.array([7.0, 8.0, 9.0, 10.0, 10.0], dtype=np.float32)

BASE_SEVERE_FRACTION = np.array([0.08, 0.03, 0.15, 0.10, 0.14], dtype=np.float32)
BASE_FATAL_FRACTION = np.array([0.002, 0.0005, 0.05, 0.03, 0.18], dtype=np.float32)
REPORTING_RATE = np.array([0.15, 0.10, 0.30, 0.25, 0.20], dtype=np.float32)

CROSS_PROTECTION = np.zeros((N_VIRUS, N_VIRUS), dtype=np.float32)
CROSS_PROTECTION[Virus.DENGUE, Virus.ZIKA] = 0.12
CROSS_PROTECTION[Virus.ZIKA, Virus.DENGUE] = 0.10
CROSS_PROTECTION[Virus.WEST_NILE, Virus.JAPANESE_ENCEPHALITIS] = 0.08
CROSS_PROTECTION[Virus.JAPANESE_ENCEPHALITIS, Virus.WEST_NILE] = 0.08

ADE_SEVERE_AMPLIFICATION = np.zeros((N_VIRUS, N_VIRUS), dtype=np.float32)
ADE_SEVERE_AMPLIFICATION[Virus.DENGUE, Virus.ZIKA] = 0.30
ADE_SEVERE_AMPLIFICATION[Virus.ZIKA, Virus.DENGUE] = 0.20
ADE_SEVERE_AMPLIFICATION[Virus.WEST_NILE, Virus.JAPANESE_ENCEPHALITIS] = 0.15
ADE_SEVERE_AMPLIFICATION[Virus.JAPANESE_ENCEPHALITIS, Virus.WEST_NILE] = 0.15

LANDTYPE_PREFERENCE = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.00, 0.30, 0.20, 0.05, 0.05],
        [0.50, 1.00, 0.30, 0.10, 0.10],
        [0.10, 0.10, 1.00, 0.50, 0.30],
        [0.05, 0.20, 0.40, 1.00, 0.10],
        [0.20, 0.80, 0.20, 0.20, 1.00],
        [0.10, 0.60, 0.10, 0.10, 0.70],
        [0.15, 0.70, 0.15, 0.10, 0.80],
        [0.10, 0.50, 0.20, 0.15, 0.50],
        [0.05, 0.20, 0.25, 0.30, 0.20],
        [0.02, 0.05, 0.10, 0.05, 0.05],
        [0.01, 0.01, 0.05, 0.02, 0.01],
        [0.00, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=np.float32,
)
