from __future__ import annotations

from .constants import VectorSpecies, Virus


DEFAULT_SEED_REPORTS = [
    {
        "virus": int(Virus.DENGUE),
        "species": int(VectorSpecies.AE_AEGYPTI),
        "label": "Ouagadougou dengue seed",
        "lat": 12.3714,
        "lon": -1.5197,
        "initial_reported_cases": 17098,
        "report_date": "2024-04-28",
        "source": "https://www.who.int/emergencies/disease-outbreak-news/item/2024-DON518",
        "source_note": "WHO reported 17,098 dengue cases in Burkina Faso in 2024 as of 28 April 2024.",
    },
    {
        "virus": int(Virus.ZIKA),
        "species": int(VectorSpecies.AE_AEGYPTI),
        "label": "Brasilia zika seed",
        "lat": -15.7939,
        "lon": -47.8828,
        "initial_reported_cases": 7352,
        "report_date": "2023-05-27",
        "source": "https://www.paho.org/sites/default/files/2023-06/2023-jun-phe-update-arbovirus-eng.pdf",
        "source_note": "PAHO reported 7,352 Zika cases in Brazil up to epidemiological week 21 of 2023.",
    },
    {
        "virus": int(Virus.YELLOW_FEVER),
        "species": int(VectorSpecies.AE_AEGYPTI),
        "label": "Lima yellow fever seed",
        "lat": -12.0464,
        "lon": -77.0428,
        "initial_reported_cases": 35,
        "report_date": "2025-04-26",
        "source": "https://www.who.int/emergencies/disease-outbreak-news/item/2025-DON570",
        "source_note": "WHO reported 35 confirmed yellow fever cases in Peru as of 26 April 2025.",
    },
    {
        "virus": int(Virus.WEST_NILE),
        "species": int(VectorSpecies.CULEX),
        "label": "Phoenix west nile seed",
        "lat": 33.4484,
        "lon": -112.0740,
        "initial_reported_cases": 2628,
        "report_date": "2023-12-31",
        "source": "https://www.cdc.gov/mmwr/volumes/74/wr/mm7421a1.htm",
        "source_note": "CDC reported 2,628 West Nile virus cases in the United States for 2023.",
    },
    {
        "virus": int(Virus.JAPANESE_ENCEPHALITIS),
        "species": int(VectorSpecies.CULEX),
        "label": "Canberra japanese encephalitis seed",
        "lat": -35.2809,
        "lon": 149.1300,
        "initial_reported_cases": 37,
        "report_date": "2022-04-28",
        "source": "https://www.who.int/emergencies/disease-outbreak-news/item/2022-DON365",
        "source_note": "WHO reported 37 cumulative confirmed and probable Japanese encephalitis cases in Australia as of 28 April 2022.",
    },
]
