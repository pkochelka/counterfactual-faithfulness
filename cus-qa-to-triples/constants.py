from enum import StrEnum
from pathlib import Path
import urllib.request


class MemberType(StrEnum):
    JMENO_UMELECKE_DILO = "jmeno_umelecke_dilo"
    JMENO_CLOVEK = "jmeno_clovek"
    JMENO = "jmeno"
    MISTO = "misto"
    DATUM = "datum"
    CISLO = "cislo"
    ZAMESTNANI = "zamestnani"


_DATA = Path(__file__).parent / "data"
_DATA.mkdir(parents=True, exist_ok=True)

FACTUAL_PHRASED_DATA = _DATA / "Factual-phrased.xml"
COUNTER_FACTUAL_PHRASED_DATA = _DATA / "CounterFactual-phrased.xml"


FACTUAL_TRIPLES = _DATA / "Factual-triples.xml"
COUNTER_FACTUAL_TRIPLES = _DATA / "CounterFactual-triples.xml"

CUS_QA_TEXT_CZ = _DATA / "cus-qa-text-cz.parquet"
_URL_CUS_QA_TEXT_CZ = (
    "https://huggingface.co/datasets/ufal/cus-qa/resolve/main/text-CZ/dev.parquet"
)

if not CUS_QA_TEXT_CZ.exists():
    urllib.request.urlretrieve(_URL_CUS_QA_TEXT_CZ, CUS_QA_TEXT_CZ)
    print(f"Downloaded {CUS_QA_TEXT_CZ.resolve()}")
