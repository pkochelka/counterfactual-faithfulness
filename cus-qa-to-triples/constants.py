from enum import StrEnum
from pathlib import Path
import urllib.request

class _MemberType(StrEnum):
    pass

class CZMemberType(_MemberType):
    JMENO_UMELECKE_DILO = "jmeno_umelecke_dilo"
    JMENO_CLOVEK = "jmeno_clovek"
    JMENO = "jmeno"
    MISTO = "misto"
    DATUM = "datum"
    CISLO = "cislo"
    ZAMESTNANI = "zamestnani"


class SKMemberType(_MemberType):
    MENO_UMELECKE_DIELO = "meno_umelecke_dielo"
    MENO_CLOVEK = "meno_clovek"
    MENO = "meno"
    MIESTO = "miesto"
    DATUM = "datum"
    CISLO = "cislo"
    ZAMESTNANIE = "zamestnanie"


_DATA = Path(__file__).parent / "data"
_DATA.mkdir(parents=True, exist_ok=True)
CZ_DATA = _DATA / "cz"
SK_DATA = _DATA / "sk"

CZ_DATA.mkdir(parents=True, exist_ok=True)
SK_DATA.mkdir(parents=True, exist_ok=True)

# FACTUAL_PHRASED_DATA = _DATA / "Factual-phrased.xml"
# COUNTER_FACTUAL_PHRASED_DATA = _DATA / "CounterFactual-phrased.xml"


# FACTUAL_TRIPLES = _DATA / "Factual-triples.xml"
# COUNTER_FACTUAL_TRIPLES = _DATA / "CounterFactual-triples.xml"

CUS_QA_TEXT_CZ = CZ_DATA / "text-cz.parquet"
_URL_CUS_QA_TEXT_CZ = (
    "https://huggingface.co/datasets/ufal/cus-qa/resolve/main/text-CZ/dev.parquet"
)

CUS_QA_TEXT_SK = SK_DATA / "text-sk.parquet"
_URL_CUS_QA_TEXT_SK = (
    "https://huggingface.co/datasets/ufal/cus-qa/resolve/main/text-SK/dev.parquet"
)

if not CUS_QA_TEXT_CZ.exists():
    urllib.request.urlretrieve(_URL_CUS_QA_TEXT_CZ, CUS_QA_TEXT_CZ)
    print(f"Downloaded {CUS_QA_TEXT_CZ.resolve()}")

if not CUS_QA_TEXT_SK.exists():
    urllib.request.urlretrieve(_URL_CUS_QA_TEXT_SK, CUS_QA_TEXT_SK)
    print(f"Downloaded {CUS_QA_TEXT_SK.resolve()}")
