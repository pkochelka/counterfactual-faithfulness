from constants import (
    FACTUAL_PHRASED_DATA,
    FACTUAL_TRIPLES,
    COUNTER_FACTUAL_PHRASED_DATA,
    COUNTER_FACTUAL_TRIPLES,
)
from xml_triples import tagged_file_to_triples_xml_file

if __name__ == "__main__":
    tagged_file_to_triples_xml_file(
        FACTUAL_PHRASED_DATA,
        FACTUAL_TRIPLES
    )
    tagged_file_to_triples_xml_file(
        COUNTER_FACTUAL_PHRASED_DATA,
        COUNTER_FACTUAL_TRIPLES
    )
    