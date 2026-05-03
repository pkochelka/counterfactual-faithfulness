from constants import FACTUAL_PHRASED_DATA
from xml_tagged import import_from_xml_file

pts = import_from_xml_file(FACTUAL_PHRASED_DATA)

for pt in pts:
    if pt.id_ == 1:
        print(pt)