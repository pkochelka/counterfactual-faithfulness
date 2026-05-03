from xml.etree.ElementTree import Element, tostring
from xml.dom import minidom


def root_to_pretty_xml(root: Element, indent: int | str = 2) -> str:
    """
    Converts given etree to a pretty string.

    https://stackoverflow.com/a/28814053
    """
    if isinstance(indent, int):
        indent = indent * " "

    xmlstr = minidom.parseString(tostring(root)).toprettyxml(indent=indent)
    return xmlstr
