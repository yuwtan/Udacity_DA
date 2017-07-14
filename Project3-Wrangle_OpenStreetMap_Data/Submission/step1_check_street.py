import xml.etree.cElementTree as ET

OSMFILE = "sample.osm"

expected_street_names = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road",
                         "Trail", "Parkway", "Commons", "Circle", "Highway", "Park", "Place", "Plaza", "Terrace",
                         "Broadway"]


def audit_item_type(item_types, item_name, expected_list):
    """check if the item_name is mentioned in expected_list, and add it to item_types if not"""
    m = item_name
    if m:
        if m not in expected_list:
            item_types.add(item_name)


def is_street_type(elem):
    return elem.attrib['k'] == "addr:street"


def audit_street(osmfile):
    osm_file = open(osmfile, "r")
    item_types = set()
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_type(tag):
                    street_name = tag.attrib['v'].split()[-1]

                    type_len = len(item_types)
                    audit_item_type(item_types, street_name, expected_street_names)

                    # print the street example if it is added to item_types
                    if len(item_types) > type_len:
                        print(tag.attrib['v'])

    osm_file.close()
    return item_types


street_types = audit_street(OSMFILE)

for s in street_types:
    print(s)
