import xml.etree.cElementTree as ET

OSMFILE = "sample.osm"

expected_state_names = ["IL"]


def audit_item_type(item_types, item_name, expected_list):
    """check if the item_name is mentioned in expected_list, and add it to item_types if not"""
    m = item_name
    if m:
        if m not in expected_list:
            item_types.add(item_name)


def is_state_type(elem):
    return elem.attrib['k'] == "addr:state"


def audit_state(osmfile):
    osm_file = open(osmfile, "r")
    item_types = set()
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            state_name = ""

            for tag in elem.iter("tag"):
                if is_state_type(tag):
                    state_name = tag.attrib['v']
                    audit_item_type(item_types, state_name, expected_state_names)

            # print out the IN example
            if state_name == "IN":
                print(elem)
                for tag in elem.iter("tag"):
                    print(tag.attrib)

    osm_file.close()
    return item_types


state_types = audit_state(OSMFILE)

for s in state_types:
    print(s)

