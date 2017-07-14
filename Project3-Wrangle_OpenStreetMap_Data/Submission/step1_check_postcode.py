import xml.etree.cElementTree as ET
import re

OSMFILE = "sample.osm"

postcode_format = re.compile(r'6[0-9][0-9][0-9][0-9]')


def is_postcode_type(elem):
    return elem.attrib['k'] == "addr:postcode"


def audit_postcode(osmfile):
    osm_file = open(osmfile, "r")
    item_types = set()
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":

            for tag in elem.iter("tag"):
                if is_postcode_type(tag):
                    postcode = tag.attrib['v']

                    if postcode_format.match(postcode):
                        pass
                    else:
                        item_types.add(postcode)

    osm_file.close()
    return item_types


postcodes = audit_postcode(OSMFILE)

for s in postcodes:
    print(s)

