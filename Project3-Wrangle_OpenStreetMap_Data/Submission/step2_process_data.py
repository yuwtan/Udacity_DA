import xml.etree.cElementTree as ET
import codecs
import json
import re

OSM_FILE = "chicago_illinois.osm"
JSON_FILE = "chicago_illinois.json"

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problem_chars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

# the list of information related to record creation that we want to save
CREATED_INFO = ["version", "changeset", "timestamp", "user", "uid"]

# the mapping we use to standardize street names
street_mapping = {"St": "Street",
                  "St.": "Street",
                  "Rd.": "Road",
                  "Ave": "Avenue",
                  "Ct": "Court",
                  "Cir": "Circle",
                  "Ln": "Lane",
                  "Dr": "Drive"
                  }


# the mapping we use to standardize street names
state_mapping = {"Il": "IL",
                 "Illinois": "IL",
                 "IL - Illinois": "IL"
                 }


def update_street(name, mapping):
    """This function update a street name input with the given mapping"""

    words = name.split()
    st_name = words[-1]
    for abbr in mapping:
        if abbr == st_name:
            st_name = mapping[abbr]
            break

    words[-1] = st_name
    name = " ".join(words)
    return name


def update_state(name, mapping):
    """This function update a state name input with the given mapping"""

    for abbr in mapping:
        if abbr == name:
            name = mapping[abbr]
            break

    return name


def shape_element(element):
    node = {}

    if element.tag == "node" or element.tag == "way":
        node['tag_type'] = element.tag
        created = {}
        pos = []

        for key in element.attrib:

            # save the creation information to dictionary "created"
            if key in CREATED_INFO:
                created[key] = element.attrib[key]
            # save the position to list "pos"
            elif key == 'lat':
                pos = [float(element.attrib['lat']), float(element.attrib['lon'])]
            elif key == 'lon':
                pass
            # save other information
            else:
                node[key] = element.attrib[key]

        # save "created" as one element of dictionary "node"
        if created:
            node['created'] = created

        # save "pos" as one element of dictionary "node"
        if pos:
            node['pos'] = pos

        addr = {}
        ref = []

        for child in element:

            if child.tag == 'tag':
                # first check if the tag value has problematic chars
                if problem_chars.match(child.attrib['k']):
                    continue
                # then check if the tag value starts with "addr:"
                elif child.attrib['k'].startswith('addr:'):
                    # if there is only one colon
                    if lower.match(child.attrib['k'].split(':', 1)[1]):
                        # update street names
                        if child.attrib['k'].split(':', 1)[1] == "street":
                            addr[child.attrib['k'].split(':', 1)[1]] = update_street(child.attrib['v'], street_mapping)
                        # update state names
                        elif child.attrib['k'].split(':', 1)[1] == "state":
                            addr[child.attrib['k'].split(':', 1)[1]] = update_state(child.attrib['v'], state_mapping)
                        else:
                            addr[child.attrib['k'].split(':', 1)[1]] = child.attrib['v']
                    else:
                        pass
                elif lower.match(child.attrib['k']):
                    node[child.attrib['k']] = child.attrib['v']
                elif lower_colon.match(child.attrib['k']):
                    words = child.attrib['k'].split(':')
                    node["_".join(words)] = child.attrib['v']

            # save ref info in list "ref"
            elif child.tag == 'nd':
                ref.append(child.attrib['ref'])

        # save "ref" as one element of dictionary "node"
        if ref:
            node['node_refs'] = ref

        # save "addr" as one element of dictionary "node"
        if addr:
            node['address'] = addr

        return node

    else:
        return None


def process_map(file_in, pretty=False):
    file_out = JSON_FILE
    json_data = []

    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                json_data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2) + "\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return json_data


data = process_map(OSM_FILE, False)
