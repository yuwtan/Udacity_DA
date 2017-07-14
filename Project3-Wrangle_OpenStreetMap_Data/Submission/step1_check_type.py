import xml.etree.cElementTree as ET


OSM_FILE = "chicago_illinois.osm"


for _, element in ET.iterparse(OSM_FILE):
    if element and (element.tag == 'node' or element.tag == 'way'):
        for child in element:
            if child.attrib['k'] == 'type':
                print(child.attrib['v'])
                print(element.attrib['id'])