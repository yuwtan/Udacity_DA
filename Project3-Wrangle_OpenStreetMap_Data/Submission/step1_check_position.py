import xml.etree.cElementTree as ET

OSMFILE = "sample.osm"


def audit_street(osmfile):
    osm_file = open(osmfile, "r")
    pos = []
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.attrib:
                # save positions
                if tag == 'lat':
                    pos.append([float(elem.attrib['lat']), float(elem.attrib['lon'])])
                elif tag == 'lon':
                    pass


    osm_file.close()
    return pos

positions = audit_street(OSMFILE)

lat = []
lon = []

# assign longitudes and latitudes
for s in positions:
    lat.append(s[0])
    lon.append(s[1])

# make the plot
import matplotlib.pyplot as plt
plt.plot(lon, lat, 'o')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show()
