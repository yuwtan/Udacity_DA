from pymongo import MongoClient

client = MongoClient()
db = client.local

# find restaurants with position info available
cursor = db.chicago.find({"pos":{"$exists":"true"},"amenity":"restaurant"})

lat = []
lon = []

# assign longitudes and latitudes
for document in cursor:
    s = document['pos']
    lat.append(s[0])
    lon.append(s[1])

# make the plot
import matplotlib.pyplot as plt
plt.plot(lon, lat, 'o')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show()