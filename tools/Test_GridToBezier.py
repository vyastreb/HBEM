import GridToBezier as GtB
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

## Test on a binary matrix from simulation of oxidized contact

oxide = np.load("TestData/Oxide_seed2_kl8_A02_H04.npz")['oxide']
contact = np.load("TestData/Conduc_1_kl32_ks64_H05_a001.npz")['conduc']
contact = (contact + 1)/2.
# Intersect the inverse of the oxide with the contact
data = (1-oxide)*contact
# Label the connected components
labels, numL = label(data)

"""
Parameters:
::cutoff distance:: we will keep only those points to construct the Bezier spline which are separated by a distance larger than cutoff_distance
::scale:: the Bezier control points are scaled by scale in [0,1]
::separator_factor:: the intersection points are separated by a factor of separator_factor
::minimal_curvature_radius:: the minimal curvature radius of the Bezier spline
::smallest_contour_size:: if a contour has less than smallest_contour_size points, then discard it
"""
cutoff_distance = 2.
scale = 0.5 
separator_factor = 0.2
minimal_curvature_radius = 0.1
smallest_contour_size = 6

beziers = GtB.construct_Bezier_from_labels(labels,numL,cutoff_distance,scale,separator_factor,minimal_curvature_radius,smallest_contour_size)

# Plot the map and associated Bezier curves
data[data==0] = np.nan
plt.rcParams['figure.figsize'] = [10, 10]

fig, ax = plt.subplots()
ax.set_aspect(1)
# Plot the oxide
ax.imshow(data, extent=[0, data.shape[0], 0, data.shape[1]], cmap='rainbow', interpolation='none', origin='lower')
for i in range(1,len(beziers)):
    X = beziers[i][0]
    # Connect the last point to the first one
    X = np.append(X,np.array([X[0]]),axis=0)
    C1 = beziers[i][1]
    C2 = beziers[i][2]
    GtB.plot_cubic_bezier_spline(X, C1, C2, "k", plot_tangent = False, plot_point = True, label="__nolegend__")
plt.tight_layout()
plt.show()

