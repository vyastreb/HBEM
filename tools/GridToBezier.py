"""
A tool that converts a grid of points (binary matrices) into a set of smooth curves describing the contours of non-connected regions.
The tool is based on the OpenCV library to find contours and on cubic Bezier splines.
Author: Vladislav A. Yastrebov, CNRS, MINES Paris - PSL, Centre des Materiaux, UMR 7633, BP 87, 91003 Evry, France
Licencse: CC0
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import linalg as LA

def compute_control_points(P,scale):
    N = len(P)
    Nt = N - 1
    T = np.zeros((Nt, 2))
    Tnew = np.zeros((N, 2))
    d = np.zeros(Nt)

    # Calculate tangent vectors
    for i in range(Nt):
        T[i] = P[(i+1)%Nt] - P[i]
        d[i] = np.linalg.norm(T[i])
        if d[i] == 0:
            # Return error line number  
            print("Zero length line segment detected.")
            exit(1)
    
    # Calculate tangent vectors as weighted average of neighboring line segments
    for i in range(Nt):
         Tnew[i] = (d[(i-1)%Nt]*T[(i-1)%Nt] + d[(i)%Nt]*T[(i)%Nt]) / (d[(i-1)%Nt] + d[(i)%Nt])

    # Calculate control points
    C1 = np.zeros((Nt, 2))
    C2 = np.zeros((Nt, 2))
    for i in range(Nt):
        xi1 = scale * d[i] / (d[i] + d[(i-1)%Nt])
        C1[i] = P[i] + xi1 * Tnew[i]

        xi2 = scale * d[(i-1)%Nt] / (d[i] + d[(i-1)%Nt])
        C2[(i-1)%Nt] = P[i] - xi2 * Tnew[i]

    return C1, C2
    
def cubic_bezier(P0, P1, P2, P3, t):
    return ((1-t)**3)*P0 + 3*((1-t)**2)*t*P1 + 3*(1-t)*(t**2)*P2 + (t**3)*P3

def cubic_bezier_derivative(P0, P1, P2, P3, t):
    return 3*(1-t)**2*(P1-P0) + 6*(1-t)*t*(P2-P1) + 3*t**2*(P3-P2)

def cubic_bezier_curvature(P0, P1, P2, P3, t):
    dt = 1e-6
    return (cubic_bezier_derivative(P0, P1, P2, P3, t+dt) - cubic_bezier_derivative(P0, P1, P2, P3, t))/dt

def curvature_radius(P0, P1, P2, P3, t):
    P_prime = cubic_bezier_derivative(P0, P1, P2, P3, t)
    P_double_prime = cubic_bezier_curvature(P0, P1, P2, P3, t)
    sign = np.sign(np.cross(P_prime, P_double_prime))

    num = np.linalg.norm(P_prime)**3
    denom = np.linalg.norm(np.cross(P_prime, P_double_prime))

    if denom != 0:
        return sign*(num / denom)
    else:
        return np.inf  # Straight line (infinite radius)

def plot_cubic_bezier_spline(P, C1, C2, color, plot_tangent = False, plot_point = False,label="__nolegend__"):
    t = np.linspace(0, 1, 100)
    N = len(P)

    for i in range(N-1):
        P0 = P[i]
        P1 = C1[i]
        P2 = C2[i]
        P3 = P[(i+1)%N]
        Bezier_curve = np.array([cubic_bezier(P0, P1, P2, P3, t_val) for t_val in t])
        if i == 0:
            plt.plot(Bezier_curve[:, 0], Bezier_curve[:, 1], str(color),lw="2",label=label)
        else:
            plt.plot(Bezier_curve[:, 0], Bezier_curve[:, 1], str(color),lw="2",label="__nolegend__")
        if plot_tangent:
            plt.plot([P0[0], P1[0]], [P0[1], P1[1]], 'b-')
            plt.plot([P2[0], P3[0]], [P2[1], P3[1]], 'b-')
            plt.plot(P1[0],P1[1], 'bs')
            plt.plot(P2[0],P2[1], 'bs')
        if plot_point:
            plt.plot(P0[0],P0[1], 'go')
            plt.plot(P3[0],P3[1], 'go')

    plt.axis('equal')

## Test

# X = np.array([[0,0],[-1,0],[-1,3],[2,4],[3,4],[4,3],[4,2],[3,1],[2,1],[1,2],[0,0]])
# scale = 0.5
# fig,ax = plt.subplots()

# C1, C2 = compute_control_points(X,scale)

# plt.plot(X[:, 0], X[:, 1], "-", markersize="20",color="grey", lw=2,label="OpenCV Contour")
# for i,x in enumerate(X):
#     plt.text(x[0]+np.random.random()*0.2,x[1]+np.random.random()*0.2,str(i))
# plot_cubic_bezier_spline(X, C1, C2,"red", plot_tangent = True, plot_point = True)
# plt.show()


'''
Resampling: keep only (1) min/max point, (2) XY extremum points i.e. with the same X or Y coordinate and (3) points which sufficiently far one from another
The second is controlled by the parameter cutoff_distance
'''
def coarsen_MinMax(Xin,cutoff_distance=3.):
    if Xin.shape[0] < 5:
        return Xin
    
    N = Xin.shape[0]

    xmax = np.where(Xin[:,0] == np.max(Xin[:,0]))[0]
    xmin = np.where(Xin[:,0] == np.min(Xin[:,0]))[0]
    ymax = np.where(Xin[:,1] == np.max(Xin[:,1]))[0]
    ymin = np.where(Xin[:,1] == np.min(Xin[:,1]))[0]

    keep_points = [xmax[0],xmin[0],ymax[0],ymin[0]] 
    # if 0 point is in keep points, then remove it
    if 0 in keep_points:
        keep_points.remove(0)

    NEWX = np.array([[Xin[0,0],Xin[0,1]]])
    for i in range(1,N):
        if Xin[i,0] == NEWX[-1,0] or Xin[i,1] == NEWX[-1,1] and (i not in keep_points):
            meanX = 0.5*(Xin[i,0]+NEWX[-1,0])
            meanY = 0.5*(Xin[i,1]+NEWX[-1,1])
            NEWX[-1] = np.array([[meanX,meanY]])
        else:
            NEWX = np.append(NEWX,np.array([[ Xin[i,0],Xin[i,1] ]]),axis=0)
    Xin = NEWX
    oldN = N
    N = Xin.shape[0]
    if N > oldN:
        print("Problem with coarsening")
        exit(1)
    xmax = np.where(Xin[:,0] == np.max(Xin[:,0]))[0]
    xmin = np.where(Xin[:,0] == np.min(Xin[:,0]))[0]
    ymax = np.where(Xin[:,1] == np.max(Xin[:,1]))[0]
    ymin = np.where(Xin[:,1] == np.min(Xin[:,1]))[0]

    keep_points = [xmax[0],xmin[0],ymax[0],ymin[0]] 
    # if 0 point is in keep points, then remove it
    if 0 in keep_points:
        keep_points.remove(0)

    NEWX = np.array([[Xin[0,0],Xin[0,1]]])
    for i in range(1,N):
        d1 = NEWX[-1] - Xin[i]
        if np.linalg.norm(d1) > cutoff_distance or (i in keep_points):
            NEWX = np.append(NEWX,np.array([[ Xin[i,0],Xin[i,1] ]]),axis=0)
    if not (NEWX[-1,0] == NEWX[0,0] and NEWX[-1,1] == NEWX[0,1]):
        NEWX = np.append(NEWX,np.array([[Xin[0,0],Xin[0,1]]]),axis=0)
    return NEWX

def remove_bezier_nodes_with_too_small_curvature(P, C1, C2, min_curvature_radius=0.1):
    if len(P) < 7:
        return P #,C1,C2
    """
    compute curvature at every control point P and adjust control points if the curvature is too small .
    """
    N = len(P)
    r0 = np.zeros(N)
    r1 = np.zeros(N)
    for i in range(N-1):
        P0 = P[i]
        P1 = C1[i]
        P2 = C2[i]
        P3 = P[(i+1)%N]
        r0[i] =  curvature_radius(P0, P1, P2, P3, 0)
        r1[i] =  curvature_radius(P0, P1, P2, P3, 1)
    # Check if curvature is too small
    tobedeleted = np.array([],dtype=int)
    for i in range(N-1):
        if abs(r0[i]) < min_curvature_radius or abs(r1[(i-1)%(N-1)]) < min_curvature_radius:
            tobedeleted = np.append(tobedeleted,i)
    tobedeleted = np.unique(tobedeleted)
    if len(tobedeleted) == 0:
        return P

    newP = np.zeros((P.shape[0]-len(tobedeleted),2))
    newC1 = np.zeros((C1.shape[0]-len(tobedeleted),2))
    newC2 = np.zeros((C2.shape[0]-len(tobedeleted),2))
    id = 0
    print("Points to be deleted: ", tobedeleted)
    for i in range(N-1):
        if i not in tobedeleted and (i+1)%(N-1) not in tobedeleted:
            newP[id] = P[i]
            newC1[id] = C1[i]
            newC2[id] = C2[i]
            id += 1
        elif i not in tobedeleted and (i+1)%(N-1) in tobedeleted:
            newP[id] = P[i]
            newC1[id] = C1[i]
            newC2[id] = C2[(i+1)%(N-1)]
            id += 1
    if 0 in tobedeleted:
        for i in range(1,N-1):
            if i not in tobedeleted:
                newP[id] = newP[0]
                break
    print("newP=",newP)
    return newP #,newC1,newC2 

def line_intersection(p1, p2, p3, p4):
    """
    Check if the line segment (p1, p2) intersects with the line segment (p3, p4).
    """
    # Unpack points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Compute differences
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    # Determinant
    det = dx1 * dy2 - dy1 * dx2
    if det == 0:
        return 123,123  # Parallel lines

    # Relative positions
    u = ((x3 - x1) * dy1 + (y1 - y3) * dx1) / det
    v = ((x3 - x1) * dy2 + (y1 - y3) * dx2) / det

    # Intersection occurs if u and v are between 0 and 1
    if not (0 <= u <= 1 and 0 <= v <= 1):
        return 123,123
    else:
        return u, v

def check_self_intersection(points):
    """
    Check if a closed loop formed by a set of ordered points has self-intersections.
    """
    n = len(points)
    for i in range(n):
        for j in range(i + 3, n - 2):
            # Ensure not checking adjacent segments or segment against itself
            if j % n == (i + 1) % n or j % n == i:
                continue

            u,v = line_intersection(points[i], points[(i + 1) % n], points[j % n], points[(j + 1) % n])
            if (u,v) != (123,123):
                if u <= 0.5 and v <= 0.5:
                    return (i+1),(j+1)%n
                elif u > 0.5 and v > 0.5:
                    return i,j
                elif u <= 0.5 and v > 0.5:
                    return (i+1)%n,j
                elif u > 0.5 and v <= 0.5:
                    return i,(j+1)%n
    return -1,-1  # No intersections

def check_and_repair_self_intersection(ci,XIN,separator_factor=0.2):
        intersection_i,intersection_j = check_self_intersection(XIN)
        if intersection_i != -1 and intersection_j != -1:
            xprev   = XIN[(intersection_i-1)%XIN.shape[0]].copy()
            x0      = XIN[intersection_i].copy()
            xnext   = XIN[(intersection_i+1)%XIN.shape[0]].copy()
            yprev   = XIN[(intersection_j-1)%XIN.shape[0]].copy()
            y0      = XIN[intersection_j].copy()
            ynext   = XIN[(intersection_j+1)%XIN.shape[0]].copy()
            # Compute two tangents at the intersection points and take the weighted average
            t1 = x0 - xprev
            t2 = xnext - x0
            t0 = (t1 + t2)/2.
            t0 /= np.linalg.norm(t0)
            n0 = np.array([-t0[1], t0[0]])
            dist = max((np.dot(t1,n0) + np.dot(t2,n0))/2.,1.)
            XIN[intersection_i] = x0 + separator_factor*dist*n0

            t1 = y0 - yprev
            t2 = ynext - y0
            t0 = (t1 + t2)/2.
            t0 /= np.linalg.norm(t0)
            n1 = np.array([-t0[1], t0[0]])
            dist = max((np.dot(t1,n1) + np.dot(t2,n1))/2.,1.)
            XIN[intersection_j] = y0 + separator_factor*dist*n1
            print("Self-intersection found for contour ", ci, "... repaired")
            
            return XIN,True
        else:
            return XIN,False


# Function that constructs Bezier splines from a set of labels
def construct_Bezier_from_labels(labels,numL,cutoff_distance,scale,separator_factor,minimal_curvature_radius,smallest_contour_size):
    N0 = labels.shape[0]
    all_contours = []

    # Iterate through each label to find contours
    for label_id in range(1, numL + 1):
        # Create a binary image for the current label
        label_image = np.uint8(labels == label_id) * 255

        # Find contours using OpenCV
        contours, _ = cv2.findContours(label_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # Add contours to the list
        all_contours.extend(contours)

    print("Number of contours: ", len(all_contours))

    all_bezier_points =  {}
    new_ci = 0
    for ci, contour in enumerate(all_contours):
        # Extract x and y coordinates from the contour
        # Contour structure: array of [[x, y]] coordinates
        x_coords = contour[:, 0, 0]
        y_coords = contour[:, 0, 1]

        # Connect the ends
        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])

        XIN = np.zeros((x_coords.shape[0], 2))
        for i in range(XIN.shape[0]):
            XIN[i, 0] = x_coords[i] + 0.5
            XIN[i, 1] = y_coords[i] + 0.5

        if XIN.shape[0] > smallest_contour_size:
            XIN,fixed = check_and_repair_self_intersection(ci,XIN,separator_factor)
            if fixed:
                while fixed:
                    XIN,fixed = check_and_repair_self_intersection(ci,XIN,separator_factor)

            N0 = XIN.shape[0]
            NEWX = coarsen_MinMax(XIN,cutoff_distance)
            N1 = NEWX.shape[0]
            if N1 > N0:
                print(">> Problem with coarsening")
                exit(1)
            # NEWX = XIN
            if NEWX.shape[0] >= 3:
                C1, C2 = compute_control_points(NEWX,scale)  
            else:
                continue

            # # Check consistency of C1 and C2, so that there's no kink points
            # chk = check_consistency(NEWX,C1,C2)
            # if chk != -1:
            #     NEWX,C1,C2 = fix_bezier_kink_points(NEWX,C1,C2,chk)
            #     print("Inconsistent C1 and C2 at point ", chk)


            # NEWX = remove_bezier_nodes_with_too_small_curvature(NEWX, C1, C2, minimal_curvature_radius)
            # print("NEWX=",NEWX)
            # C1, C2 = compute_control_points(NEWX,scale)

            # append to the dictionary            
            if NEWX.shape[0] >= 3:
                all_bezier_points[new_ci] = np.array([NEWX[:-1],C1,C2])
                new_ci += 1

    return all_bezier_points





