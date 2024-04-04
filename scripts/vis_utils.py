import sys
import matplotlib.pyplot as plt
import numpy as np
import utm

import rospy

from background_map import get_background_image

def plot_background_map(ax, image, coords_data):
    '''
    Plot background map.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot the map on.
    image : PIL.Image
        Image of the map.
    coords_data : map_data.CoordsData
        Coordinates data.
    '''
    min_utm = utm.fromLatLong(coords_data.min_lat - coords_data.y_margin, coords_data.min_long - coords_data.x_margin)
    max_utm = utm.fromLatLong(coords_data.max_lat + coords_data.y_margin, coords_data.max_long + coords_data.x_margin)
    ax.imshow(image, extent = [min_utm.easting, max_utm.easting,\
                               min_utm.northing, max_utm.northing], alpha = 1, zorder = 0)

    ax.set_ylim([min_utm.northing, max_utm.northing])
    ax.set_xlim([min_utm.easting, max_utm.easting])
    rospy.loginfo("Background map plotted")

def plot_path(ax, path):
    '''
    Plot path.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot the path on.
    path : np.array
        Path to plot.
    '''
    ax.scatter(path[:,0], path[:,1], color='#000000', alpha=0.8, s=3, marker='o', zorder = 18000)
    ax.scatter(path[:,0], path[:,1], color='#50C2F6', alpha=0.8, s=2, marker='o', zorder = 20000)
    rospy.loginfo("Path plotted")

def plot_barrier_areas(ax, barrier_areas):
    '''
    Plot barriers in map.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot the barriers on.
    barrier_areas : list
        List of barrier areas.
    '''
    for area in barrier_areas:
        x,y = area.line.exterior.xy
        ax.plot(x, y, c='#BF0009', linewidth=1, zorder = 7)
        
        if area.in_out != "inner":
            ax.fill(x, y, c='#BF0009', alpha=0.4, zorder = 5)
    rospy.loginfo("Barrier areas plotted")

def plot_footways(ax, footways):
    '''
    Plot footways in map.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot the footways on.
    footways : list
        List of footways.
    '''
    for footway in footways:
        x,y = footway.line.exterior.xy
        ax.plot(x, y, c='#FFD700', linewidth=0.5, zorder = 6)
        ax.fill(x, y, c='#FFD700', alpha=0.4, zorder = 4)
    rospy.loginfo("Footways plotted")

def plot_roads(ax, roads):
    '''
    Plot roads in map.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot the roads on.
    roads : list
        List of roads.
    '''
    for road in roads:
        x,y = road.line.exterior.xy
        ax.plot(x, y, c='#000000', linewidth=1, zorder = 6)
        ax.fill(x, y, c='#000000', alpha=0.8, zorder = 5)
    rospy.loginfo("Roads plotted")

def save_map(file_name):
    '''
    Save map to file.

    Parameters:
    -----------
    file_name : str
        Path and name of the file.
    '''
    plt.savefig(file_name)
    rospy.loginfo(f"Map saved to {file_name}")

def plot_map(map_data, save_bgd = False):
    '''
    Plot map from map_data.

    Parameters:
    -----------
    map_data : map_data.MapData
        Map data.
    save_bgd : bool
        Save background image.
    '''
    _, ax = plt.subplots(figsize=(12, 12), dpi=400)

    coords_data = map_data.coords_data
    bgd_map = get_background_image(coords_data.min_long, coords_data.max_long, coords_data.min_lat,\
                                   coords_data.max_lat, coords_data.x_margin, coords_data.y_margin)

    plot_background_map(ax, bgd_map, coords_data)

    if save_bgd:
        path = sys.path[0]
        bgd_map.save(path + "/../data/bgd_map.png")
    
    plot_barrier_areas(ax, np.array(map_data.barriers_list))
    plot_footways(ax, np.array(map_data.footways_list))
    plot_roads(ax, np.array(map_data.roads_list))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

def plot_footways_plan(map_data, save_bgd = False):
    '''
    Plot only footways from map_data.

    Parameters:
    -----------
    map_data : map_data.MapData
        Map data.
    save_bgd : bool
        Save background image.
    '''
    _, ax = plt.subplots(figsize=(12, 12), dpi=400)

    coords_data = map_data.coords_data
    bgd_map = get_background_image(coords_data.min_long, coords_data.max_long, coords_data.min_lat,\
                                   coords_data.max_lat, coords_data.x_margin, coords_data.y_margin)

    plot_background_map(ax, bgd_map, coords_data)

    if save_bgd:
        path = sys.path[0]
        bgd_map.save(path + "/../data/bgd_map.png")

    plot_footways(ax, np.array(map_data.footways_list))
