#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from geodesy import utm
import pickle
import sys

from background_map import get_background_image


def plot_background_map(ax, image, coords_data):
    min_utm = utm.fromLatLong(coords_data.min_lat - coords_data.y_margin, coords_data.min_long - coords_data.x_margin)
    max_utm = utm.fromLatLong(coords_data.max_lat + coords_data.y_margin, coords_data.max_long + coords_data.x_margin)
    ax.imshow(image, extent = [min_utm.easting, max_utm.easting,\
                               min_utm.northing, max_utm.northing], alpha = 1, zorder = 0)

    ax.set_ylim([min_utm.northing, max_utm.northing])
    ax.set_xlim([min_utm.easting, max_utm.easting])

def plot_path(ax, path):
    ax.scatter(path[:,0], path[:,1], color='#000000', alpha=0.8, s=3, marker='o', zorder = 18000)
    ax.scatter(path[:,0], path[:,1], color='#50C2F6', alpha=0.8, s=2, marker='o', zorder = 20000)

def plot_barrier_areas(ax, barrier_areas):
    for area in barrier_areas:
        x,y = area.line.exterior.xy
        ax.plot(x, y, c='#BF0009', linewidth=1, zorder = 7)
        
        if area.in_out != "inner":
            ax.fill(x,y,c='#BF0009', alpha=0.4, zorder = 5)

def plot_map(map_data):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=400)

    coords_data = map_data.coords_data
    bgd_map = get_background_image(coords_data.min_long, coords_data.max_long, coords_data.min_lat,\
                                   coords_data.max_lat, coords_data.x_margin, coords_data.y_margin)

    plot_background_map(ax, bgd_map, coords_data)
    
    plot_barrier_areas(ax, np.array(map_data.barriers_list))


if __name__ == "__main__":
    path = sys.path[0]

    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = "/../data/buchlovice_1.mapdata"

    with open(path + file_name, "rb") as fh:
        map_data = pickle.load(fh)

    plot_map(map_data)
