# map_data
ROS tools to work with OSM data.

## Overview
This is a ROS package that parses a .gpx file with GPS coordinats into a python class, which is later serialized and saved as a .mapdata file.

The class uses OSM as a source of data and parses it into a few cathegories:
- barriers - obstacles that we assume to be untraversable
- footways - paths designed for humans
- roads - areas designed for vehicles

We also provide a ROS node that uses the parsed map data and publishes a point cloud of footways. This may be used as one of the inputs for path-planning nodes.

There are two .gpx files in the `./data/` directory that can be used to test the package.

The package was developed and tested on Ubuntu 20.04 with ROS Noetic. However, it should be fully compatible with ROS Melodic.

## How to use
### Parsing a .gpx file and creating a .mapdata file
The script `create_mapdata` is used to create a .mapdata file from a .gpx file or to parse a .mapdata file which was created earlier.
There are two flags that can be used:
- `-d` - this signals that no prior .mapdata file exists and the script should parse a .gpx file and download data from OSM
- `-f` - this flags allows the user to either specify a .gpx file to parse or a .mapdata file to read from, depending on the `-d` flag, the filename should be a second argument after the flag

In both cases the script will create a .mapdata file in the `./data/` directory and taking the .gpx file there as well. The name of the .mapdata file will be the same as the .gpx file.

The script can be run either with the `rosrun` command or by running it as an executable.

Using `rosrun` and creating a .mapdata file from a .gpx file:
```bash
rosrun map_data create_mapdata -d -f coords.gpx
```
This will create a `coords.mapdata` file in the `./data/` directory, downloading and parsing data from OSM.

Running as an executable and parsing a .mapdata file:
```bash
./scripts/create_mapdata -f coords.mapdata
```
This will load the `coords.mapdata` file from the `./data/` directory and parse it, saving it back to the same file.

### Visualizing the parsed data
The script `visualize_mapdata` is used to visualize the parsed data from a .mapdata file. It will show two images, first one with all parsed data (barriers, footways and roads), while the second one will only containt the footways. There are multiple flags that can be used:
- `-f` - this flag is used to specify the .mapdata file to read from, it should be followed by the filename, and the file should be in the `./data/` directory
- `-sm` - this flag is used to specify if the first image should be saved, it may be followed by a `-if` flag and a filename to save the image to, if the `-if` flag is not used the image will be saved as `map.png`
- `-sb` - this flag is used to specify if the background should be saved, it may be followed by a `-bf` flag and a filename to save the image to, if the `-if` flag is not used the image will be saved as `bgd_map.png`

All saved images will be located in the `./data/` directory.

The script can be run either with the `rosrun` command or by running it as an executable.

Using `rosrun` and visualizing the parsed data:
```bash
rosrun map_data visualize_mapdata -f coords.mapdata
```

Running as an executable and visualizing the parsed data and saving the images:
```bash
./scripts/visualize_mapdata -f coords.mapdata -sm -sb
```

### Publishing a point cloud of footways
The script `osm_cloud` is used to publish a point cloud of footways. The point cloud is published as a `sensor_msgs/PointCloud2` message on the `/osm_cloud` topic. The point cloud is published on the `grid` topic.

The script takes several parameters from the ROS parameter server:
- `~utm_frame` - the UTM frame of the map data, default is `utm`
- `~local_frame` - the local frame of the robot, default is `map`
- `~utm_to_local` - the transformation from UTM to local frame, if set as an empty list the script will lookup for the transformation, default is `None`
- `~mapdata_file` - absolute path for the .mapdata file to read from, default is `None`
- `~gpx_file` - absolute path for the .gpx file to read from, the script will create and parse mapdata, if the .mapdata file is not provided, default is `None`
- `~save_mapdata` - if set to `true` the script will save the parsed mapdata from the .gpx file, if the .mapdata file is not provided, default is `false`
- `~max_path_dist` - the maximum distance a point in the grid can be from a footway, default is `1.0`
- `~neighbor_cost` - in what way to calculate the cost - linear or quadratic or zero, default is `linear`
- `~grid_res` - the resolution of the grid, default is `0.25`
- `~grid_max` - the maximal value of the grid, default is `[250, 250]`
- `~grid_min` - the minimal value of the grid, default is `[-250, -250]`

The script can be run either with the `rosrun` command or by running it as an executable. However we would recommend launching it with a provided launch file. The launch file will set the parameters for the script and launch the node. With the launch file you can also change the topic on which the point cloud is published. The launch file is located in the `./launch/` directory. The launch file has three arguments:
- `mapdata_file` - the absolute path for the .mapdata file to read from
- `gpx_file` - the absolute path for the .gpx file to read from
- `grid_topic` - the topic on which the point cloud is published

## License

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/vras-robotour/map_data/blob/master/LICENSE)
