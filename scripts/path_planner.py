#!/usr/bin/env python

try:
    import cPickle as pickle
except ImportError:
    import pickle

import rospy
import overpy
import shapely.geometry as geometry
from shapely.prepared import prep
from shapely.ops import linemerge
import os
import utm
import numpy as np
from random import random
import time
from points_to_graph_points import points_to_graph_points, points_arr_to_point_line, get_point_line
import graph_tool.all as gt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iter, desc=None, *args, **kwargs):
        if desc is not None:
            rospy.loginfo(desc)
        return iter

import gpxpy
import gpxpy.gpx

from graph_search_params import *

OSM_URL = "https://www.openstreetmap.org/api/0.6/way/{}/relations"
#TERRAIN_TAGS = ['landuse','natural','public_transport','service']
#TERRAIN_VALUES = ['park']
#TERRAIN_OR_BARRIER_TAGS = ['leisure']
#BARRIER_TAGS = ['waterway','barrier','man_made','building','amenity','sport']
#BARRIER_TAGS = csv_to_dict('barrier_tags.csv')
#NOT_BARRIER_VALUES = ['underground','underwater','overhead']    # location tag
#NOT_BARRIER_AREA_VALUES = ['parking']
#NOT_BARRIER_TAGS = csv_to_dict('not_barrier_tags.csv')
#ROAD_TAGS = ['highway','footway','surface']
FOOTWAY_VALUES = ['living_street','pedestrian','footway','bridleway','corridor','track','steps', 'cycleway', 'path'] # living_street,pedestrian,track,crossing can be accessed by cars
#TAGS_KEY_ONLY = ['building']
#OBSTACLE_TAGS = ['historic','amenity','natural','tourism','information']
#NOT_OBSTACLE_TAGS = ['addr:country','addr:street']
#OBSTACLE_TAGS = csv_to_dict('obstacle_tags.csv')
#NOT_OBSTACLE_TAGS = csv_to_dict('not_obstacle_tags.csv')

MAX_ROAD_DIST = 10
MAX_FOOTWAY_DIST = 5
MAX_BARRIER_DIST = 10
MAX_OBSTACLE_DIST = 10

MAX_REL_MEMBERS = 1000
OBSTACLE_RADIUS = 2

    
class Way():
    def __init__(self,id=-1,is_area=False,nodes=[],tags=None,line=None,in_out=""):
        self.id = id
        self.is_area = is_area
        self.nodes = nodes
        self.tags = tags
        self.line = line
        self.in_out = in_out

        self.pcd_points = None
    
    def is_road(self):
        if self.tags.get('highway', None) and not self.tags.get('highway', None) in FOOTWAY_VALUES:
            return True

    def is_footway(self):
        if self.tags.get('highway', None) and self.tags.get('highway', None) in FOOTWAY_VALUES:
            return True
        
    def is_terrain(self):
        #if any(tag in TERRAIN_TAGS+TERRAIN_OR_BARRIER_TAGS for tag in self.tags) and not any(tag in BARRIER_TAGS for tag in self.tags):
        return True
    
    def is_barrier(self, yes_tags, not_tags, anti_tags):
        if any(key in yes_tags and (self.tags[key] in yes_tags[key] or ('*' in yes_tags[key] and not self.tags[key] in not_tags.get(key,[]))) for key in self.tags) and not any(key in anti_tags and (self.tags[key] in anti_tags[key]) for key in self.tags):
            return True

    def to_pcd_points(self, density=2, filled=True):
        # https://stackoverflow.com/questions/44399749/get-all-lattice-points-lying-inside-a-shapely-polygon
        
        if self.pcd_points is None:
            if filled:
                xmin, ymin, xmax, ymax = self.line.bounds
                x = np.arange(np.floor(xmin * density) / density, np.ceil(xmax * density) / density, 1 / density) 
                y = np.arange(np.floor(ymin * density) / density, np.ceil(ymax * density) / density, 1 / density)
                xv,yv = np.meshgrid(x,y)
                xv = xv.ravel()
                yv = yv.ravel()
                points = geometry.MultiPoint(np.array([xv,yv]).T).geoms
        
                points = self.mask_points(points,self.line)
                self.pcd_points = list(points)
                self.pcd_points = np.array(list(geometry.LineString(self.pcd_points).xy)).T
            else:
                points = self.line.exterior.coords
                pcd_points = np.array([]).reshape((0,2))
                
                for i in range(len(points)):
                    if i+1 <= len(points)-1:
                        p1 = geometry.Point(points[i])
                        p2 = geometry.Point(points[i+1])

                    else:
                        p1 = geometry.Point(points[i])
                        p2 = geometry.Point(points[0])

                    _,line,_ = get_point_line(p1,p2,0.5)
                    pcd_points = np.concatenate([pcd_points,line])
                self.pcd_points = pcd_points
        
        return self.pcd_points
    
    def mask_points(self, points, polygon):
        polygon = prep(polygon)

        contains = lambda p: polygon.contains(p)

        ret = filter(contains, points)

        return ret


class PathAnalysis:
    def __init__(self, coords_file, current_robot_position=None, use_osm=True, use_solitary_nodes=True, flip=False):
        
        self.robot_position_first_point = False
        self.api = overpy.Overpass(url="https://overpass.kumi.systems/api/interpreter")

        self.use_osm = use_osm
        self.use_solitary_nodes = use_solitary_nodes

        self.flip = flip

        # Gpx file to numpy array.
        self.coords_file = coords_file
        gpx_f = open(coords_file, 'r')
        gpx_object = gpxpy.parse(gpx_f)
        self.waypoints = np.array([[point.latitude,point.longitude] for point in gpx_object.waypoints])
        self.waypoints, self.zone_number, self.zone_letter = self.waypoints_to_utm()

        if self.flip:
            self.waypoints = np.flip(self.waypoints, 0)

        if current_robot_position is not None:
            self.robot_position_first_point = True
            self.waypoints = np.concatenate([current_robot_position,self.waypoints])
        
        self.max_x = np.max(self.waypoints[:,0]) + RESERVE
        self.min_x = np.min(self.waypoints[:,0]) - RESERVE
        self.max_y = np.max(self.waypoints[:,1]) + RESERVE
        self.min_y = np.min(self.waypoints[:,1]) - RESERVE

        self.max_lat = utm.to_latlon(self.max_x + OSM_RECTANGLE_MARGIN, self.max_y + OSM_RECTANGLE_MARGIN, self.zone_number, self.zone_letter)[0]
        self.max_long = utm.to_latlon(self.max_x + OSM_RECTANGLE_MARGIN, self.max_y + OSM_RECTANGLE_MARGIN, self.zone_number, self.zone_letter)[1]
        self.min_lat = utm.to_latlon(self.min_x - OSM_RECTANGLE_MARGIN, self.min_y - OSM_RECTANGLE_MARGIN, self.zone_number, self.zone_letter)[0]
        self.min_long = utm.to_latlon(self.min_x - OSM_RECTANGLE_MARGIN, self.min_y - OSM_RECTANGLE_MARGIN, self.zone_number, self.zone_letter)[1]

        self.points = list(map(geometry.Point, zip(self.waypoints[:,0], self.waypoints[:,1])))

        self.points_information = []
        self.way_node_ids = set() 
        
        self.roads = set()
        self.footways = set()
        self.barriers = set()

        self.roads_list = []
        self.footways_list = []
        self.barriers_list = []

        self.road_polygons = []

        self.ways = dict()

        self.BARRIER_TAGS = self.csv_to_dict(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'parameters/barrier_tags.csv'))
        self.NOT_BARRIER_TAGS = self.csv_to_dict(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'parameters/not_barrier_tags.csv'))
        self.ANTI_BARRIER_TAGS = self.csv_to_dict(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'parameters/anti_barrier_tags.csv'))

        self.OBSTACLE_TAGS = self.csv_to_dict(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'parameters/obstacle_tags.csv'))
        self.NOT_OBSTACLE_TAGS = self.csv_to_dict(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'parameters/not_obstacle_tags.csv'))

        self.path = []
    
    def csv_to_dict(self,f):
        arr = np.genfromtxt(f, dtype=str, delimiter=',')
        dic = dict()
        for row in arr:
            if row[0] in dic:
                dic[row[0]].append(row[1])
            else:
                dic[row[0]] = [row[1]]
        return dic
    
    def waypoints_to_utm(self):
        utm_arr = utm.from_latlon(self.waypoints[:,0],self.waypoints[:,1])
        utm_coords = np.concatenate((utm_arr[0].reshape(-1,1), utm_arr[1].reshape(-1,1)),axis=1)
        zone_number = utm_arr[2]
        zone_letter = utm_arr[3]
        return utm_coords, zone_number, zone_letter

    def get_way_query(self):
        query = """(way({}, {}, {}, {});
                    >;
                    );
                    out;""".format(self.min_lat,self.min_long,self.max_lat,self.max_long)

        return query
    
    def get_rel_query(self):
        query = """(way({}, {}, {}, {});
                    <;
                    );
                    out;""".format(self.min_lat,self.min_long,self.max_lat,self.max_long)

        return query
    
    def get_node_query(self):
        query = """(node({}, {}, {}, {});
                    );
                    out;""".format(self.min_lat,self.min_long,self.max_lat,self.max_long)

        return query

    def parse_ways(self):
        """ 1. Fill self.ways, a dictionary of id:way pairs, from all the ways from the query."""

        for way in tqdm(self.osm_ways_data.ways, desc="Parse ways"):
            way_to_store = Way()
            coords = []
            is_area = False

            # Convert WGS -> UTM.
            lats = np.array([float(node.lat) for node in way.nodes])
            lons = np.array([float(node.lon) for node in way.nodes])
            utm_coords = utm.from_latlon(lats,lons)
            coords = list(zip(utm_coords[0], utm_coords[1]))

            # Keep track of IDs of each node, so that in parse_nodes we can distinguish them from solitary nodes.
            ids = [node.id for node in way.nodes]
            if self.way_node_ids is None:
                self.way_node_ids = dict()
            self.way_node_ids.update(ids)              
            
            # Distinguish areas and non-areas (we use a single class for both cases).
            if coords[0] == coords[-1]:
                is_area = True
            
            way_to_store.id = way.id
            way_to_store.is_area = is_area
            way_to_store.nodes = way.nodes
            way_to_store.tags = way.tags

            if way_to_store.tags is None:
                way_to_store.tags = dict()

            if is_area:
                way_to_store.line = geometry.Polygon(coords)
            else:
                way_to_store.line = geometry.LineString(coords)
            
            self.ways[way.id] = way_to_store

    def combine_ways(self,ids):
        ways = []
        for id in ids:
            ways.append(self.ways[id])
        i = 0
        while i < len(ways):
            j = 0
            while j < len(ways):
                if i != j:
                    if (ways[i].nodes[0].id == ways[j].nodes[0].id) and (not ways[i].is_area and not ways[j].is_area):
                        ways[i].nodes.reverse()
                    elif (ways[i].nodes[-1].id == ways[j].nodes[-1].id) and (not ways[i].is_area and not ways[j].is_area):
                        ways[j].nodes.reverse()

                    if ways[i].nodes[-1].id == ways[j].nodes[0].id and (not ways[i].is_area and not ways[j].is_area):
                        
                        combined_line = linemerge([ways[i].line, ways[j].line])

                        new_way = Way()
                        new_way.id = int(-10**15*random())
                        while new_way.id in self.ways.keys():
                            new_way.id = int(-10**15*random())
                        # tady zlobi ten update
                        new_way.nodes = ways[i].nodes + ways[j].nodes[1:] 
                        
                        if ways[i].tags is None:
                            ways[i].tags = dict()
                        if ways[j].tags is None:
                            ways[j].tags = dict()
                        new_way.tags = ways[i].tags.update(ways[j].tags)
                        new_way.line = combined_line
  
                        if new_way.nodes[0].id == new_way.nodes[-1].id:
                            new_way.is_area = True
                            new_way.line = geometry.Polygon(new_way.line.coords)
                        self.ways[new_way.id] = new_way
                        ways[j] = new_way
                        ids[j] = new_way.id
                        ids.pop(i)
                        ways.pop(i)
                        i -= 1
                        j -= 1
                        break
                j += 1
            i += 1
        
        return ids

    def parse_rels(self):
        """ 2. Needs self.ways DICTIONARY (key is id) with a self.is_area parameter, which is obtained from parse_ways.
            Use relations to alter ways - combine neighbor ways, add tags...
        """
        for rel in tqdm(self.osm_rels_data.relations, desc="Parse rels"):
            if len(rel.members) <= MAX_REL_MEMBERS:   # A lot of members is very likely some relation we are not interested in.
                inner_ids = []
                outer_ids = []
                keys = self.ways.keys()

                # Separate inner and outer ways of relation.
                for member in rel.members:
                    if member._type_value == "way":
                        if int(member.ref) in keys:
                            if member.role == "outer":
                                outer_ids.append(int(member.ref))
                            else:
                                inner_ids.append(int(member.ref))

                # If two ways are "connected" (they share a node), combine them into one.
                outer_ids = self.combine_ways(outer_ids)

                for id in outer_ids:
                    way = self.ways[id]

                    way.in_out = "outer"
                    
                    if way.tags is None:
                        way.tags = dict()
                    if rel.tags is None:
                        rel.tags = dict()
                    way.tags.update(rel.tags)
                    self.ways[id] = way

                for id in inner_ids:
                    way = self.ways[id]
                    way.in_out = "inner"
                    self.ways[id] = way

    def parse_nodes(self):
        """ Convert solitary nodes (not part of a way) to barrier areas. """

        for node in tqdm(self.osm_nodes_data.nodes, desc="Parse nodes"):
            if not node.id in self.way_node_ids:
                # Check if node is a obstacle.
                if any(key in self.OBSTACLE_TAGS and (node.tags[key] in self.OBSTACLE_TAGS[key] or ('*' in self.OBSTACLE_TAGS[key] and not node.tags[key]  in self.NOT_OBSTACLE_TAGS.get(key,[]))) for key in node.tags):
                    obstacle = Way()
                    obstacle.id = node.id
                    obstacle.is_area = True
                    obstacle.tags = node.tags

                    coords = utm.from_latlon(float(node.lat),float(node.lon))
                    point = geometry.Point([coords[0], coords[1]])
                    polygon = self.point_to_polygon(point,OBSTACLE_RADIUS)
                    obstacle.line = polygon

                    self.barriers.add(obstacle)

    def point_to_polygon(self, point, r):
        """ Convert a node (= a point) to a circle area, with a given radius."""

        polygon = point.buffer(r)
        return polygon

    def line_to_polygon(self, way, width=4):
        """ The width of the buffer should depend on,
            the type of way (river x fence, highway x path)... """
        polygon = way.line.buffer(width/2)
        way.line = polygon
        way.is_area = True
        return way

    def separate_ways(self):
        """ Separate ways (dict) into roads, footways and barriers (lists). """

        for way in tqdm(self.ways.values(), desc="Separate ways"):
            if way.is_road():
                way = self.line_to_polygon(way,width=7)
                self.roads.add(way)
            
            elif way.is_footway():
                way = self.line_to_polygon(way,width=3)
                self.footways.add(way)

            elif way.is_barrier(self.BARRIER_TAGS, self.NOT_BARRIER_TAGS, self.ANTI_BARRIER_TAGS):
                if not way.is_area:
                    way = self.line_to_polygon(way,width=2)
                self.barriers.add(way)

    def is_point_valid(self, point, objects):
        valid = True
        for way in objects['barriers']:
            if way.line.contains(point):
                valid = False
                break
        return valid

    def get_contain_mask(self, points, areas):
        if len(areas) >= 1:
            multi_polygon = geometry.MultiPolygon([area.line for area in areas] if type(areas[0]) is type(Way()) else areas)
            multi_polygon = multi_polygon.buffer(0)
            multi_polygon = prep(multi_polygon)
            contains = lambda p: multi_polygon.contains(p)
        else:
            contains = lambda p: False
        mask = np.array(list(map(contains, points)))
        return mask
    
    def mask_points(self, points, other_array, areas):
        multi_polygon = geometry.MultiPolygon((area.line for area in areas))
        multi_polygon = multi_polygon.buffer(0)
        multi_polygon = prep(multi_polygon)

        does_not_contain = lambda p: not multi_polygon.contains(p[0])

        arr = zip(points,other_array)
        ret = filter(does_not_contain, arr)

        ret = list(zip(*ret))

        return ret[0],ret[1]
    
    def get_barriers_along_line(self,line):
        areas = []
        for area in self.barriers_list:
            if line.intersects(area.line):
                areas.append(area)
        return areas

    def get_reduced_objects(self, min_x, max_x, min_y, max_y, reserve=RESERVE):
        reduced_objects = {}

        max_x += reserve
        max_y += reserve
        min_x -= reserve
        min_y -= reserve

        check_polygon = geometry.Polygon(([min_x, min_y],
                                            [max_x, min_y],
                                            [max_x, max_y],
                                            [min_x, max_y]))
        check_polygon = prep(check_polygon)

        check_roads = self.roads_list
        check_footways = self.footways_list
        check_barriers = self.barriers_list
        check_road_polygons = [[self.road_polygons[i], 'road_polygons{}'.format(i)] for i in range(len(self.road_polygons))]

        check_us = [[check_roads,'roads'],[check_footways,'footways'],[check_barriers,'barriers']]

        check_func = lambda ob: check_polygon.intersects(ob.line)

        for check_me in check_us:
            reduced_objects[check_me[1]] = list(filter(check_func, check_me[0]))

        check_func_poly = lambda ob: check_polygon.intersects(ob)

        for check_poly in check_road_polygons:
            reduced_objects[check_poly[1]] = list(filter(check_func_poly, check_poly[0]))

        return reduced_objects

    def get_min_index(self, indices, threshold):
        ind = 1000000
        min_indices = np.where(indices > threshold)
        if min_indices:
            ind = min_indices[0][0]
            ind = indices[ind]
        return ind
    
    def generate_goal_points(self, points_line, density, dist, barriers, original_waypoint_indices):
        """ Choose goal as a point at distance (on the line) dist from start.
            If there is an obstacle between the two points, throw away the goal and instead
            take the point before and after the first obstacle as new goals. Continue form the latter point."""
        goal_points = []
        
        points_validity = [False] * len(points_line)
        for i,point in enumerate(points_line):
            points_validity[i] = self.is_point_valid(point, {'barriers':barriers})
        points_validity = np.array(points_validity)

        interval = round(dist/density)

        start_index = np.where(points_validity==True)[0][0]         # First start point is the first valid point
        goal_points.append(points_line[int(start_index)])

        goal_index = min(start_index+interval, len(points_line), self.get_min_index(original_waypoint_indices, start_index))

        while True:
            start_index = int(start_index)
            goal_index = int(goal_index)
            if goal_index < len(points_line)-1:
                pass
            else:
                goal_index = len(points_line)-1

            current_validity = points_validity[int(start_index):int(goal_index+1)]

            if np.sum(current_validity) == len(current_validity):
                pass

            elif np.sum(current_validity) > 1:
                if points_validity[start_index+1] == True:
                    goal_index = np.where(current_validity==False)[0][0] - 1 + start_index  # point before obstacle
                else:
                    goal_index = np.where(current_validity==True)[0][1] + start_index       # point after obstacle
                    
            else:
                try:
                    goal_index = np.where(points_validity[start_index+1:]==True)[0][0]
                except:
                    break
                if goal_index:      # point after large obstacle  
                    goal_index = goal_index + start_index +1
                else:               # large obstacle until the end (no more goal points)        
                    break
            
            goal_points.append(points_line[int(goal_index)])

            for area in self.barriers:
                if area.line.contains(points_line[int(goal_index)]):
                    rospy.loginfo("point bad")

            if goal_index == len(points_line)-1:
                break

            start_index = goal_index
            goal_index = min(start_index+interval, len(points_line)-1, self.get_min_index(original_waypoint_indices, start_index))

        inds_to_pop = []

        for i in range(len(goal_points)-1):
            if goal_points[i] == goal_points[i+1]:
                inds_to_pop.append(i)

        inds_to_pop.sort(reverse=True)
        for i in inds_to_pop:
            goal_points.pop(i)
        return goal_points

    def graph_search_path(self):
        """ Using graph search prepare the path for the robot from the given waypoints. """

        rospy.loginfo("Running graph search.")

        graph_counter = 0

        points_line, original_waypoint_indices = points_arr_to_point_line(self.points,density=DENSITY)

        line = geometry.LineString(points_line)
        barriers_along_point_line = self.get_barriers_along_line(line)
                
        goal_points = self.generate_goal_points(points_line,
                                                density = DENSITY,
                                                dist = GOAL_BASE_DIST, 
                                                barriers = barriers_along_point_line, 
                                                original_waypoint_indices = np.array(original_waypoint_indices))
        num_sub_paths = len(goal_points) -1
        
        start_point = goal_points.pop(0)    # Is updated at the end of each cycle as the previous goal point.

        path = []

        while goal_points:              # Main cycle.
            goal_point = goal_points.pop(0)

            objects_in_area = None
            increase_graph = 0
            density = DENSITY
            keep_previous_start = False

            while True:
                graph_range =  INIT_WIDTH + 2*increase_graph
                graph_points, points_line, dist_from_line = points_to_graph_points(start_point, goal_point, density=density, width=graph_range)

                objects_in_area = self.get_reduced_objects(graph_points.bounds[0],\
                                                            graph_points.bounds[1],\
                                                            graph_points.bounds[2],\
                                                            graph_points.bounds[3],\
                                                            reserve=graph_range)
 
                graph_points,dist_from_line = self.mask_points(graph_points.geoms, dist_from_line, objects_in_area['barriers'])
                dist_from_line = np.array(dist_from_line)

                # TODO: Find a way to speed this up, longest time is taken by multi_polygon.buffer(0) in get_contain_mask
                road_points_mask = self.get_contain_mask(graph_points, objects_in_area['roads'])
                
                footway_points_mask = self.get_contain_mask(graph_points, objects_in_area['footways'])

                out_of_max_dist_mask = dist_from_line >= MAX_DIST_LOSS
                out_of_max_dist_mask = np.squeeze(out_of_max_dist_mask)

                graph_points = np.array(list(geometry.LineString(graph_points).xy)).T # faster than list compr. or MultiPoint

                start_index_graph = np.where(graph_points==[start_point.x,start_point.y])[0][0]
                goal_index_graph = np.where(graph_points==[goal_point.x,goal_point.y])[0][0]

                graph, _ = gt.geometric_graph(graph_points, 1.5*density)

                edges = graph.get_edges()
                graph.clear_edges()

                eprop_cost = graph.new_edge_property("double")

                edge_points_1 = graph_points[edges[:,0]]
                edge_points_2 = graph_points[edges[:,1]]

                road_points = []
                not_footway_points = (~footway_points_mask[edges[:,0]] + ~footway_points_mask[edges[:,1]])
                road_points = (road_points_mask[edges[:,0]] + road_points_mask[edges[:,1]])

                dist_cost = np.divide(dist_from_line[edges[:,0]] + dist_from_line[edges[:,1]], 2)
                dist_cost = np.minimum(dist_cost, MAX_DIST_LOSS)

                no_footways = (out_of_max_dist_mask[edges[:,0]] * out_of_max_dist_mask[edges[:,1]]) * (~footway_points_mask[edges[:,0]] + ~footway_points_mask[edges[:,1]])

                costs = self.get_costs(edge_points_1, edge_points_2, road_points, dist_cost, ROAD_LOSS, no_footways, NO_FOOTWAY_LOSS)

                edges = np.concatenate((edges,costs),axis=1)

                graph.add_edge_list(edges, eprops=[eprop_cost])
                               
                weights = eprop_cost
                
                shortest_path_vertices,shortest_path_edges = gt.shortest_path(graph, \
                                                graph.vertex(start_index_graph), \
                                                graph.vertex(goal_index_graph), \
                                                weights=weights)

                shortest_path_cost = self.get_path_cost(shortest_path_edges,weights)
                
                color = graph.new_vertex_property("vector<double>")
                
                color[graph.vertex(start_index_graph)] = (0,1,0,1)
                color[graph.vertex(goal_index_graph)] = (0,1,0,1)

                if shortest_path_vertices:
                    if (shortest_path_cost < MAX_COST_PER_METER * start_point.distance(goal_point)) or increase_graph > 0: #or graph_range >= MAX_RANGE:
                        path += [graph_points[graph.vertex_index[v]] for v in shortest_path_vertices]
                        graph_counter += 1
                        rospy.loginfo("Finished {}/{} sub-graphs.".format(graph_counter,num_sub_paths))

                        break
                    else:
                        increase_graph += INCREASE
                else:
                    if graph_range >= MAX_RANGE:
                        rospy.loginfo("Sub-graph did not find solution in range {} m.".format(graph_range))
                        graph_counter += 1
                        break
                    else:
                        increase_graph += INCREASE

            if not keep_previous_start:
                keep_previous_start = False
                start_point = goal_point

        self.path = np.array(path)

    def get_path_cost(self,edges,costs):
        cost = 0
        for edge in edges:
            cost += costs[edge]
        return cost
    
    def get_points_costs(self,road_points_mask,road_points_mask_bool,footway_points_mask,barrier_points_mask,dist_from_line,out_of_max_dist_mask,road_loss,no_footway_loss,barrier_loss):
        barrier_points = (barrier_points_mask * ~road_points_mask_bool * ~footway_points_mask).reshape(-1,1)
        no_footway_points = (out_of_max_dist_mask * ~footway_points_mask).reshape(-1,1)
        
        cost = dist_from_line + road_loss*road_points_mask + no_footway_loss*no_footway_points + barrier_loss*barrier_points
        return [cost,dist_from_line,no_footway_loss*no_footway_points,barrier_loss*barrier_points]
    
    def get_costs(self, p1, p2, roads, dist_cost, road_loss, no_footways, no_footway_loss, use_osm=True):
        vertices_dist_cost = np.sqrt(np.sum(np.square(p1-p2),axis=1))
        if use_osm:
            return np.reshape(vertices_dist_cost *(
                                roads * road_loss + 
                                no_footway_loss * no_footways), (len(p1),1)) + dist_cost
        else:
            return np.reshape(vertices_dist_cost, (len(p1),1)) + dist_cost

    def sets_to_lists(self):
        self.roads_list     = list(self.roads)
        self.footways_list  = list(self.footways)
        self.barriers_list  = list(self.barriers)

    def run_parse(self):
        """ Parse OSM data. """
        rospy.loginfo("Running analysis.")
        self.parse_ways()
        self.parse_rels()

        if self.use_solitary_nodes:
            self.parse_nodes()

        self.separate_ways()

        self.sets_to_lists()
    
    def run_full_search(self, gpx_fn):
        """ Run to get the complete graph search. """
        start_t = time.time()
        self.graph_search_path()
        rospy.loginfo("Path took: %.5f" % (time.time()-start_t))
        self.save_path_as_gpx(gpx_fn)
    
    def run_queries(self):
        """ Obtain data from OSM through their API. """
        break_time = 5
        tries = 1
        while tries < 4 and not rospy.is_shutdown():
            rospy.loginfo("Running 1/3 OSM query.")
            try:
                way_query = self.get_way_query()
                osm_ways_data = self.api.query(way_query)
                self.way_query = way_query
                self.osm_ways_data = make_overpy_result_picklable(osm_ways_data)
                break
            except Exception as e:
                rospy.loginfo(e)
                rospy.loginfo("--------------\nQuery failed.\nRerunning the query after {} s.".format(break_time))
                time.sleep(break_time)
                tries += 1

        tries = 1
        while tries < 4 and not rospy.is_shutdown():
            rospy.loginfo("Running 2/3 OSM query.")
            try:
                rel_query = self.get_rel_query()
                osm_rels_data = self.api.query(rel_query)
                self.rel_query = rel_query
                self.osm_rels_data = make_overpy_result_picklable(osm_rels_data)
                break
            except Exception as e:
                rospy.loginfo(e)
                rospy.loginfo("--------------\nQuery failed.\nRerunning the query after {} s.".format(break_time))
                time.sleep(break_time)
                tries += 1

        tries = 1
        if self.use_solitary_nodes:
            while tries < 4 and not rospy.is_shutdown():
                rospy.loginfo("Running 3/3 OSM query.")
                try:  
                    node_query = self.get_node_query()  # Query sometimes times out...
                    osm_nodes_data = self.api.query(node_query)
                    self.node_query = node_query
                    self.osm_nodes_data = make_overpy_result_picklable(osm_nodes_data)
                    break
                except Exception as e:
                    rospy.loginfo(e)
                    rospy.loginfo("--------------\nQuery failed.\nRerunning the query after {} s.".format(break_time))
                    time.sleep(break_time)
                    tries += 1

        rospy.loginfo("Queries finished.")

    def run_standalone(self, gpx_fn):
        self.run_queries()
        self.run_parse()
        self.run_full_search(gpx_fn)
    
    def write_to_file(self,fn):
        with open(fn,'w+') as f:
            for point in self.points_information:
                f.write(point.__str__())
                f.write("\n")

    def save_path_as_gpx(self, fn=os.path.join(os.path.dirname(os.path.dirname(__file__)), "gpx/path.gpx")):
        gpx = gpxpy.gpx.GPX()

        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)

        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        for point in self.path:
            point = utm.to_latlon(point[0], point[1], 33, 'T')
            gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(point[0], point[1], elevation=0))

        xml = gpx.to_xml()
        with open(fn, "w") as f:
            f.write(xml)
            f.close()
        
        rospy.loginfo("Path saved to {}.".format(fn))

    def save_to_pickle(self, filename=None):
        fn = self.coords_file if filename is None else filename
        with open(fn[:-4]+'.osm_planner', 'wb') as handle:
            pickle.dump(self, handle, protocol=2)


# Useful links:
# https://gis.stackexchange.com/questions/259422/how-to-get-a-multipolygon-object-from-overpass-ql


# fix for pickling overpy.Result on Melodic
def make_overpy_picklable(data):
    if hasattr(data, '_attribute_modifiers'):
        data._attribute_modifiers = None
    return data

def make_overpy_result_picklable(sample):
    sample = make_overpy_picklable(sample)
    for i in range(len(sample.nodes)):
        sample.nodes[i] = make_overpy_picklable(sample.nodes[i])
    for i in range(len(sample.ways)):
        sample.ways[i] = make_overpy_picklable(sample.ways[i])
    for i in range(len(sample.areas)):
        sample.areas[i] = make_overpy_picklable(sample.areas[i])
    for i in range(len(sample.relations)):
        sample.relations[i] = make_overpy_picklable(sample.relations[i])
    return sample
