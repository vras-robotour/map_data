INIT_WIDTH = 50          # In meters.

OSM_RECTANGLE_MARGIN = 100 # METERS. nafouknuti obdelniku stahovanych dat

DENSITY = 0.5             # In meters. How dense is the graph (distance between neighboring nodes).
DIST_COST_MULTIPLIER = 0
GOAL_BASE_DIST = 30     # In meters. Desired distance between start and goal points in a sub-graph.
                        # If there is an obstacle in the way, the distance is modified accordingly.

RESERVE = 50 # meters

INCREASE = 25           # In meters. How much the graph rectangle grows.
                        
MAX_RANGE = 75         # In meters. Maximum range of the area that will be searched.

MAX_EFFECTIVE_RANGE = 50000 #  CURRENTLY NOT BEING USED.

MAX_DIST_LOSS = 3       # In meters. Any points furter than MAX_DIST_LOSS from the desired trajectory will
                        # be penalized as if they were only at MAX_DIST_LOSS distance.

MAX_COST_PER_METER = 10 # If the INITIAL solution of a sub-graph has a total cost of more than
                        # MAX_COST_PER_METER * meters_between_start_and_goal, increase graph range and try again.
                        # (((If the graph has found a solution, but it is too expensive, we have it try once more
                        # in an extended range. Important for crossing roads.)))

ROAD_LOSS = 0.2        # Penalization of an edge with at least one vertex on a road.
                        # (((Makes the graph search cross roads across footways - those are not considered
                        # as roads - or at least in the shortest manner possible.)))

NO_FOOTWAY_LOSS = 10   # Penalization of an edge at least MAX_DIST_LOSS far and with neither vertex on a footway.
                        # (((The graph search will prefer walking along footways ONCE IT IS FAR from the
                        # desired trajectory anyway.)))

BARRIER_LOSS = 10000
                    
MAX_DETOUR_COUNTER = 3  #  CURRENTLY NOT BEING USED.

OBJECTS_RESERVE = 10    # Margin for searching for objects in the area of a subgraph

if INIT_WIDTH < 1:
    INIT_WIDTH = 1
    print("Parameter INIT_WIDTH set to 1 (minimum).")
