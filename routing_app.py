
import streamlit as st
import requests
from typing import Tuple, List
import json
import pandas as pd
import pydeck as pdk
# Constraint Programming
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from pydantic.dataclasses import dataclass
from typing import List

MAPQUEST_KEY = "TZt5RVTQrGVrcN4GW18QCGA5plt6h6No"


#------------------------------------------Variables------------------------------------------------------
Coordinate = Tuple[float,float]  # latitude, longitude
Path = List[Coordinate]


#------------------------------------------Mapping Tools------------------------------------------------------
class GeoCoder:

    def encode(address: str) -> Coordinate:
        pass


class MapQuestGeoCoder(GeoCoder):

    @st.cache
    def encode(address: str) -> Coordinate:
        r = requests.get(f'http://www.mapquestapi.com/geocoding/v1/address?key={MAPQUEST_KEY}&location={address}')
        result = json.loads(r.content)

        assert result['results'][0]['locations'][0]['adminArea1'] == 'BE', "Location Outside Belgium"
        return tuple(result['results'][0]['locations'][0]['displayLatLng'].values())




class RouteMatrix:
    #https://developer.mapquest.com/documentation/directions-api/route-matrix/post/

    @st.cache
    def get_distance_matrix(self, locations: List[str]):
        
        data = json.dumps({
                "locations": locations,
                "options": {"allToAll": True}
                })

        r = requests.post(f'http://www.mapquestapi.com/directions/v2/routematrix?key={MAPQUEST_KEY}', data = data)

        result = json.loads(r.content)
        distances = result['distance']
        times = result['time']
        res_locations = result['locations']
        return distances

class Directions:
    #https://developer.mapquest.com/documentation/directions-api/directions/post/

    @st.cache
    def get_directions(self, locations: List[str]) -> Path:
        
        data = json.dumps({
                        "locations": locations,
                        "options": {
                        }
                    })

        r = requests.post(f'http://www.mapquestapi.com/directions/v2/route?key={MAPQUEST_KEY}', data=data)

        result = json.loads(r.content)
        path = [list(m['startPoint'].values()) for l in result['route']['legs'] for m in l['maneuvers']]
        return path
#----------------------------------------------------Solvers----------------------------------------------------------------------------

@dataclass
class Problem:
    distance_matrix: List[List[int]]
    num_locations: int
    num_vehicles: int
    depot: int

@dataclass
class Solution:
    vehicle_to_locations: List[List[int]]
    distances: List[int]

class ORToolsSolver:
#https://developers.google.com/optimization/routing/vrp

    @st.cache
    def solve(self , problem: Problem) -> Solution:
        manager = pywrapcp.RoutingIndexManager(problem.num_locations, problem.num_vehicles, problem.depot)
        routing = pywrapcp.RoutingModel(manager)


        # -----------------------Cost Of Transit-----------------------
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return problem.distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


        # -----------------------Maximum Distance Constraint-----------------------
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            3000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)


        # -----------------------Search Parameters-----------------------
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # -----------------------Solve the problem-----------------------
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            formated_solution = self.format_solution(problem, manager, routing, solution)
        else:
            print('No solution found !')

        return formated_solution if formated_solution else []

    def format_solution(self, problem, manager, routing, solution):

        vehicle_to_locations = []
        distances = []
        print(f'Objective: {solution.ObjectiveValue()}')
        max_route_distance = 0
        for vehicle_id in range(problem.num_vehicles):
            index = routing.Start(vehicle_id)
            route = []

            route_distance = 0
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)

            distances.append(route_distance)
            route.append(manager.IndexToNode(index))
            vehicle_to_locations.append(route)

        return Solution(vehicle_to_locations, distances)

#------------------------------------------Display Tools------------------------------------------------------

class Map:

    def show(self):

        # locs = self.get_locs()
        routes = self.get_routes()[0]

        # coords = [(l,*(MapQuestGeoCoder.encode(l))) for l in locs]

        # df = pd.DataFrame(coords, columns=['address', 'lat', 'lng'])
        # st.write(df)
        r = self.plot_routes(routes)
        st.pydeck_chart(r)
    
    @st.cache
    def plot_coords(self, df):
        view_state = pdk.ViewState(latitude=50.844041, longitude=4.367202, zoom=8)

        layer = pdk.Layer(
            type="ScatterplotLayer",
            get_position=['lng', 'lat'],
            data=df,
            pickable=True,
            auto_highlight=True,
            get_radius=5000,          # Radius is given in meters
            get_fill_color=[180, 0, 200, 140],  # Set an RGBA value for fill
        )

        r = pdk.Deck(layers=[layer], initial_view_state=view_state)
        return r

    @st.cache
    def plot_routes(self, routes: List[Path]):
        view_state = pdk.ViewState(latitude=50.844041, longitude=4.367202, zoom=7)

        colors = [[255, 0, 0, 255], [0, 255, 0, 255]]
        data = [{'path': path,
                'color': colors[i]} for i, path in enumerate(routes)]

        # df = pd.DataFrame(path, columns=['path'])
        layer = pdk.Layer(
            type="PathLayer",
            data=data,
            pickable=True,
            width_scale=20,
            width_min_pixels=2,
            get_width=5,
            get_color='color',
            get_path='path'
        )

        r = pdk.Deck(layers=[layer],
                    initial_view_state=view_state,
                    map_style='road')
        return r

    def get_locs(self) -> List[str]:
        return st.session_state.screen.get_locations()

    def get_routes(self) -> List[Path]:
        path = st.session_state.screen.get_routes()
        return [path]




# path = Directions.get_directions('Brussels, BE', 'Gent, BE')
# st.write(path)
# st.pydeck_chart(plot_route(path))

class LocationInput:

    def show(self):
        st.button("Use Example", on_click=self.on_use_example)
        st.button("Reset", on_click=self.on_reset)
        st.markdown("Enter locations")
        location = st.text_input("Enter address (Add ,BE to ensure belgian lcoations are used)")
        st.button("Add location", on_click=self.on_add_location, args=(location,))
        st.markdown("Current locations:")
        st.write(self.get_locations())

    def on_add_location(self, location: str):
        success = st.session_state.screen.add_location(location)
        if not success:
            st.markdown("Please Enter a valid location inside belgium")

    def get_locations(self) -> List[str]:
        return st.session_state.screen.get_locations()

    def on_use_example(self):
        example_locs = ['Gent, BE', 'Brussels, BE', 'Antwerpen, BE']
        st.session_state.screen.set_locations(example_locs)

    def on_reset(self):
        st.session_state.screen.set_locations([])

class RoutingScreen:

    def __init__(self):
        self.locations = []
        self.routes = []
        self.num_vehicles = 2

    def show(self):
        input_col, map_col = st.columns(2)

        with input_col:
            LocationInput().show()

        with map_col:
            Map().show()

    def get_locations(self) -> List[str]:
        return self.locations

    def add_location(self, location: str) -> bool:
        try:
            self.locations.append(location)
            return True
        except Exception as e:
            return False

    def compute_routes(self, locations: List[str]) -> List[Path]:
        if len(locations) == 0:
            return []

        solution = self.get_solution()
        paths = []
        for location_ids in solution.vehicle_to_locations:
            locations = [self.locations[i] for i in location_ids]
            st.write(locations)
            paths.append(Directions().get_directions(locations))
        return paths

    def get_routes(self) -> List[Path]:
        return self.compute_routes(self.locations)

    def set_locations(self, locations: List[str]):
        self.locations = locations

    def prepare_problem(self):
        locations = self.get_locations()
        distance_matrix = RouteMatrix().get_distance_matrix(locations)
        problem = Problem(
                          distance_matrix=distance_matrix,
                          num_locations=len(distance_matrix),
                          num_vehicles=self.num_vehicles,
                          depot=0
                        )
        return problem

    def get_solution(self):
        return ORToolsSolver().solve(self.prepare_problem())

# @dataclass
# class Problem:
#     distance_matrix: List[List[int]]
#     num_locations: int
#     num_vehicles: int
#     depot: int

# @dataclass
# class Solution:
#     vehicle_to_locations: List[List[int]]

# class ORToolsSolver:

#     def solver(self , problem: Problem) -> Solution:
        




def init_state():
    if "screen" not in st.session_state:
        st.session_state.screen = RoutingScreen()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    """
    Scenario: Routing

    A team of salesmen need to vist a collection of potential clients. These clients are spread out around belgium. 
    We need to find an optimal way to route the sales people so that they can visit all the clients while wasting the minimal amount of time driving
    """

    init_state()
    st.session_state.screen.show()

# Input Locations: 
# Components: 
#   Location Input

# Get Distance Matrix
    #DistanceMatrix

# Solve with solver
    # Solver

# Get directions]
    # Directions

# Plot Routes
    # Map