"""
Problem Description:

Given a set of Facilities F_i, i = 1...f, with a production capacity for each facility PC_i denoting the amount it can produce at each timestep, 
a holding cost H_i denoting the cost to keep unused products from the facility for one timestepand a set of dependency facilities FD_i /subset {1..n}. 
The depenencies form a DAG. We also have a set of products P_i, i = 1...p each of which also has a set of dependencies PD_i /subset {1..n}.

Finally we have a set of orders O_i, i=1.. n each of which require an amount O_{i,j} of product j at a Deadline T_i. We also have a time horizon TH such that T_i <= TH for all i.

We want to minimize the total holding cost required to produce teh Orders

json_spec:
{
    Facilities: [
        {
            name: str
            Production Capacity: int
            Holding Cost: int
            Dependecies: [
                int
            ]
        }
    ],
    
    Products: [
        {
            Name: str
            Dependencies: [
                int
            ]
        }
    ]

    Orders: [
        {
            Amounts: {
                Product 1: int
                Product 2: int
                ...
            }
            Deadline: int
        }
    ]
}

Problem Forumulation:

All variables int

Variables: 
Production at time t at facility i: P_{i,t}:
Holding at time t at facility i: H_{i,t}

Constraints:
Holding Continuity: H_{i,t} = H_{i,t-1} + P_{i,t-1} - sum_{j in FD_i} (P_{j,t-1}) - sum_{j =1 1...n}( O_{j,i,t})
Positive Holding: H_{i,t} >= 0 for all t, for all i
Production Capacity: P_{i,t} <= PC_i for all t for all i

Objective:
sum_{i,t} ( HC_i*H_{i,t})
"""
import streamlit as st
from mip import Model, INTEGER, xsum, OptimizationStatus
from typing import List
from dataclasses import dataclass
import graphviz as graphviz
import pandas as pd
import altair as alt
import json

# ----------------------------------------------------- Solvers ---------------------------------------------------------------------------
@dataclass
class Facility:
    name: str
    capacity: int
    holding_cost: int
    dependents: List[int]


@dataclass
class Product:
    name: str
    dependencies: List[int]


@dataclass
class Order:
    product_amounts: List[int]
    deadline: int

@dataclass
class Problem:
    facilities: List[Facility]
    products: List[Product]
    orders: List[Order]

    def __post_init__(self):
        self.horizon = max(order.deadline for order in self.orders) + 1

        for order in self.orders:
            facility_amounts = []
            for f_id, f in enumerate(self.facilities):
                facility_amounts.append(sum(order.product_amounts[p_id] for p_id, p in enumerate(self.products) if f_id in p.dependencies))
            order.facility_amounts = facility_amounts

@dataclass
class Schedule:
    production: List[int]
    holding: List[int]

@dataclass
class Solution:
    schedules: List[Schedule]
    quality: str
    cost: float
    problem: Problem


def parse_json(settings: dict) -> Problem:
    if type(settings) == str:
        settings = json.loads(settings)
    print(settings)
    facilities = [
        Facility(
            facility["name"], 
            facility["capacity"], 
            facility["holding_cost"], 
            facility["dependents"]
        )
        for facility in settings["facilities"]
    ]

    products = [
        Product(
            product["name"], 
            product["dependencies"]
        )
        for product in settings["products"]
    ]

    orders = [
        Order(
            order["product_amounts"],
            order["deadline"]
        )
        for order in settings["orders"]
    ]

    return Problem(facilities, products, orders)

def get_example() -> Problem:

    facilities = [
        Facility("A", 1, 10, [1]),
        Facility("B", 2, 100, []),
        Facility("C", 1, 40, []),
    ]

    products = [
        Product("One", [0, 2]),
        Product("Two", [1, 2])
    ]

    orders = [
        Order([2, 0], 2),
        Order([0, 2], 3)
    ]
    return Problem(facilities, products, orders)

def get_example_json():
    s = """
        {
    "facilities":[
        {
            "name":"A",
            "capacity":1,
            "holding_cost":10,
            "dependents":[
                1
            ]
        },
        {
            "name":"B",
            "capacity":2,
            "holding_cost":100,
            "dependents":[]
        },
        {
            "name":"C",
            "capacity":1,
            "holding_cost":40,
            "dependents":[]
        }
    ],
    "products":[
        {
            "name":"One",
            "dependencies":[
                1,
                2
            ]
        },
        {
            "name":"Two",
            "dependencies":[
                0,
                2
            ]
        }
    ],
    "orders":[
        {
            "product_amounts":[1,2],
            "deadline": 4
        },
        {
            "product_amounts":[2,0],
            "deadline": 5
        }
    ]
}
    """
    return json.loads(s)

class MIPSolver:

    @st.cache
    def solve(problem: Problem):

        timesteps = range(problem.horizon)
        facilities = problem.facilities
        m = Model()

        # Productions + Capacities
        P = [[m.add_var(name=f'P_{f.name}_{t}', var_type=INTEGER, lb=0, ub=f.capacity) for t in timesteps] for f in facilities]

        # Holding + Positive Constraint
        H = [[m.add_var(name=f'H_{f.name}_{t}', var_type=INTEGER, lb=0) for t in range(problem.horizon+1)] for f in facilities]

        # Holding Continuity
        for f_id, f in enumerate(facilities):
            for t in timesteps:
                m += H[f_id][t + 1] == H[f_id][t] + P[f_id][t] - xsum(P[fd][t] for fd in f.dependents) - sum(order.facility_amounts[f_id] for order in problem.orders if order.deadline == t)

        # Inital State
        for f_id, f in enumerate(facilities):
            m += H[f_id][0] == 0

        # Objctive
        m.objective = xsum(f.holding_cost * H[f_id][t] for t in timesteps for f_id, f in enumerate(facilities))
        
        m.max_gap = 0.01
        status = m.optimize(max_seconds=60)

        schedules = []
        if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
            for f in facilities:
                production = [m.var_by_name(f'P_{f.name}_{t}').x for t in timesteps]
                holding = [m.var_by_name(f'H_{f.name}_{t}').x for t in timesteps]
                schedules.append(Schedule(production, holding))


        if status == OptimizationStatus.OPTIMAL:
            qual = "Optimal"
        elif status == OptimizationStatus.FEASIBLE:
            qual = "Feasible"
        return Solution(schedules, qual, m.objective_value, problem)

# ----------------------------------------------------- Displays ---------------------------------------------------------------------------


class GraphDisplay:

    def show(self):
        problem = self.get_problem()
        st.markdown("### Production Setup")
        st.graphviz_chart(self.generate_graph(problem))

    def generate_graph(self, problem: Problem) -> object:
        graph = graphviz.Digraph()

        for facility in problem.facilities[::-1]:
            graph.node(name=facility.name, label=f"Facility: {facility.name}\nCapacity: {facility.capacity}\nHolding Cost: {facility.holding_cost}")
            for facility_id in facility.dependents:
                graph.edge(facility.name, problem.facilities[facility_id].name)

        for product in problem.products:
            graph.node(name=product.name, label=f"Product: {product.name}", style='filled')
            for facility_id in product.dependencies:
                graph.edge(problem.facilities[facility_id].name, product.name)

        return graph

    def get_problem(self):
        return st.session_state.screen.get_problem()


class OrderDisplay:

    def show(self):
        problem = self.get_problem()
        st.markdown("### Current Orders")
        st.write(self.orders_to_df(problem.orders, problem.products))

    def get_problem(self):
        return st.session_state.screen.get_problem()

    @staticmethod
    def orders_to_df(orders, products):
        df_arr = []
        for order in orders:
            temp = {f"Product {products[i].name}": amount for i, amount in enumerate(order.product_amounts)}
            temp['Deadline'] = order.deadline
            df_arr.append(temp)

        return pd.DataFrame(df_arr)


class Settings:

    def show(self):
        st.markdown("### Settings")
        st.button("Use Default", on_click=self.on_use_default)
        setup = self.get_setup()
        with st.expander("Show setup"):
            modified_setup = st.text_area(label="JSON Spec", value=json.dumps(setup, indent=4, sort_keys=True), height=500)
            st.button("Update Settings", on_click=self.on_update_settings, args=(modified_setup,))

    def on_use_default(self):
        st.session_state.screen.use_default()

    def get_setup(self):
        return st.session_state.screen.get_setup()

    def on_update_settings(self, setup):
        try:
            parse_json(setup)
            st.session_state.screen.set_setup(setup)
        except Exception as e:
            st.write(f"Invalid Spec: {str(e)}")

class SolutionDisplay:

    def show(self):
        solution = self.get_solution()
        problem = self.get_problem()
        sol_df = self.schedules_to_df(solution)
        prod_chart = self.generate_prod_chart(sol_df)
        hold_chart = self.generate_hold_chart(sol_df)

        st.markdown(f"Optimized Cost {solution.cost}")
        st.altair_chart(prod_chart)
        st.altair_chart(hold_chart)


    def get_solution(self) -> Solution:
        return st.session_state.screen.get_solution()

    def get_problem(self) -> Problem:
        return st.session_state.screen.get_problem()

    @staticmethod
    def generate_prod_chart(df):
        df = df[df["Type"] == "Production"]
        chart = alt.Chart(df).mark_bar().encode(
            x='Time:O',
            y=alt.Y(
                'Amount',
                title='Amount to produce',
                axis=alt.Axis(format='~s')
            ),
            facet=alt.Facet('Facility', columns=5),
        ).properties(
            title='Optimized Production Schedules',
            width=400
        )
        return chart

    @staticmethod
    def generate_hold_chart(df):
        df = df[df["Type"] == "Holding"]
        chart = alt.Chart(df).mark_bar().encode(
            x='Time:O',
            y=alt.Y(
                'Amount',
                title='Amount to produce',
                axis=alt.Axis(format='~s')
            ),
            facet=alt.Facet('Facility', columns=5),
        ).properties(
            title='Holding Schedules',
            width=400
        )
        return chart

    @staticmethod
    def schedules_to_df(solution):
        schedules = solution.schedules
        problem = solution.problem
        entries = []
        for i, schedule in enumerate(schedules):
            for t, a in enumerate(schedule.production):
                entries.append({'Time': t, 'Amount': a, 'Type': "Production", 'Facility': problem.facilities[i].name})
            for t, a in enumerate(schedule.holding):
                entries.append({'Time': t, 'Amount': a, 'Type': "Holding", 'Facility': problem.facilities[i].name})
        return pd.DataFrame(entries)


# ----------------------------------------------------- Screen ---------------------------------------------------------------------------
class PlanningScreen:

    def __init__(self):
        self.setup = get_example_json()
        self.problem = parse_json(self.setup)

    def show(self):
        st.markdown(
        """
        Scenario: Production Planning
        """
        )
        graph_col, order_col, setting_col = st.columns(3)

        with graph_col:
            GraphDisplay().show()

        with order_col:
            OrderDisplay().show()

        with setting_col:
            Settings().show()

        SolutionDisplay().show()

    def get_problem(self) -> Problem:
        return self.problem

    def get_solution(self) -> Solution:
        return MIPSolver.solve(self.problem)

    def get_setup(self):
        return self.setup

    def use_default(self):
        self.setup = get_example_json()
        self.problem = parse_json(self.setup)

    def set_setup(self, setup):
        self.setup = setup
        self.problem = parse_json(self.setup)

def init_state():
    if "screen" not in st.session_state:
        st.session_state.screen = PlanningScreen()


if __name__ == '__main__':
    st.set_page_config(layout="wide")

    init_state()
    st.session_state.screen.show()




