from scheduling_app import JobSchedulerScreen
from forecasting_app import ForecastScreen
from planning_app import PlanningScreen
from routing_app import RoutingScreen
from queue_app import QueueScreen
import streamlit as st
st.set_page_config(layout="wide")

screens = {
    "Scheduling": JobSchedulerScreen,
    "Forecasting": ForecastScreen,
    "Planning": PlanningScreen,
    "Routing": RoutingScreen,
    "Simulation": QueueScreen
    }

def change_screen():
    st.session_state.screen = screens[st.session_state.screen_choice]()

st.sidebar.selectbox("Choose Application", options=screens.keys(), on_change=change_screen, key='screen_choice')

if "screen" not in st.session_state:
    st.session_state.screen = JobSchedulerScreen()

st.session_state.screen.show()