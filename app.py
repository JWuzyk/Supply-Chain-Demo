
import streamlit as st
st.set_page_config(layout="wide")
class EmptyScreen:

    def show():
        st.markdown("Failed to Load")

try:
    from scheduling_app import JobSchedulerScreen
except:
    JobSchedulerScreen = EmptyScreen

try:
    from forecasting_app import ForecastScreen
except:
    ForecastScreen = EmptyScreen
    print("Failed to import:")

try:
    from planning_app import PlanningScreen
except:
    PlanningScreen = EmptyScreen
    print("Failed to import:")

try:
    from routing_app import RoutingScreen
except:
    RoutingScreen = EmptyScreen
    print("Failed to import:")

try:
    JobSchedulerScreen = EmptyScreen
    from queue_app import QueueScreen
except:
    print("Failed to import:")

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