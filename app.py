
import streamlit as st
st.set_page_config(layout="wide")
class EmptyScreen:

    def show(self):
        st.markdown("Failed to Load")

try:
    from scheduling_app import JobSchedulerScreen
except Exception as e:
    JobSchedulerScreen = EmptyScreen
    print(f"Failed to import: JobSchedulerScreen - {e}")

try:
    from forecasting_app import ForecastScreen
except Exception as e:
    ForecastScreen = EmptyScreen
    print(f"Failed to import: ForecastScreen - {e}")

try:
    from planning_app import PlanningScreen
except Exception as e:
    PlanningScreen = EmptyScreen
    print(f"Failed to import: PlanningScreen - {e}")

try:
    from routing_app import RoutingScreen
except Exception as e:
    RoutingScreen = EmptyScreen
    print(f"Failed to import: RoutingScreen - {e}")

try:
    from queue_app import QueueScreen
except Exception as e:
    QueueScreen = EmptyScreen
    print(f"Failed to import: QueueScreen - {e}")

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