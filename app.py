import streamlit as st
from typing import List, Optional, Dict, Set
import altair as alt
from ortools.sat.python import cp_model  # Import Python wrapper for or-tools CP-SAT solver.
from pydantic.dataclasses import dataclass
import pandas as pd
import collections



# -----------------------------Problem Variables-----------------------------------
Machine = str


@dataclass
class Task:
    machine: Machine
    duration: int


@dataclass
class AssignedTask:
    job_id: int
    machine: Machine
    duration: int
    start: int
    end: int

    def __post_init__(self):
        assert self.start + self.duration == self.end, f"Assigned Task has invalid start, end time, {self.start} + {self.duration} != {self.end}"


Job = List[Task]
MachineSchedule = List[AssignedTask]
Schedule = Dict[Machine, MachineSchedule]  # Solution


@dataclass
class Problem:
    machines: Set[Machine]
    jobs: List[Job]

    def add_job(self, job: Job):
        for t in job:
            assert t.machine in self.machines
        self.jobs.append(job)

    def add_machine(self, machine: Machine) -> bool:
        if machine in self.machines:
            print("Machine already added")
            return False

        self.machines.add(machine)

# -----------------------------Solvers-----------------------------------
# TODO: Add Timeouts


class Solver:

    def __call__(self, Problem) -> Schedule:
        pass


class NaiveSolver(Solver):

    def __call__(self, problem: Problem) -> Schedule:
        machine_schedules = {m: [] for m in problem.machines}
        for job_id, job in enumerate(problem.jobs):
            previous_finished = 0
            for i, task in enumerate(job):
                schedule = machine_schedules[task.machine]
                machine_free = schedule[-1].end if len(schedule) > 0 else 0
                start = max(machine_free, previous_finished)
                end = start + task.duration
                previous_finished = end

                schedule.append(AssignedTask(job_id, task.machine, task.duration, start, end))

        return machine_schedules


class ORToolsSolver(Solver):
    # https://developers.google.com/optimization/scheduling/job_shop

    def __call__(self, problem: Problem) -> Schedule:
        # """Minimal jobshop problem solver using google OR-Tools"""

        # Create the model
        model = cp_model.CpModel()  # Constraint programming model

        # Computes horizon dynamically as the sum of all durations.
        horizon = sum(task.duration for job in problem.jobs for task in job)

        # Named tuple to store information about created variables.
        task_type = collections.namedtuple('task_type', 'start end interval')

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        # Create variables
        for job_id, job in enumerate(problem.jobs):
            for task_id, task in enumerate(job):
                machine = task.machine
                duration = task.duration
                suffix = f'_{job_id}_{task_id}'

                start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                    'interval' + suffix)

                all_tasks[job_id, task_id] = task_type(start=start_var,
                                                       end=end_var,
                                                       interval=interval_var)

                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in problem.machines:
            model.AddNoOverlap(machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(problem.jobs):
            for task_id, task in enumerate(job[:-1]):
                model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

        # Makespan objective.
        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, [
            all_tasks[job_id, len(job) - 1].end
            for job_id, job in enumerate(problem.jobs)
        ])
        model.Minimize(obj_var)

        # Solve model.
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        machine_schedules = collections.defaultdict(list)

        if status == cp_model.OPTIMAL:
            # Create one list of assigned tasks per machine.
            for job_id, job in enumerate(problem.jobs):
                for task_id, task in enumerate(job):
                    machine = task.machine
                    start = solver.Value(all_tasks[job_id, task_id].start)
                    machine_schedules[machine].append(
                        AssignedTask(job_id=job_id,
                                     machine=machine,
                                     duration=task.duration,
                                     start=start,
                                     end=start + task.duration)
                        )
        return machine_schedules

# -----------------------------Display Components-----------------------------------
class SetSolver:

    def show(self):
        available_solvers: List[str] = st.session_state.screen.get_available_solvers()
        solver = st.selectbox("Select Solver", available_solvers)
        st.session_state.screen.set_solver(solver)


class Display:

    def show(self):
        solution = st.session_state.screen.get_solution()
        chart = self.make_chart(solution)
        if chart is None:
            st.markdown("No Jobs to plot")
        else:
            st.altair_chart(chart, use_container_width=True)
            makespan = self.get_makespan(solution)
            if makespan > 0:
                st.markdown(f"Total Time to complete all jobs: {makespan}")

    def make_chart(self, solution):
        task_list = []
        for name, machine_schedule in solution.items():
            for task in machine_schedule:
                task_list.append({"machine": name, "start": task.start, "end": task.end, "job_id": task.job_id})
        print(task_list)
        return alt.Chart(pd.DataFrame(task_list)).mark_bar().encode(
            x='start',
            x2='end',
            y='machine',
            color=alt.Color('job_id:N')
        ).properties(
            width=200,
            height=150
        ) if task_list else None

    @staticmethod
    def get_makespan(solution: Schedule) -> int:
        return max([machine_schedule[-1].end for machine_schedule in solution.values()]) if len(solution) > 0 else 0


class MachineCreator:

    def show(self):
        st.markdown("Add Machine")
        name = st.text_input("Machine Name")
        st.button("Add Machine", on_click=self.on_add_machine, args=(name,))

    def on_add_machine(self, name: str):
        st.session_state.screen.add_machine(name)


class MachineDisplay:

    def show(self):
        st.markdown("Available Machines:")
        machines = self.get_machines()
        if not machines:
            st.markdown("No Available Machines")
        else:   
            df = pd.DataFrame(machines)
            df.columns = ["Machine Name"]
            st.write(df)

    def get_machines(self) -> List[Machine]:
        return st.session_state.screen.get_machines()


class JobDisplay:

    def show(self):
        st.markdown("Current jobs:")
        jobs = self.get_jobs()
        if not jobs:
            st.markdown("No Current Jobs")
        else:
            for job_id, job in enumerate(jobs):
                st.markdown(f"Job {job_id}")
                df = pd.DataFrame([
                    {"Machine": task.machine, "Task Length": task.duration}
                    for task in job])
                st.write(df)

    def get_jobs(self) -> List[Job]:
        return st.session_state.screen.get_jobs()


class JobCreator:

    def __init__(self):
        self.temp_job: Job = []

    def show(self):
        machines = st.session_state.screen.get_machines()
        st.markdown("Create Job")
        duration = st.number_input("Job Length", min_value=0, format="%i")
        machine_name = st.selectbox("Machine Name", machines)
        st.button("Add task", on_click=self.on_add_task, args=(machine_name, duration))
        st.markdown(f"Job Currently Beign Created")
        df = pd.DataFrame([
            {"Machine": task.machine, "Task Length": task.duration}
            for task in self.temp_job])
        st.write(df)
        st.button("Create Job", on_click=self.on_create_job)

    def on_create_job(self):
        if len(self.temp_job) > 0:
            st.session_state.screen.add_job(self.temp_job)
            self.temp_job = []

    def on_add_task(self, name: str, duration: int):
        self.temp_job.append(Task(name, duration))


class Defaults:

    def show(self):
        st.button("Use Example", on_click=self.on_use_example_problem)
        st.button("Reset", on_click=self.on_reset)

    def on_use_example_problem(self):

        machines = ["Lathe", "Milling Machine", "Rolling Machine", "Drill", "Workbench"]
        jobs = [
            [Task("Lathe", 5), Task("Milling Machine", 2), Task("Workbench", 8)],
            [Task("Rolling Machine", 5), Task("Milling Machine", 3), Task("Workbench", 2), Task("Drill", 1)],
            [Task("Lathe", 5), Task("Drill", 3)],
            [Task("Drill", 1), Task("Milling Machine", 2), Task("Drill", 1)]
        ]
        example_problem = Problem(machines, jobs)
        st.session_state.screen.set_problem(example_problem)

    def on_reset(self):
        empty_problem = Problem(set(), [])
        st.session_state.screen.set_problem(empty_problem)


class Screen:
    pass


class JobSchedulerScreen(Screen):

    def __init__(self):
        self.problem = Problem(set(), [])
        self.available_solvers = {"SAT": ORToolsSolver(), "Naive": NaiveSolver()}
        self.solver = NaiveSolver()
        self.job_creator = JobCreator()

    def show(self):
        # Columns: Create, Current, Solution

        create_col, current_col, solution_col = st.columns(3)

        with create_col:
            st.markdown("# Create Jobs")
            MachineCreator().show()
            self.job_creator.show()

        with current_col:
            st.markdown("# Current Jobs")
            Defaults().show()
            MachineDisplay().show()
            JobDisplay().show()

        with solution_col:
            st.markdown("# Solution")
            SetSolver().show()
            Display().show()

    def get_solution(self):
        return self.solver(self.problem)

    def get_machines(self) -> Set[Machine]:
        return self.problem.machines

    def get_jobs(self) -> Job:
        return self.problem.jobs

    def add_machine(self, machine: Machine):
        self.problem.machines.add(machine)

    def add_job(self, job: Job):
        self.problem.jobs.append(job)

    def set_problem(self, problem: Problem):
        self.problem = problem

    def get_available_solvers(self) -> List[str]:
        return self.available_solvers.keys()

    def set_solver(self, solver: str):
        self.solver = self.available_solvers[solver]


def init_state():
    if "screen" not in st.session_state:
        st.session_state.screen = JobSchedulerScreen()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    """
    # Scenario: Job Shop Problem

    A workshop with a few machines recieves a variety of jobs. Each job requires some time on some of the machines in the workshop in some fixed order.
    This dashbaord allows you to enter a list of jobs and displays a schdeule to perform the jobs in the least amount of time.

    To see an example click on the "Use Example" button.
    """

    init_state()
    st.session_state.screen.show()