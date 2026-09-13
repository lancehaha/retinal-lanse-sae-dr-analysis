"""
Assignment 1 - Problem 1 PDDL submission template.

* Group Member 1:
    - Name:
    - Matric number:

* Group Member 2:
    - Name:
    - Matric number:

* Group Member 3:
    - Name:
    - Matric number:

* Group Member 4:
    - Name:
    - Matric number:
"""

### AFTER YOU COMPLETE 1_PDDL.ipynb, COPY THE MARKED SECTIONS HERE ###

# COPY-FLAG-1 and COPY-FLAG-2 are required for the complete notebook workflow
# to run, but they carry no marks. COPY-FLAG-3 and COPY-FLAG-4 are graded.


# OPTIONAL, UNGRADED AI-AUDIT NOTES
#
# You may record your answers to the notebook's AI-audit activities below.
# The autograder does not read this variable, and leaving it blank has no
# effect on your score. Do not include an AI transcript; concise technical
# conclusions are enough.
optional_ai_audit_notes = r"""
1. AI draft check
   Assumption or invariant examined:
   Smallest counterexample tested:
   Revision made, if any:

2. Fragment 1 (loading while the door may be closed)
   Violated invariant:
   Concrete counterexample:
   Smallest repair:

3. Fragment 2 (incorrect movement effect)
   Violated invariant:
   Concrete counterexample:
   Smallest repair:

4. Fragment 3 (using each passenger's start as the goal)
   Violated invariant:
   Concrete counterexample:
   Smallest repair:

5. Capacity extension
   Why the unload count-link direction decrements occupancy:
   What can happen if load omits (not (reached ?p)):
"""


# COPY-FLAG-1-START

pddl_domain = """
(define (domain elevator)
  (:requirements :strips :typing :negative-preconditions)
  (:types level person)

  (:predicates
    (elevator_at ?l - level)
    (person_at ?p - person ?l - level)
    (person_in_elevator ?p - person)
    (elevator_empty)
    (door_open ?l - level)
    (adjacent_up ?from ?to - level)
    (adjacent_down ?from ?to - level)
  )

  (:action move_up
    :parameters (?from ?to - level)
    :precondition (and ___)
    :effect (and ___)
  )

  (:action move_down
    :parameters (?from ?to - level)
    :precondition (and ___)
    :effect (and ___)
  )

  (:action open_door
    :parameters (?l - level)
    :precondition (and ___)
    :effect (and ___)
  )

  (:action close_door
    :parameters (?l - level)
    :precondition (and ___)
    :effect (and ___)
  )

  (:action load
    :parameters (?p - person ?l - level)
    :precondition (and ___)
    :effect (and ___)
  )

  (:action unload
    :parameters (?p - person ?l - level)
    :precondition (and ___)
    :effect (and ___)
  )
)
"""

# COPY-FLAG-1-END


def generate_pddl_domain(output_file="elevator_domain.pddl"):
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(pddl_domain)


def validate_config(config):
    """Validate the fixed configuration schema used in Problem 1."""
    required = {"num_levels", "elevator_start", "requests"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"config must contain exactly these keys: {sorted(required)}")

    num_levels = config["num_levels"]
    elevator_start = config["elevator_start"]
    requests = config["requests"]

    if not isinstance(num_levels, int) or isinstance(num_levels, bool) or num_levels < 1:
        raise ValueError("num_levels must be a positive integer")
    if (
        not isinstance(elevator_start, int)
        or isinstance(elevator_start, bool)
        or not 0 <= elevator_start < num_levels
    ):
        raise ValueError("elevator_start must name an existing floor")
    if not isinstance(requests, (list, tuple)):
        raise ValueError("requests must be a list or tuple of (start, goal) pairs")

    for request in requests:
        if not isinstance(request, (list, tuple)) or len(request) != 2:
            raise ValueError("each request must be a (start, goal) pair")
        start, goal = request
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(goal, int)
            or isinstance(goal, bool)
        ):
            raise ValueError("request floors must be integers")
        if not 0 <= start < num_levels or not 0 <= goal < num_levels:
            raise ValueError("request floors must name existing floors")


# COPY-FLAG-2-START

def generate_pddl_from_config(config, output_file):
    """Generate one PDDL problem from the validated configuration.

    ``config["requests"][i]`` is ``(start_floor, goal_floor)`` for
    ``person{i + 1}``. Multiple people may share a start or goal, and a
    person's start may already equal their goal.
    """
    validate_config(config)

    num_levels = config["num_levels"]
    elevator_start = config["elevator_start"]
    requests = config["requests"]
    persons = [f"person{i + 1}" for i in range(len(requests))]

    pddl = "(define (problem elevator_problem)\n"
    pddl += "  (:domain elevator)\n"

    levels = " ".join(f"level{i}" for i in range(num_levels))
    pddl += "  (:objects\n"
    pddl += f"    {levels} - level\n"
    if persons:
        pddl += f"    {' '.join(persons)} - person\n"
    pddl += "  )\n\n"

    pddl += "  (:init\n"
    pddl += f"    (___________)\n"  # Use config["elevator_start"]
    pddl += "    (elevator_empty)\n"

    for person, (start, _goal) in zip(persons, requests):
        pddl += f"    (___________)\n"  # Place person at start

    for floor in range(num_levels - 1):
        pddl += f"    (___________)\n"  # Upward adjacency
        pddl += f"    (___________)\n"  # Downward adjacency
    pddl += "  )\n\n"

    pddl += "  (:goal\n"
    pddl += "    (and\n"
    for person, (_start, goal) in zip(persons, requests):
        pddl += f"      (___________)\n"  # Place person at their goal
    pddl += "    )\n"
    pddl += "  )\n"
    pddl += ")"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(pddl)

    return pddl

# COPY-FLAG-2-END


def validate_capacity_config(config):
    """Validate the fixed schema for the capacity extension."""
    required = {"num_levels", "elevator_start", "capacity", "requests"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"config must contain exactly these keys: {sorted(required)}")
    validate_config({key: config[key] for key in ("num_levels", "elevator_start", "requests")})
    capacity = config["capacity"]
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
        raise ValueError("capacity must be a positive integer")


# COPY-FLAG-3-START

pddl_domain_capacity = """
(define (domain elevator)
  (:requirements :strips :typing :negative-preconditions)
  (:types level person count)

  (:predicates
    (elevator_at ?l - level)
    (person_at ?p - person ?l - level)
    (person_in_elevator ?p - person)
    (destination ?p - person ?l - level)
    (reached ?p - person)
    (door_open ?l - level)
    (adjacent_up ?from ?to - level)
    (adjacent_down ?from ?to - level)
    (lift_count ?c - count)
    (next_count ?current ?next - count)
  )

  (:action move_up
    :parameters (?from ?to - level)
    :precondition (and (elevator_at ?from) (adjacent_up ?from ?to)
                       (not (door_open ?from)))
    :effect (and (not (elevator_at ?from)) (elevator_at ?to))
  )

  (:action move_down
    :parameters (?from ?to - level)
    :precondition (and (elevator_at ?from) (adjacent_down ?from ?to)
                       (not (door_open ?from)))
    :effect (and (not (elevator_at ?from)) (elevator_at ?to))
  )

  (:action open_door
    :parameters (?l - level)
    :precondition (and (elevator_at ?l) (not (door_open ?l)))
    :effect (and (door_open ?l))
  )

  (:action close_door
    :parameters (?l - level)
    :precondition (and (elevator_at ?l) (door_open ?l))
    :effect (and (not (door_open ?l)))
  )

  (:action load
    :parameters (?p - person ?l - level ?current ?next - count)
    :precondition (and ___)
    :effect (and ___)
  )

  (:action unload
    :parameters (?p - person ?l - level ?previous ?current - count)
    :precondition (and ___)
    :effect (and ___)
  )
)
"""

# COPY-FLAG-3-END


def generate_capacity_pddl_domain(output_file="elevator_domain_capacity.pddl"):
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(pddl_domain_capacity)


# COPY-FLAG-4-START

def generate_capacity_pddl_from_config(config, output_file):
    """Generate a capacity-aware problem from the validated fixed schema."""
    validate_capacity_config(config)

    num_levels = config["num_levels"]
    elevator_start = config["elevator_start"]
    capacity = config["capacity"]
    requests = config["requests"]
    persons = [f"person{i + 1}" for i in range(len(requests))]
    counts = [f"c{i}" for i in range(capacity + 1)]

    pddl = "(define (problem elevator_capacity_problem)\n"
    pddl += "  (:domain elevator)\n"
    pddl += "  (:objects\n"
    pddl += f"    {' '.join(f'level{i}' for i in range(num_levels))} - level\n"
    if persons:
        pddl += f"    {' '.join(persons)} - person\n"
    pddl += f"    {' '.join(counts)} - count\n"
    pddl += "  )\n\n"

    pddl += "  (:init\n"
    pddl += f"    (elevator_at level{elevator_start})\n"
    pddl += "    (lift_count c0)\n"

    for floor in range(num_levels - 1):
        pddl += f"    (adjacent_up level{floor} level{floor + 1})\n"
        pddl += f"    (adjacent_down level{floor + 1} level{floor})\n"

    for count in range(capacity):
        pddl += f"    (___________)\n"  # c{count} -> c{count + 1}

    for person, (start, goal) in zip(persons, requests):
        pddl += f"    (___________)\n"  # Person's start
        pddl += f"    (___________)\n"  # Person's destination
        if start == goal:
            pddl += f"    (___________)\n"  # Already delivered
    pddl += "  )\n\n"

    pddl += "  (:goal\n"
    pddl += "    (and\n"
    for person in persons:
        pddl += f"      (___________)\n"  # Person has been delivered
    pddl += "    )\n"
    pddl += "  )\n"
    pddl += ")"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(pddl)
    return pddl

# COPY-FLAG-4-END
