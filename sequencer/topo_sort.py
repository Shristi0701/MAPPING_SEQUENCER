from sequencer.graph_model import CourseGraph


def build_semester_plan(courses, max_credits_per_sem=12):
    """
    courses: list of dicts like:
      {
        "id": "CS101",
        "credits": 3,
        "prerequisites": ["CS100"]
      }

    Returns: (plan, error)
      plan  = list of semesters, each is a list of course IDs
      error = None if successful, error string if failed
    """
    cg = CourseGraph()

    # Add all courses to graph
    for c in courses:
        cg.add_course(c["id"], c.get("credits", 3))

     # Add all prerequisite edges
    for c in courses:
        for prereq in c.get("prerequisites", []):
            cg.add_prerequisite(c["id"], prereq)

    # Get valid topological order
    order, error = cg.get_topo_order()
    if error:
        return None, error

    # Distribute courses into semesters by credit limit
    semesters = []
    current_sem = []
    current_credits = 0

    for course_id in order:
        credits = cg.credits.get(course_id, 3)

        # Start new semester if adding this course exceeds limit
        if current_credits + credits > max_credits_per_sem and current_sem:
            semesters.append(current_sem)
            current_sem = []
            current_credits = 0
        
        current_sem.append(course_id)
        current_credits += credits

    # Add last semester
    if current_sem:
        semesters.append(current_sem)

    return semesters, None