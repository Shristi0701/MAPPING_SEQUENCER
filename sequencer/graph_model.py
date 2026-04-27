import networkx as nx


class CourseGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.credits = {}

    def add_course(self, course_id, credits=3):
        self.graph.add_node(course_id)
        self.credits[course_id] = credits

    def add_prerequisite(self, course, prereq):
        # Edge means: prereq must come BEFORE course
        self.graph.add_edge(prereq, course)

    def has_cycle(self):
        return not nx.is_directed_acyclic_graph(self.graph)

    def get_topo_order(self):
        if self.has_cycle():
            return None, "Cycle detected — check prerequisites"
        return list(nx.topological_sort(self.graph)), None