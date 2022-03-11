import graphviz


class Node:

    def __init__(self, key: str, label: str, is_start: bool, is_end: bool):
        self.key = key
        self.label = label
        self.is_start = is_start
        self.is_end = is_end


class Edge:

    def __init__(self, key: str, source_key: str, target_key: str, color: str):
        self.key = key
        self.source_key = source_key
        self.target_key = target_key
        self.color = color


class Graph:

    def __init__(self):
        self.nodes = dict()
        self.edges = dict()

    def add_node(self, key: str, label: str, is_start: bool = False, is_end: bool = False):
        self.nodes[key] = Node(key, label, is_start, is_end)

    def add_edge(self, key: str, source_key: str, target_key: str, color: str = "black"):
        assert source_key in self.nodes and target_key in self.nodes
        self.edges[key] = Edge(key, source_key, target_key, color)

    def print(self):
        for node in self.nodes.values():
            print(f"{node.key}: {node.label}")
        for edge in self.edges.values():
            print(f"{edge.key}: {edge.source_key}, {edge.target_key}, {edge.color}")

    def print_fancy(self):
        dot = graphviz.Digraph(comment='graph')
        for node in self.nodes.values():
            if node.is_start:
                dot.node(node.key, node.label)
            dot.node(node.key, node.label)
        for edge in self.edges.values():
            dot.edge(edge.source_key, edge.target_key, color=edge.color)
        dot.render("graph.gv", view=True)


my_graph = Graph()
my_graph.add_node("n0", "$", is_start=True)
my_graph.add_node("n1", "XOR")
my_graph.add_node("n2", "end", is_end=True)
my_graph.add_edge("e0", "n0", "n1")
my_graph.add_edge("e1", "n1", "n2", "red")
my_graph.print()
my_graph.print_fancy()
