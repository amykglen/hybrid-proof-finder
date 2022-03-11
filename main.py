import graphviz


class Node:

    def __init__(self, key: str, label: str, is_input: bool, is_output: bool):
        self.key = key
        self.label = label
        self.is_input = is_input
        self.is_output = is_output


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

    def add_node(self, key: str, label: str, is_input: bool = False, is_output: bool = False):
        assert key not in self.nodes
        self.nodes[key] = Node(key, label, is_input, is_output)

    def add_edge(self, key: str, source_key: str, target_key: str, color: str = "black"):
        assert key not in self.edges
        assert source_key in self.nodes and target_key in self.nodes
        self.edges[key] = Edge(key, source_key, target_key, color)

    def print(self):
        for node in self.nodes.values():
            print(f"{node.key}: {node.label} {'INPUT' if node.is_input else ''} {'OUTPUT' if node.is_output else ''}")
        for edge in self.edges.values():
            print(f"{edge.key}: {edge.source_key}, {edge.target_key}, {edge.color}")

    def print_fancy(self):
        dot = graphviz.Digraph(comment='graph')
        for node in self.nodes.values():
            if node.is_input or node.is_output:
                dot.node(node.key, node.label, color="grey")
            dot.node(node.key, node.label)
        for edge in self.edges.values():
            dot.edge(edge.source_key, edge.target_key, color=edge.color)
        dot.render("graph.gv", view=True)


my_graph = Graph()
my_graph.add_node("n0i", "m", is_input=True)
my_graph.add_node("n0", "$")
my_graph.add_node("n1", "XOR")
my_graph.add_node("n1.5", "F")
my_graph.add_node("n2", "output", is_output=True)
my_graph.add_node("n3", "output", is_output=True)
my_graph.add_edge("e0", "n0", "n1")
my_graph.add_edge("e0i", "n0i", "n1")
my_graph.add_edge("e1", "n1", "n2", "red")
my_graph.add_edge("e2", "n1", "n1.5")
my_graph.add_edge("e3", "n1.5", "n3")
my_graph.print()
my_graph.print_fancy()
