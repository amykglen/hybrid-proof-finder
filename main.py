from collections import defaultdict

import graphviz

RANDOM = "$"
RANDOM_NO_REPLACE = "$-"
G = "G"
F = "F"
E = "E"
x = "x"
XOR = "XOR"
OUTPUT = "OUT"
WILDCARD = "*"


class Node:

    def __init__(self, key: str, kind: str, is_input: bool, is_output: bool, color: str):
        self.key = key
        self.kind = kind
        self.is_input = is_input
        self.is_output = is_output
        self.color = color


class Edge:

    def __init__(self, key: str, source_key: str, target_key: str, color: str):
        self.key = key
        self.source_key = source_key
        self.target_key = target_key
        self.color = color


class Graph:

    def __init__(self, name: str):
        self.name = name
        self.nodes = dict()
        self.edges = dict()

    def add_node(self, key: str, kind: str, is_input: bool = False, is_output: bool = False):
        assert key not in self.nodes
        if is_input or is_output:
            color = "grey"
        elif kind == RANDOM_NO_REPLACE:
            color = "red"
        else:
            color = "black"
        self.nodes[key] = Node(key, kind, is_input, is_output, color)

    def add_edge(self, source_key: str, target_key: str):
        key = f"e:{source_key}--{target_key}"
        assert key not in self.edges
        assert source_key in self.nodes and target_key in self.nodes
        color = "red" if self.nodes[source_key].kind == RANDOM_NO_REPLACE else "black"
        self.edges[key] = Edge(key, source_key, target_key, color)

    def print(self):
        for node in self.nodes.values():
            print(f"{node.key}: {node.kind} {'INPUT' if node.is_input else ''} {'OUTPUT' if node.is_output else ''}")
        for edge in self.edges.values():
            print(f"{edge.key}: {edge.source_key}, {edge.target_key}, {edge.color}")

    def print_fancy(self):
        dot = graphviz.Digraph(comment=self.name)
        for node in self.nodes.values():
            dot.node(node.key, node.kind, color=node.color)
        for edge in self.edges.values():
            dot.edge(edge.source_key, edge.target_key, color=edge.color)
        dot.render(f"{self.name}.gv", view=True)
        # TODO: Try to pin all input nodes at same top layer and output nodes at same bottom layer?


class IndistinguishablePair:

    def __init__(self, name: str):
        self.name = name
        self.graph_a = Graph(f"{name}_a")
        self.graph_b = Graph(f"{name}_b")


def create_standard_rules():
    # Rule 1 - ?? PRG thing?
    rule_1 = IndistinguishablePair("standard_1")
    # First graph
    rule_1.graph_a.add_node("$", RANDOM, is_input=True)
    rule_1.graph_a.add_node("G", G)
    rule_1.graph_a.add_edge("$", "G")
    rule_1.graph_a.add_node("out1", WILDCARD, is_output=True)
    rule_1.graph_a.add_node("out2", WILDCARD, is_output=True)
    rule_1.graph_a.add_edge("G", "out1")
    rule_1.graph_a.add_edge("G", "out2")
    # Second graph
    rule_1.graph_b.add_node("$0", RANDOM, is_input=True)
    rule_1.graph_b.add_node("deadend", WILDCARD)
    rule_1.graph_b.add_edge("$0", "deadend")
    rule_1.graph_b.add_node("$1", RANDOM)
    rule_1.graph_b.add_node("$2", RANDOM)
    rule_1.graph_b.add_node("out1", WILDCARD, is_output=True)
    rule_1.graph_b.add_node("out2", WILDCARD, is_output=True)
    rule_1.graph_b.add_edge("$1", "out1")
    rule_1.graph_b.add_edge("$2", "out2")

    # Rule 2 - ?? OTP thing? maybe revisit? remove F node and add ordering? is that actually how should work?
    rule_2 = IndistinguishablePair("standard_2")
    # First graph
    rule_2.graph_a.add_node("$", RANDOM)
    rule_2.graph_a.add_node("out1", WILDCARD, is_output=True)
    rule_2.graph_a.add_edge("$", "out1")
    rule_2.graph_a.add_node("xor", XOR)
    rule_2.graph_a.add_node("F", F)
    rule_2.graph_a.add_node("out2", WILDCARD, is_output=True)
    rule_2.graph_a.add_edge("xor", "F")
    rule_2.graph_a.add_edge("F", "out2")
    rule_2.graph_a.add_edge("$", "xor")
    rule_2.graph_a.add_node("in", WILDCARD, is_input=True)
    rule_2.graph_a.add_edge("in", "xor")
    # Second graph
    rule_2.graph_b.add_node("xor", XOR)
    rule_2.graph_b.add_node("out1", WILDCARD, is_output=True)
    rule_2.graph_b.add_edge("xor", "out1")
    rule_2.graph_b.add_node("in", WILDCARD, is_input=True)
    rule_2.graph_b.add_edge("in", "xor")
    rule_2.graph_b.add_node("$", RANDOM)
    rule_2.graph_b.add_edge("$", "xor")
    rule_2.graph_b.add_node("F", F)
    rule_2.graph_b.add_node("out2", WILDCARD, is_output=True)
    rule_2.graph_b.add_edge("$", "F")
    rule_2.graph_b.add_edge("F", "out2")

    # Rule 3 - rand with/without replacement
    rule_3 = IndistinguishablePair("standard_3")
    # First graph
    rule_3.graph_a.add_node("$", RANDOM)
    rule_3.graph_a.add_node("out", WILDCARD, is_output=True)
    rule_3.graph_a.add_edge("$", "out")
    # Second graph
    rule_3.graph_b.add_node("$", RANDOM_NO_REPLACE)
    rule_3.graph_b.add_node("out", WILDCARD, is_output=True)
    rule_3.graph_b.add_edge("$", "out")

    # Rule 4 - what is E again??
    rule_4 = IndistinguishablePair("standard_4")
    # First graph
    rule_4.graph_a.add_node("E", E)
    rule_4.graph_a.add_node("in", WILDCARD, is_input=True)
    rule_4.graph_a.add_node("out", WILDCARD, is_output=True)
    rule_4.graph_a.add_edge("in", "E")
    rule_4.graph_a.add_edge("E", "out")
    # Second graph
    rule_4.graph_b.add_node("$", RANDOM)
    rule_4.graph_b.add_node("in", WILDCARD, is_input=True)
    rule_4.graph_b.add_node("deadend", WILDCARD)
    rule_4.graph_b.add_node("out", WILDCARD, is_output=True)
    rule_4.graph_b.add_edge("in", "deadend")
    rule_4.graph_b.add_edge("$", "out")

    return [rule_4]


rules = create_standard_rules()

for rule in rules:
    rule.graph_a.print_fancy()
    rule.graph_b.print_fancy()





# my_graph = Graph("test")
# my_graph.add_node("n0i", "m", is_input=True)
# my_graph.add_node("n0", "$")
# my_graph.add_node("n1", "XOR")
# my_graph.add_node("n1.5", "F")
# my_graph.add_node("n2", "output", is_output=True)
# my_graph.add_node("n3", "output", is_output=True)
# my_graph.add_edge("n0", "n1")
# my_graph.add_edge("n0i", "n1")
# my_graph.add_edge("n1", "n2")
# my_graph.add_edge("n1", "n1.5")
# my_graph.add_edge("n1.5", "n3")
# my_graph.print()
# my_graph.print_fancy()
