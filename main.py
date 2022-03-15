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

    def __init__(self, key: str, kind: str, input_rank: int, output_rank: int, color: str, style: str):
        self.key = key
        self.kind = kind
        self.input_rank = input_rank
        self.output_rank = output_rank
        self.color = color
        self.style = style


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

    def add_node(self, key: str, kind: str, input_rank: int = 0, output_rank: int = 0):
        assert key not in self.nodes
        if kind == RANDOM_NO_REPLACE:
            color = "red"
        else:
            color = "black"
        if input_rank:
            style = "dotted"
        elif output_rank:
            style = "dashed"
        else:
            style = "solid"
        self.nodes[key] = Node(key, kind, input_rank, output_rank, color, style)

    def add_edge(self, source_key: str, target_key: str):
        key = f"e:{source_key}--{target_key}"
        assert key not in self.edges
        assert source_key in self.nodes and target_key in self.nodes
        color = "red" if self.nodes[source_key].kind == RANDOM_NO_REPLACE else "black"
        self.edges[key] = Edge(key, source_key, target_key, color)

    def print(self):
        for node in self.nodes.values():
            print(f"{node.key}: {node.kind} {'INPUT' if node.input_rank else ''} {'OUTPUT' if node.output_rank else ''}")
        for edge in self.edges.values():
            print(f"{edge.key}: {edge.source_key}, {edge.target_key}, {edge.color}")

    def print_fancy(self):
        dot = graphviz.Digraph(comment=self.name)
        for node in self.nodes.values():
            if node.input_rank:
                caption = f"in{node.input_rank}"
            elif node.output_rank:
                caption = f"out{node.output_rank}"
            else:
                caption = None
            dot.node(node.key, node.kind, color=node.color, style=node.style, xlabel=caption)
        for edge in self.edges.values():
            dot.edge(edge.source_key, edge.target_key, color=edge.color)
        dot.render(f"{self.name}.gv", view=True)
        # TODO: Try to pin all input nodes at same top layer and output nodes at same bottom layer?


class IndistinguishablePair:

    def __init__(self, name: str):
        self.name = name
        self.graph_a = Graph(f"{name}_a")
        self.graph_b = Graph(f"{name}_b")

    def validate(self):
        # Make sure there are matching numbers of inputs/outputs in the two graphs
        inputs_a = sorted([node.input_rank for node in self.graph_a.nodes.values() if node.input_rank])
        inputs_b = sorted([node.input_rank for node in self.graph_b.nodes.values() if node.input_rank])
        assert inputs_a == inputs_b
        outputs_a = sorted([node.output_rank for node in self.graph_a.nodes.values() if node.output_rank])
        outputs_b = sorted([node.output_rank for node in self.graph_b.nodes.values() if node.output_rank])
        assert outputs_a == outputs_b

        # Make sure at least some inputs or outputs are specified
        assert inputs_a or outputs_a

        # Make sure all 'ranks' are unique
        assert len(set(inputs_a)) == len(inputs_a)
        assert len(set(outputs_a)) == len(outputs_a)


def create_standard_rules():
    # Rule 1 - ?? PRG thing? one random in turns into two out
    rule_1 = IndistinguishablePair("rule_1")
    # First graph
    rule_1.graph_a.add_node("$", RANDOM, input_rank=1)
    rule_1.graph_a.add_node("G", G)
    rule_1.graph_a.add_edge("$", "G")
    rule_1.graph_a.add_node("out1", WILDCARD, output_rank=1)
    rule_1.graph_a.add_node("out2", WILDCARD, output_rank=2)
    rule_1.graph_a.add_edge("G", "out1")
    rule_1.graph_a.add_edge("G", "out2")
    # Second graph
    rule_1.graph_b.add_node("$", RANDOM, input_rank=1)  # Think this could also be wildcard technically?
    rule_1.graph_b.add_node("$1", RANDOM, output_rank=1)
    rule_1.graph_b.add_node("$2", RANDOM, output_rank=2)

    # Rule 2 - ?? OTP thing? maybe revisit? remove F node and add ordering? is that actually how should work?
    rule_2 = IndistinguishablePair("rule_2")
    # First graph
    rule_2.graph_a.add_node("$", RANDOM)
    rule_2.graph_a.add_node("out1", WILDCARD, output_rank=1)
    rule_2.graph_a.add_edge("$", "out1")
    rule_2.graph_a.add_node("xor", XOR)
    rule_2.graph_a.add_node("out2", WILDCARD, output_rank=2)
    rule_2.graph_a.add_edge("xor", "out2")
    rule_2.graph_a.add_edge("$", "xor")
    rule_2.graph_a.add_node("in", WILDCARD, input_rank=1)
    rule_2.graph_a.add_edge("in", "xor")
    # Second graph
    rule_2.graph_b.add_node("xor", XOR)
    rule_2.graph_b.add_node("out1", WILDCARD, output_rank=1)
    rule_2.graph_b.add_edge("xor", "out1")
    rule_2.graph_b.add_node("in", WILDCARD, input_rank=1)
    rule_2.graph_b.add_edge("in", "xor")
    rule_2.graph_b.add_node("$", RANDOM)
    rule_2.graph_b.add_edge("$", "xor")
    rule_2.graph_b.add_node("out2", WILDCARD, output_rank=2)
    rule_2.graph_b.add_edge("$", "out2")

    # Rule 3 - rand with/without replacement  # Don't need out node? just means can swap node?
    rule_3 = IndistinguishablePair("rule_3")
    # First graph
    rule_3.graph_a.add_node("$", RANDOM, output_rank=1)
    # Second graph
    rule_3.graph_b.add_node("$", RANDOM_NO_REPLACE, output_rank=1)

    # Rule 4 - what is E??
    rule_4 = IndistinguishablePair("rule_4")
    # First graph
    rule_4.graph_a.add_node("in", WILDCARD, input_rank=1)
    rule_4.graph_a.add_node("E", E)
    rule_4.graph_a.add_node("out", WILDCARD, output_rank=1)
    rule_4.graph_a.add_edge("in", "E")
    rule_4.graph_a.add_edge("E", "out")
    # Second graph
    rule_4.graph_b.add_node("in", WILDCARD, input_rank=1)
    rule_4.graph_b.add_node("$", RANDOM, output_rank=1)

    # Rule 5 - PRF is like random? What does red in this case mean though?
    rule_5 = IndistinguishablePair("rule_5")
    # First graph
    rule_5.graph_a.add_node("in", RANDOM_NO_REPLACE, input_rank=1)
    rule_5.graph_a.add_node("F", F)
    rule_5.graph_a.add_node("out", WILDCARD, output_rank=1)
    rule_5.graph_a.add_edge("in", "F")
    rule_5.graph_a.add_edge("F", "out")
    # Second graph
    rule_5.graph_b.add_node("in", RANDOM_NO_REPLACE, input_rank=1)
    rule_5.graph_b.add_node("$", RANDOM, output_rank=1)

    # Rule 6 - XOR with random makes random
    rule_6 = IndistinguishablePair("rule_6")
    # First graph
    rule_6.graph_a.add_node("in", WILDCARD, input_rank=1)
    rule_6.graph_a.add_node("xor", XOR)
    rule_6.graph_a.add_edge("in", "xor")
    rule_6.graph_a.add_node("$", RANDOM)
    rule_6.graph_a.add_edge("$", "xor")
    rule_6.graph_a.add_node("out", WILDCARD, output_rank=1)
    rule_6.graph_a.add_edge("xor", "out")
    # Second graph
    rule_6.graph_b.add_node("in", WILDCARD, input_rank=1)
    rule_6.graph_b.add_node("$", RANDOM, output_rank=1)

    return [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6]


rules = create_standard_rules()

for rule in rules:
    rule.validate()
    # rule.graph_a.print_fancy()
    # rule.graph_b.print_fancy()


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
