"""
This is a proof-of-concept tool that attempts to find hybrid proofs between specified endpoints using a graph-based
method that is built on top of the visual proof representation being developed by Mike Rosulek.
"""
import copy
import itertools
import time
from collections import defaultdict
from typing import Dict, Set

import graphviz
from PyPDF2 import PdfFileMerger

RANDOM = "$"
RANDOM_NO_REPLACE = "$-"
G = "G"
F = "F"
E = "E"
XOR = "XOR"
WILDCARD = "*"
DEADEND = "//"


class Node:

    def __init__(self, key: str, kind: str, input_rank: int, output_rank: int):
        self.key = key
        self.kind = kind
        self.input_rank = input_rank
        self.output_rank = output_rank


class Edge:

    def __init__(self, key: str, source_key: str, target_key: str):
        self.key = key
        self.source_key = source_key
        self.target_key = target_key


class Graph:

    def __init__(self, name: str):
        self.name = name
        self.nodes = dict()
        self.edges = dict()

    def add_node(self, key: str, kind: str, input_rank: int = 0, output_rank: int = 0):
        assert key not in self.nodes
        self.nodes[key] = Node(key, kind, input_rank, output_rank)

    def add_edge(self, source_key: str, target_key: str):
        key = f"{source_key}--{target_key}"
        assert key not in self.edges
        assert source_key in self.nodes and target_key in self.nodes
        self.edges[key] = Edge(key, source_key, target_key)

    def remove_node(self, node_key: str):
        if node_key in self.nodes:
            del self.nodes[node_key]
        edge_keys_to_delete = [edge.key for edge in self.edges.values()
                               if edge.source_key == node_key or edge.target_key == node_key]
        for edge_key in edge_keys_to_delete:
            del self.edges[edge_key]

    def print(self):
        for node in self.nodes.values():
            print(f"{node.key}: {node.kind} {'INPUT' if node.input_rank else ''} {'OUTPUT' if node.output_rank else ''}")
        for edge in self.edges.values():
            print(f"{edge.key}: {edge.source_key}, {edge.target_key}")

    def print_fancy(self, auto_open_output: bool = False):
        dot = graphviz.Digraph(comment=self.name)
        for node in self.nodes.values():
            if node.kind == RANDOM_NO_REPLACE:
                color = "red"
            else:
                color = "black"
            if node.input_rank:
                caption = f"in{node.input_rank}"
                style = "dotted"
            elif node.output_rank:
                caption = f"out{node.output_rank}"
                style = "dashed"
            else:
                caption = None
                style = "solid"
            dot.node(node.key, node.kind, color=color, style=style, xlabel=caption)
        for edge in self.edges.values():
            color = "red" if self.nodes[edge.source_key].kind == RANDOM_NO_REPLACE else "black"
            dot.edge(edge.source_key, edge.target_key, color=color)
        dot.render(f"{self.name}.gv", view=auto_open_output)
        # TODO: Try to pin all input nodes at same top layer and output nodes at same bottom layer?

    def change_node_key(self, old_key: str, new_key: str):
        # Delete the old node
        assert old_key in self.nodes
        node = self.nodes[old_key]
        del self.nodes[old_key]

        # Make sure new node key isn't in the graph already
        if new_key in self.nodes:
            new_new_key = f"{new_key}--{time.time()}"
            self.change_node_key(new_key, new_new_key)

        # Add node under new key
        assert new_key not in self.nodes
        node.key = new_key
        self.nodes[new_key] = node

        # Then edit any attached edges
        edge_keys_to_modify = {edge.key for edge in self.edges.values()
                               if edge.source_key == old_key or edge.target_key == old_key}
        for edge_key in edge_keys_to_modify:
            edge = self.edges[edge_key]
            del self.edges[edge_key]
            if edge.source_key == old_key:
                edge.source_key = new_key
            elif edge.target_key == old_key:
                edge.target_key = new_key
            new_edge_key = self.get_edge_key(edge)
            edge.key = new_edge_key
            self.edges[new_edge_key] = edge

    @staticmethod
    def get_edge_key(edge: Edge) -> str:
        return f"{edge.source_key}--{edge.target_key}"


class Rule:
    """
    A rule defines two graphs (a and b) that are considered indistinguishable (in either direction). The two graphs
    must have the same number of 'input' and 'output' nodes. The 'rank' of input and output nodes allows for order to
    matter; input and output nodes in one template are mapped to input/output nodes with the same rank in the other.
    """

    def __init__(self, name: str):
        self.name = name
        self.graph_a = Graph(f"{name}_a")
        self.graph_b = Graph(f"{name}_b")

    def validate(self):
        # Make sure there are matching numbers of inputs/outputs in the two graphs
        inputs_a = [node.input_rank for node in self.graph_a.nodes.values() if node.input_rank]
        outputs_a = [node.output_rank for node in self.graph_a.nodes.values() if node.output_rank]
        inputs_b = [node.input_rank for node in self.graph_b.nodes.values() if node.input_rank]
        outputs_b = [node.output_rank for node in self.graph_b.nodes.values() if node.output_rank]
        assert inputs_a or outputs_a
        assert sorted(inputs_a) == sorted(inputs_b)
        assert sorted(outputs_a) == sorted(outputs_b)

        # Make sure all 'ranks' are unique
        assert len(set(inputs_a)) == len(inputs_a)
        assert len(set(outputs_a)) == len(outputs_a)


class ProofFinder:

    def __init__(self, use_standard_rules: bool = True):
        self.start = Graph("start")
        self.end = Graph("end")
        self.rules = self.create_standard_rules() if use_standard_rules else []
        # Store standard node "kinds"
        self.random = RANDOM
        self.random_no_replace = RANDOM_NO_REPLACE
        self.g = G
        self.f = F
        self.e = E
        self.xor = XOR
        self.wildcard = WILDCARD
        self.deadend = DEADEND

    def find_proof(self):
        start = self.start
        end = self.end
        rules = self.rules
        graph_paths = [[(copy.deepcopy(start), "-")]]
        count = 0

        while not any(self.has_reached_end_state(graph_path, start, end) for graph_path in graph_paths) and count < 30:
            count += 1
            print(f"On iteration {count} of while loop. Have {len(graph_paths)} graph paths currently.")
            extended_graph_paths = list()

            # Get rid of any identical graph paths
            paths_seen = set()
            pruned_paths = []
            for graph_path in graph_paths:
                latest_graph = graph_path[-1][0]
                if latest_graph.name not in paths_seen:
                    pruned_paths.append(graph_path)
                    paths_seen.add(latest_graph.name)
            print(f"After pruning of identical paths, have {len(pruned_paths)} graph paths")
            graph_paths = pruned_paths

            for graph_path in graph_paths:
                latest_graph_tuple = graph_path[-1]
                latest_graph = latest_graph_tuple[0]
                print(f"  Attempting to extend graph path {graph_paths.index(graph_path)}; "
                      f"latest graph is {latest_graph.name}")

                for rule in rules:
                    # TODO: Prevent from looping back to previous state?
                    # First look for swaps that can be done from a --> b
                    matching_subgraphs_a = self.has_subgraph(latest_graph, rule.graph_a)
                    print(f"    Found {len(matching_subgraphs_a)} subgraphs in latest graph in this path that match "
                          f"{rule.name}a template")
                    for matching_subgraph, template_a_to_graph_key_map in matching_subgraphs_a:
                        new_graph_with_b = self.swap_subgraphs(rule.graph_a, rule.graph_b, latest_graph,
                                                               template_a_to_graph_key_map)
                        extended_path = copy.deepcopy(graph_path)
                        extended_path.append((new_graph_with_b, f"{rule.name}_a-->b"))
                        extended_graph_paths.append(extended_path)

                    # Then look for swaps that can be done from b --> a
                    matching_subgraphs_b = self.has_subgraph(latest_graph, rule.graph_b)
                    print(f"    Found {len(matching_subgraphs_b)} subgraphs in latest graph in this path that match "
                          f"{rule.name}b template")
                    for matching_subgraph, template_b_to_graph_key_map in matching_subgraphs_b:
                        new_graph_with_a = self.swap_subgraphs(rule.graph_b, rule.graph_a, latest_graph,
                                                               template_b_to_graph_key_map)
                        extended_path = copy.deepcopy(graph_path)
                        extended_path.append((new_graph_with_a, f"{rule.name}_b-->a"))
                        extended_graph_paths.append(extended_path)

            if extended_graph_paths:
                graph_paths = extended_graph_paths

        # Save all solutions found at the first depth a solution was found in
        for graph_path in graph_paths:
            if self.has_reached_end_state(graph_path, start, end):
                print(f"Found solution: See {graph_path[-1][0].name}.gv.pdf")
                self.save_graph_path(graph_path, start, end)

    def has_reached_end_state(self, graph_path: list, start: Graph, end: Graph) -> bool:
        latest_graph = graph_path[-1][0]
        reached_end_state = True if self.has_subgraph(latest_graph, end, include_edges_to_outsiders=True) else False

        if reached_end_state:
            self.save_graph_path(graph_path, start, end)
        return reached_end_state

    def has_subgraph(self, graph: Graph, template: Graph, include_edges_to_outsiders: bool = False):
        # Organize node IDs by their 'kind'
        nodes_by_kind = defaultdict(set)
        for node in graph.nodes.values():
            nodes_by_kind[node.kind].add(node.key)
        # But count any node kind as 'wildcard'
        all_non_deadend_node_ids = {node.key for node in graph.nodes.values() if not node.kind == DEADEND}  # TODO: this is kind of a patch...
        nodes_by_kind[WILDCARD] = all_non_deadend_node_ids

        # Create some other helper data structures
        template_nodes = list(template.nodes.values())
        template_adj_dict = self.get_adjacency_dict(template)
        template_key_to_index_map = {node.key: index for index, node in enumerate(template_nodes)}
        standardized_template_adj_dict = self.standardize_keys(template_adj_dict, template_key_to_index_map)

        # Create possible combinations of nodes that could be used to fulfill the template (don't check edges yet)
        node_lists = [nodes_by_kind[node.kind] for node in template_nodes]
        combinations_raw = list(itertools.product(*node_lists))
        # Get rid of any that use the same node for more than one template node
        combinations = [combination for combination in combinations_raw if len(set(combination)) == len(combination)]

        # Add edges and figure out if we have any subgraphs matching the template
        matching_subgraphs = []
        for combination_node_keys in combinations:
            combination_graph = Graph(name="--".join(combination_node_keys))

            combination_graph.nodes = {node_key: graph.nodes[node_key] for node_key in combination_node_keys}
            if include_edges_to_outsiders:
                combination_graph.edges = {edge_key: edge for edge_key, edge in graph.edges.items()
                                           if edge.source_key in combination_graph.nodes or
                                           edge.target_key in combination_graph.nodes}
                # Will it be ok leaving these as orphan edges??
            else:
                combination_graph.edges = {edge_key: edge for edge_key, edge in graph.edges.items()
                                           if edge.source_key in combination_graph.nodes and
                                           edge.target_key in combination_graph.nodes}
            combination_graph_adj_dict = self.get_adjacency_dict(combination_graph)
            # Convert to standardized IDs so we can compare to template
            index_to_comb_key_map = {index: node_key for index, node_key in enumerate(combination_node_keys)}
            comb_key_to_index_map = {node_key: index for index, node_key in enumerate(combination_node_keys)}
            template_to_comb_key_map = {template_key: index_to_comb_key_map[index]
                                        for template_key, index in template_key_to_index_map.items()}
            standardized_combination_adj_dict = self.standardize_keys(combination_graph_adj_dict, comb_key_to_index_map)
            if standardized_template_adj_dict == standardized_combination_adj_dict:
                matching_subgraphs.append([combination_graph, template_to_comb_key_map])
        return matching_subgraphs

    @staticmethod
    def get_adjacency_dict(graph: Graph) -> defaultdict:
        adjacency_dict = defaultdict(set)
        for edge in graph.edges.values():
            adjacency_dict[edge.source_key].add(edge.target_key)
        return adjacency_dict

    @staticmethod
    def standardize_keys(adjacency_dict: Dict[str, Set[str]], key_map: Dict[str, int]) -> dict:
        adj_dict_standardized = dict()
        for node_key, neighbors in adjacency_dict.items():
            standardized_node_key = key_map.get(node_key, node_key)
            neighbors_standardized = {key_map.get(neighbor_key, neighbor_key) for neighbor_key in neighbors}
            adj_dict_standardized[standardized_node_key] = neighbors_standardized
        return adj_dict_standardized

    @staticmethod
    def save_graph_path(graph_path: list, start: Graph, end: Graph):
        start.print_fancy()
        end.print_fancy()
        merger = PdfFileMerger()
        for graph, step_name in graph_path:
            graph.print_fancy()
            merger.append(f"{graph.name}.gv.pdf")
        end.print_fancy()
        merger.append(f"end.gv.pdf")
        last_graph = graph_path[-1][0]
        merger.write(f"path--{last_graph.name}.gv.pdf")
        merger.close()

    @staticmethod
    def swap_subgraphs(subgraph_a: Graph, subgraph_b: Graph, larger_graph: Graph,
                       subgraph_a_to_graph_key_map: Dict[str, str]) -> Graph:
        input_nodes_a = [node for node in subgraph_a.nodes.values() if node.input_rank]
        output_nodes_a = [node for node in subgraph_a.nodes.values() if node.output_rank]

        # Initiate a fresh copy of the subgraph that'll be swapped in (so we can modify it)
        subgraph_to_swap_in = copy.deepcopy(subgraph_b)

        # Find corresponding input nodes in b and make the new subgraph uses the original graph's IDs for these
        for input_node_a in input_nodes_a:
            corresponding_input_node_b = next(node for node in subgraph_to_swap_in.nodes.values()
                                              if node.input_rank == input_node_a.input_rank)
            corresponding_graph_key = subgraph_a_to_graph_key_map[input_node_a.key]
            subgraph_to_swap_in.change_node_key(corresponding_input_node_b.key, corresponding_graph_key)
            if subgraph_to_swap_in.nodes[corresponding_graph_key].kind == WILDCARD:
                subgraph_to_swap_in.nodes[corresponding_graph_key].kind = larger_graph.nodes[corresponding_graph_key].kind
        # Find corresponding output nodes and make the new subgraph uses the original graph's IDs for these
        for output_node_a in output_nodes_a:
            corresponding_output_node_b = next(node for node in subgraph_to_swap_in.nodes.values()
                                               if node.output_rank == output_node_a.output_rank)
            corresponding_graph_key = subgraph_a_to_graph_key_map[output_node_a.key]
            subgraph_to_swap_in.change_node_key(corresponding_output_node_b.key, corresponding_graph_key)
            if subgraph_to_swap_in.nodes[corresponding_graph_key].kind == WILDCARD:
                subgraph_to_swap_in.nodes[corresponding_graph_key].kind = larger_graph.nodes[corresponding_graph_key].kind

        # Make sure non input/output nodes in new subgraph have IDs not already used in larger graph
        non_io_node_keys_b = {node.key for node in subgraph_b.nodes.values()
                              if not node.input_rank and not node.output_rank}
        for node_key in non_io_node_keys_b:
            subgraph_to_swap_in.change_node_key(node_key, str(time.time()))

        # Remove non input/output nodes in the subgraph from the larger graph
        new_graph_with_b = copy.deepcopy(larger_graph)
        new_graph_with_b.name = f"{new_graph_with_b.name}-{subgraph_a.name}"
        io_template_node_keys = {node.key for node in input_nodes_a}.union({node.key for node in output_nodes_a})
        non_io_template_node_keys = set(subgraph_a.nodes).difference(io_template_node_keys)
        non_io_graph_keys = {subgraph_a_to_graph_key_map[template_key] for template_key in non_io_template_node_keys}
        for non_io_graph_key in non_io_graph_keys:
            new_graph_with_b.remove_node(non_io_graph_key)

        # Delete input/output nodes from larger graph (but keep their edges) since they're in the new subgraph
        io_graph_keys = {subgraph_a_to_graph_key_map[template_key] for template_key in io_template_node_keys}
        for io_graph_key in io_graph_keys:
            del new_graph_with_b.nodes[io_graph_key]

        # Avoid marking nodes as input/output in the larger graph
        for node in subgraph_to_swap_in.nodes.values():
            node.input_rank = 0
            node.output_rank = 0

        # Add the (now prepped) equivalent graph b into the larger graph
        new_graph_with_b.nodes.update(subgraph_to_swap_in.nodes)
        new_graph_with_b.edges.update(subgraph_to_swap_in.edges)

        return new_graph_with_b

    @staticmethod
    def create_standard_rules():
        # Thank you to Mike for these boilerplate "rules"

        # Rule 1 - Random into G generates two randoms
        rule_1 = Rule("rule_1")
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
        rule_1.graph_b.add_node("$1", RANDOM)
        rule_1.graph_b.add_node("$2", RANDOM)
        rule_1.graph_b.add_node("out1", WILDCARD, output_rank=1)
        rule_1.graph_b.add_node("out2", WILDCARD, output_rank=2)
        rule_1.graph_b.add_edge("$1", "out1")
        rule_1.graph_b.add_edge("$2", "out2")
        rule_1.validate()

        # Rule 2 - OTP thing
        rule_2 = Rule("rule_2")
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
        rule_2.validate()

        # Rule 3 - Rand indistinguishable from rand without replacement
        rule_3 = Rule("rule_3")
        # First graph
        rule_3.graph_a.add_node("$", RANDOM, output_rank=1)
        # Second graph
        rule_3.graph_b.add_node("$", RANDOM_NO_REPLACE, output_rank=1)
        rule_3.validate()

        # Rule 4 - E generates random
        rule_4 = Rule("rule_4")
        # First graph
        rule_4.graph_a.add_node("in", WILDCARD, input_rank=1)
        rule_4.graph_a.add_node("E", E)
        rule_4.graph_a.add_node("out", WILDCARD, output_rank=1)
        rule_4.graph_a.add_edge("in", "E")
        rule_4.graph_a.add_edge("E", "out")
        # Second graph
        rule_4.graph_b.add_node("in", WILDCARD, input_rank=1)
        rule_4.graph_b.add_node("deadend", DEADEND)
        rule_4.graph_b.add_node("$", RANDOM)
        rule_4.graph_b.add_node("out", WILDCARD, output_rank=1)
        rule_4.graph_b.add_edge("$", "out")
        rule_4.graph_b.add_edge("in", "deadend")
        rule_4.validate()

        # Rule 5 - F generates random from random without replacement
        rule_5 = Rule("rule_5")
        # First graph
        rule_5.graph_a.add_node("in", RANDOM_NO_REPLACE, input_rank=1)
        rule_5.graph_a.add_node("F", F)
        rule_5.graph_a.add_node("out", WILDCARD, output_rank=1)
        rule_5.graph_a.add_edge("in", "F")
        rule_5.graph_a.add_edge("F", "out")
        # Second graph
        rule_5.graph_b.add_node("in", RANDOM_NO_REPLACE, input_rank=1)
        rule_5.graph_b.add_node("deadend", DEADEND)
        rule_5.graph_b.add_node("$", RANDOM)
        rule_5.graph_b.add_node("out", WILDCARD, output_rank=1)
        rule_5.graph_b.add_edge("$", "out")
        rule_5.graph_b.add_edge("in", "deadend")
        rule_5.validate()

        # Rule 6 - Basic OTP
        rule_6 = Rule("rule_6")
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
        rule_6.graph_b.add_node("deadend", DEADEND)
        rule_6.graph_b.add_node("$", RANDOM)
        rule_6.graph_b.add_node("out", WILDCARD, output_rank=1)
        rule_6.graph_b.add_edge("$", "out")
        rule_6.graph_b.add_edge("in", "deadend")
        rule_5.validate()

        return [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6]
