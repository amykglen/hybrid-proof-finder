from proof_finder import ProofFinder

proof_finder = ProofFinder(use_standard_rules=True)

# Declare some node "kinds" for easy access
WILDCARD = proof_finder.wildcard
RANDOM = proof_finder.random
XOR = proof_finder.xor
F = proof_finder.f
DEADEND = proof_finder.deadend

# Create the graph the proof should start from
start = proof_finder.start
start.add_node("in", WILDCARD)
start.add_node("$", RANDOM)
start.add_node("xor", XOR)
start.add_node("F", F)
start.add_node("out1", WILDCARD)
start.add_node("out2", WILDCARD)
start.add_edge("in", "xor")
start.add_edge("$", "xor")
start.add_edge("xor", "F")
start.add_edge("$", "out1")
start.add_edge("F", "out2")

# Create the graph the proof should end at
end = proof_finder.end
end.add_node("in", WILDCARD)
end.add_node("deadend", DEADEND)
end.add_node("$1", RANDOM)
end.add_node("$2", RANDOM)
end.add_node("out1", WILDCARD)
end.add_node("out2", WILDCARD)
end.add_edge("in", "deadend")
end.add_edge("$1", "out1")
end.add_edge("$2", "out2")

proof_finder.find_proof()
