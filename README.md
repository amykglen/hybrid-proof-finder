# hybrid-proof-finder

This is a **proof-of-concept** tool for finding hybrid proofs between specified endpoints using a graph-based
method that is built on top of the visual proof representation being developed by Mike Rosulek.

To run the tool on a provided example, do:

```
python main.py
```

That will create PDFs containing (visual) hybrid proofs for the provided example.

You may define your own rules and start/end points in a fashion similar to that seen in `main.py`.

You may define your own node "kinds", though if they require special handling (like the "wildcard" kind), the tool likely won't work properly with them.

Example proof:

