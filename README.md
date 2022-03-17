# hybrid-proof-finder

This is a **proof-of-concept** tool for finding hybrid proofs between specified endpoints using a graph-based
method that is built on top of the visual proof representation being developed by Mike Rosulek.

To run the tool on a provided example, do:

```
pip install -r requirements.txt
python main.py
```

That will create PDFs containing (visual) hybrid proofs for the provided example.

You may define your own rules and start/end points in a fashion similar to that seen in `main.py`.

You may define your own node "kinds", though if they require special handling (like the "wildcard" kind), the tool likely won't work properly with them.

*Note: This proof-of-concept tool has been to designed to work only on simple problems currently, and isn't guaranteed to produce correct proofs (or any proofs), particularly for more complex problems! Its ability to find a proof also depends on the rules provided to it.* 