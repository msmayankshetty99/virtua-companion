"""Leaf utilities every feature package may use: messages, stream filters, cancellation, the turn gate, runtime status and
turn context, workers, bounded cleanup, timings, torch device names and background budgets. kernel/ imports nothing else
from app_core (tests/test_package_boundaries.py), and this file imports none of its modules, so each loads on its own."""
