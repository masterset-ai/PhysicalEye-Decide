"""Yes or no: is there a visible crack?"""
from decide import Decider

d = Decider("masterset-ai/PhysicalEye-Decide-35B")
print(d.decide(
    images=["part.jpg"],
    state="Steel bracket on a conveyor belt.",
    question={"type": "noul", "instructions": "Is there a visible crack?"},
))
