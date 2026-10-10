"""Score on a scale: how damaged is this package?"""
from decide import Decider

d = Decider("masterset-ai/PhysicalEye-Decide-35B")
print(d.decide(
    images=["box.jpg"],
    state="Parcel at the receiving dock.",
    question={
        "type": "score",
        "instructions": "How damaged is this package?",
        "criteria": ["no damage", "minor dents", "torn or crushed", "destroyed"],
    },
))
