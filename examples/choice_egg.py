"""Pick one option: is this egg good or bad?"""
from decide import Decider

d = Decider("masterset-ai/PhysicalEye-Decide-35B")
print(d.decide(
    images=["egg.jpg"],
    state="Candling photo: a light is shone through the egg in a dark room.",
    question={
        "type": "choice",
        "instructions": "Is this egg good or bad?",
        "criteria": {"good": "Normal, healthy egg", "bad": "Defective, spoiled or damaged egg"},
    },
))
