"""Inspect a folder of images and route uncertain ones to a person."""
import sys
from pathlib import Path
from decide import Decider

THRESHOLD = 0.8
d = Decider("masterset-ai/PhysicalEye-Decide-35B")
question = {"type": "choice", "instructions": "Is this egg good or bad?",
            "criteria": {"good": "Normal, healthy egg", "bad": "Defective, spoiled or damaged egg"}}

for path in sorted(Path(sys.argv[1]).glob("*.jpg")):
    r = d.decide(question, images=[str(path)], state="Candling photo on the grading line.")
    route = "AUTO" if r["confidence"] >= THRESHOLD else "REVIEW"
    print(f"{path.name}\t{r['choice']}\t{r['confidence']:.2f}\t{route}")
