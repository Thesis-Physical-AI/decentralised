"""Semantic map: world entity name -> where it is and how to approach it.

Loaded from the "entities" section of a world file (sim/worlds/*.json). The executor uses it to
turn symbolic step arguments (e.g. movetowards{target: "table"}) into navigation goals, the way a
Nav2-based stack turns them into NavigateToPose goals in the map frame.
"""
import json
import math
from dataclasses import dataclass
from typing import Optional

from .navigation import Pose2D


@dataclass
class Entity:
    name: str
    kind: str  # object | container | door | surface | room (same kinds as the centralized world)
    description: str  # phrase for perception, e.g. "a red apple"
    position: Optional[tuple] = None  # (x, y, z) map frame; None for rooms
    approach: Optional[Pose2D] = None  # where to stand to reach or see it
    parent: Optional[str] = None


class SemanticMap:
    def __init__(self, entities):
        self.entities = {entity.name: entity for entity in entities}

    @classmethod
    def from_world(cls, world):
        entities = []
        for name, spec in world["entities"].items():
            approach = spec.get("approach")
            entities.append(Entity(
                name=name, kind=spec["kind"], description=spec.get("description", name.replace("_", " ")),
                position=tuple(spec["position"]) if "position" in spec else None,
                approach=Pose2D(approach[0], approach[1], math.radians(approach[2])) if approach else None,
                parent=spec.get("in")))
        return cls(entities)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_world(json.load(f))

    def get(self, name):
        return self.entities.get(name)
