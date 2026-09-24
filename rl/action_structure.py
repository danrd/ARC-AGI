"""An object action as its parts: type, colours, direction, and how many
objects it takes.

The env's vocabulary is flat - `red_emission_with_blue_object_recolor_N` is
one index among several hundred, and a Linear head over those indices has
one weight per name and nothing shared between them. What the policy learns
about blue emission east says nothing about red emission south, so a
held-out pair that needs a name no training pair used is out of reach
however long the agent trains: on 25d487eb the pairs need blue emission
east, green north and red south, and the test sky north, whose logit never
received a gradient.

The names are generated from parts (rl.utils.define_feasible_actions) and
this takes them apart again, so a policy can choose the parts one after
another - the type, then the objects, then the colours, then the direction
- each from a head shared across every name that has that part. The flat
index is what the env still receives; the parts are the policy's business.

It also says which object slots an action reads. A transform either takes
one object or two (rl.arc_world.World.apply_transform dispatches on whether
the two slots name the same object), and submit takes none: a single-object
action with two different slots, or a two-object one with the same slot
twice, reaches the other branch and does nothing. Choosing the parts lets
the policy never propose those.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from data.configs.env_configs import (ALL_DIRECTIONS, COLORS_MAPPING,
                                      TWO_OBJECTS_ACTION_TYPES)

#: Index meaning "this part is absent" - a type without a colour, or with
#: one colour where others have two, or without a direction.
NO_COLOUR = 10
NO_DIRECTION = len(ALL_DIRECTIONS)
N_COLOURS = 11
N_DIRECTIONS = len(ALL_DIRECTIONS) + 1

_COLOUR_DIGIT = {name: digit for digit, name in COLORS_MAPPING.items() if digit < 10}
_DIRECTION_INDEX = {name: i for i, name in enumerate(ALL_DIRECTIONS)}

#: (row, column) step of each direction, in ALL_DIRECTIONS order - N is one
#: row up.
DIRECTION_STEPS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
                   "NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1)}


def parse_name(name: str) -> Tuple[str, List[int], Optional[str]]:
    """(type, colour digits in the order they appear, direction or None).

    The tokens that are neither a colour word nor a direction are the type,
    in order - which is how define_feasible_actions assembled the name.
    """
    base, colours, direction = [], [], None
    for token in name.split("_"):
        if token in _COLOUR_DIGIT:
            colours.append(_COLOUR_DIGIT[token])
        elif token in _DIRECTION_INDEX:
            direction = token
        else:
            base.append(token)
    return "_".join(base), colours, direction


def objects_taken(action_type: str) -> int:
    """0 for submit, 2 for the transforms World runs on a pair of objects,
    1 for the rest."""
    if action_type == "submit":
        return 0
    return 2 if action_type in TWO_OBJECTS_ACTION_TYPES else 1


class ActionStructure:
    """A flat vocabulary as parts, and the tables that go between the two.

    components[a] = (type, colour, second colour, direction) of flat index
    a, with NO_COLOUR / NO_DIRECTION where the name has no such part;
    flat[t, c1, c2, d] is the index back, -1 where no name has those parts.
    The valid-* tables say which choice is possible given the ones before
    it, so a policy choosing the parts in order is never offered a
    combination the vocabulary does not have.
    """

    def __init__(self, names: Dict[int, str]):
        if sorted(names) != list(range(len(names))):
            raise ValueError("the vocabulary's indices must run 0..n-1 without gaps")
        parsed = [parse_name(names[i]) for i in range(len(names))]
        self.types = sorted({action_type for action_type, _, _ in parsed})
        type_index = {t: i for i, t in enumerate(self.types)}
        self.components = np.zeros((len(names), 4), dtype=np.int64)
        self.flat = np.full((len(self.types), N_COLOURS, N_COLOURS, N_DIRECTIONS), -1,
                            dtype=np.int64)
        for index, (action_type, colours, direction) in enumerate(parsed):
            if len(colours) > 2:
                raise ValueError(f"{names[index]!r} names more than two colours")
            colours = colours + [NO_COLOUR] * (2 - len(colours))
            parts = (type_index[action_type], colours[0], colours[1],
                     NO_DIRECTION if direction is None else _DIRECTION_INDEX[direction])
            if self.flat[parts] != -1:
                raise ValueError(f"{names[index]!r} and {names[self.flat[parts]]!r} "
                                 "have the same parts")
            self.flat[parts] = index
            self.components[index] = parts
        present = self.flat >= 0
        self.valid_colour = present.any(axis=(2, 3))
        self.valid_second_colour = present.any(axis=3)
        self.valid_direction = present
        self.objects = np.array([objects_taken(t) for t in self.types], dtype=np.int64)

    def __len__(self):
        return len(self.components)
