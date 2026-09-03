"""The one place ARC's colour numbers are given names.

analyzer.py and findings.py both need to say "black" rather than "0", and
a second copy of this mapping is a second thing to get out of step - the
number is what everything else passes around, so a disagreement here would
show up only as two parts of one summary calling the same colour different
things.
"""

COLORS_MAPPING = {
    0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
    5: 'gray', 6: 'magenta', 7: 'orange', 8: 'sky', 9: 'brown',
}
