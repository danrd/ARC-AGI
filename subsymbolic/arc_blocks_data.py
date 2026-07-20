"""ARC project prompt data that isn't itself a template: per-role instruction
snippets, fed into `role_instruction/v1.j2` via context["role_text"].

Everything that's pure static prose (general instruction, hints, output
format, coordinator/decision instructions, validation criteria, ...) now
lives directly in the corresponding `.j2` file under data/prompts/ instead
of as a Python string constant — that's the whole point of moving to
templates: editing the wording means editing the template, not this file.
"""

# NOTE: the original `available_agents` list (in the legacy prompt module)
# had Mixer's purpose copy-pasted from Modifier ("Transform object
# coloration through pattern recognition") instead of describing its actual
# role (grid-section interaction). Fixed here.
ROLE_INSTRUCTIONS = {
    "Modifier": (
        "Role: Transform object coloration through pattern recognition\n"
        "Focus: Map visual/spatial patterns to color transformations\n"
        "Core Process: Detect patterns -> Extract coloring logic -> Apply to target objects\n"
        "Constraint: Preserve all structural properties; only modify colors"
    ),
    "Highlighter": (
        "Role: Extract key elements from the grid\n"
        "Focus: Identify and return relevant objects or grid segments based on their\n"
        "properties or relations to other objects or parts of the grid\n"
        "Key Actions: Detect selection criteria -> Select target parts -> Return selection\n"
        "or filter out irrelevant parts"
    ),
    "Connector": (
        "Role: Create links, paths, or emanations between objects and grid boundaries\n"
        "Focus: Establish spatial connections using lines, rays, or emission patterns\n"
        "Key Actions: Identify connection points -> Generate connecting elements -> Extend\n"
        "from centers or edges"
    ),
    "Constructor": (
        "Role: Create new objects by coloring background cells or multiplying patterns\n"
        "Focus: Add elements to the grid through strategic placement and pattern generation\n"
        "Key Actions: Identify placement locations -> Apply creation rules -> Generate new\n"
        "structures systematically"
    ),
    "Shifter": (
        "Role: Relocate or merge objects based on spatial rules, gravity, or positional\n"
        "relationships\n"
        "Focus: Move objects while maintaining relational positioning constraints\n"
        "Key Actions: Calculate target positions -> Apply movement rules -> Preserve relative\n"
        "arrangements"
    ),
    "Extrapolator": (
        "Role: Expand grid size by continuing patterns or upscaling existing structures\n"
        "Focus: Enlarge the output grid through pattern repetition or magnification\n"
        "Key Actions: Identify expandable patterns -> Apply scaling rules -> Fill the\n"
        "enlarged grid systematically"
    ),
    "Generalizer": (
        "Role: Abstract grid patterns through symbolic compression\n"
        "Focus: Preserve key relations while reducing structural redundancy\n"
        "Key Actions: Identify invariant motifs -> Apply compression or convolution rules ->\n"
        "Output a minimal representation that retains essential structure"
    ),
    "Mapper": (
        "Role: Establish correspondences between objects across examples\n"
        "Focus: Find consistent transformation rules that apply to all input-output pairs\n"
        "Key Actions: Compare object sets -> Identify mapping patterns -> Apply\n"
        "transformations systematically"
    ),
    "Mixer": (
        "Role: Perform grid section interaction\n"
        "Focus: Track the order of operations on sections and consider different\n"
        "transformations\n"
        "Key Actions: Retrieve grid segments -> Identify section interaction type (logic\n"
        "operations or color summation) -> Find a sequence of operations valid for all\n"
        "examples"
    ),
}
