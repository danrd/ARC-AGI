"""
Agents aggregated configurations containing:
- ARC project prompt data that isn't itself a template: per-role instruction
snippets, fed into `role_instruction/v1.j2` via context["role_text"];
- Modules general specification;
- Agents specification setting available modules depending on agent's type.
"""

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

MODULES = {
    'Symbolic': {
        'index': 1,
        'name': 'Symbolic',
        'purpose': 'Algorithmic pattern expansion.',
    },
    'Subsymbolic': {
        'index': 2,
        'name': 'Subsymbolic',
        'purpose': 'LLM-based pattern extrapolation.',
    },
    'Interactive': {
        'index': 3,
        'name': 'Interactive',
        'purpose': 'Interactive grid manipulation.',
    }
}

AGENTS_REGISTRY = [
    {
        'index': 1,
        'name': 'Generalizer',
        'purpose': 'Abstract grid patterns through symbolic compression.',
        'modules': ['Subsymbolic'],
        'available_modules': [MODULES['Subsymbolic']]
    },
    {
        'index': 2,
        'name': 'Extrapolator',
        'purpose': 'Expand grid size by continuing patterns or upscaling.',
        'modules': ['Symbolic', 'Subsymbolic'],
        'available_modules': [MODULES['Symbolic'], MODULES['Subsymbolic']]
    },
    {
        'index': 3,
        'name': 'Mapper',
        'purpose': 'Establish correspondences between objects across examples.',
        'modules': ['Subsymbolic'],
        'available_modules': [MODULES['Subsymbolic']]
    },
    {
        'index': 4,
        'name': 'Constructor',
        'purpose': 'Create new objects by coloring background cells or multiplying patterns.',
        'modules': ['Symbolic', 'Subsymbolic', 'Interactive'],
        'available_modules': [MODULES['Symbolic'], MODULES['Subsymbolic'], MODULES['Interactive']]
    },
    {
        'index': 5,
        'name': 'Shifter',
        'purpose': 'Relocate or merge objects based on spatial rules, gravity, or positional relationships.',
        'modules': ['Subsymbolic', 'Interactive'],   
        'available_modules': [MODULES['Subsymbolic'], MODULES['Interactive']]
    },
    {
        'index': 6,
        'name': 'Highlighter',
        'purpose': 'Extract key elements from the grid.',
        'modules': ['Subsymbolic', 'Interactive'],
        'available_modules': [MODULES['Subsymbolic'], MODULES['Interactive']]
    },
    {
        'index': 7,
        'name': 'Connector',
        'purpose': 'Create links, paths, or emanations between objects and grid boundaries.',
        'modules': ['Interactive'],
        'available_modules': [MODULES['Interactive']]
    },
    {
        'index': 8,
        'name': 'Modifier',
        'purpose': 'Transform object coloration through pattern recognition.',
        'modules': ['Subsymbolic', 'Interactive'],
        'available_modules': [MODULES['Subsymbolic'], MODULES['Interactive']]
    },
    {
        'index': 9,
        'name': 'Mixer',
        'purpose': 'Transform object coloration through pattern recognition.',
        'modules': ['Symbolic'],
        'available_modules': [MODULES['Symbolic']]
    }
]
