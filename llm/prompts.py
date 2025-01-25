import copy
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import List
from rl.ARC_task import ARCTask
from symbolic.utils import find_upper_left_corner

DETAILED_PROMPT =  ["general_instruction", "grid_description", "task_instruction", "examples_repr", "task_repr", "output_format"]  
BASE_PROMPT = ["general_instruction", "grid_description", "examples_repr", "task_repr", "output_format"]  
CONCISE_PROMPT = ["general_instruction", "examples_repr", "task_repr", "output_format"] 

COLOR_MAPPING = "0=black, 1=blue, 2=red, 3=green, " \
                "4=yellow, 5=gray, 6=magenta, " \
                "7=orange, 8=sky, 9=brown"

GENERAL_INSTRUCTION = f"""
You are a helpful AI assistant. Your job is to solve tasks from the Abstraction and Reasoning Challenge (ARC).
The challenge involves identifying the next image in a sequence, similar to Raven's progressive matrices.
The user will present you with sample input and output grids for each task. 
Your job will be to understand the transformation rules between the input and the output and apply them to the last input grid given by the user.
"""

GRID_DESCRIPTION = f"""
The puzzle-like inputs and outputs present a grid with height and width between 1 and 30 where each cell can be one of ten colors.  
Here is colors representation: {COLOR_MAPPING}.
Any color could be a background color. 
Groups of identically colored cells form objects: lines, rectangles, diagonals and tetris-like shapes (e.g. L-shape).
Each object has its own color, size and specific position on the grid.
"""

CLAUDE_INSTRUCTION = f"""
You are an advanced pattern recognition AI specialized in solving abstract reasoning challenges. Your primary objective is to:
- Decompose the grid into distinct object groups
- Analyze spatial, color, and structural transformations
- Identify invariant and variant properties
- Generalize a minimal set of rules explaining the transformation
- Predict the most likely output grid based on learned rules
Reasoning steps:
1. Object Identification
   - Segment grid into color-based objects
   - Note object boundaries, shapes, and positions
2. Transformation Analysis
   - Compare input and output grid differences
   - Classify transformation types:
     * Color changes
     * Position shifts
     * Shape modifications
     * Symmetry/asymmetry patterns
3. Rule Generalization
   - Formulate a concise set of transformation rules
   - Validate rules against all provided examples
   - Apply rules to predict the output grid
"""

TASK_INSTRUCTION = f"""
Firstly, the most important thing for task solving is to compare each input and output grid pairs.
Based on that you can deduce that task implies size change. 
If the size decreased then it indicates possibility of generalization, sections overlay or cropping types of tasks.
Generalizion requires some kind of convolution based on the pattern with mapping of larger figures to smaller.
Section overlay requires some kind of mask operation when combining these parts, e.g. XOR.
The output for cropping task type is just some part of input grid.\n
If the size increased it indicates possibility of extrapolation patterns - expanding object according some pattern or by object muiltiplication.\n
If the size is unchanged - you may need some objects recoloring, shape modification or line emission - when line comes from object to connect with other object or grid edge.\n 
Secondly, grid comparison helps to identify the same objects on the grid and thus possibly operations performed on them.
Each object can be shifted, rotated, recolored or partly modified. Each of these modifications can be part of general transformation pattern.
At the same time it helps to identify the only changed part that is important for coloring patterns.\n
Thirdly, some tasks use markup to distinguish some areas on the grid. Such markup can create matrix like separation of the grid, bound example for its extrapolation or split grid into several parts.
If you identify example bounded by markup - try to use it for transformation: take into account its shape and color. 
If you identified that grid is splited into several parts - you may need some kind of mask operation on these parts, e.g. XOR. It also implies decrease of output grid size.\n
Fourthly, for some tasks specific geometry notions are needed.
If objects lies in line - they can be connected. 
Each colored cell can be connected by line with edge or corner of the grid.
If some object lies on the way of line expansion - line can change direction or be reflected.
There is also some king of gravity notion: object can be pulled to other object based on color or distance or to grid egde.\n
And finally, its important to count a number of objects for each shape, color and size.
The reason is that you may need in some tasks to choose the most frequent object or to color other object with dominant color.\n 
"""

EXAMPLES_TEMPLATE = f"""
Here are the example input and output pairs from which you should learn the underlying transformation 
to later predict the output for the given test input:\n
"""

TASK_REPR = f"""Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data: """

HINTS = f"""
Most probably you need to deal with font coloring type of puzzle. Thus, take into account follwing recommendations for task solving:
1) Compare input and output grids from examples to identify what shape and color are important for the tast. 2) Find shape for identified pattern on the task input grid.
3) Color the shape with identified color. Most probably the output grid will have the same shape as the task inpur grid.
"""

OUTPUT_FORMAT = f"""
Return only output grid in the same format as examples without any explanation.
"""

def find_upper_left_corner(grid_size:tuple)->tuple:
    """Finds left upper corner of the grid to take into account padding."""
    i = min(14-(grid_size[0]%2)*((grid_size[0]//2)), 14-((grid_size[0]-1)%2)*(((grid_size[0]-1)//2)))
    j = min(14-(grid_size[1]%2)*((grid_size[1]//2)), 14-((grid_size[1]-1)%2)*(((grid_size[1]-1)//2)))
    return (i, j)

def get_propmt_for_examples(task:ARCTask, tokenizer, tokens_number:int, 
                            max_tokens:int, grid_repr_type:str='ascii')->str:
    """Get representation for grids from examples."""
    examples = ""
    tokens_number = tokens_number
    for idx, subtask in enumerate(task.subtasks):
        inp_grid = prepare_grid_for_prompt(subtask.train_inp, subtask.train_inp_shape, grid_repr_type)
        out_grid = prepare_grid_for_prompt(subtask.train_out, subtask.train_out_shape, grid_repr_type)
        example = f'Example {idx+1}:\n Input: {inp_grid}\n Output: {out_grid}\n'
        tokens_number += len(tokenizer.tokenize(example)) + 10
        if tokens_number < max_tokens:
            examples += example
        elif tokens_number > max_tokens and idx <= 1:
            return False
        else:
            break
    return examples

def prepare_grid_for_prompt(grid:np.array, shape:tuple, grid_repr_type:str='concise')->str:
    """Get representation for grid one grid from examples."""
    ul = find_upper_left_corner(shape)
    grid = copy.copy(grid[ul[0]:ul[0]+shape[0], ul[1]:ul[1]+shape[1]]*10)
    if grid_repr_type=='concise':
        return concise_grid_representation(grid)
    elif grid_repr_type=='ascii':
        return ascii_grid_representation(grid)
    else:
        return repr(grid.astype(int))
    
def concise_grid_representation(grid:np.array):
    """Concise representation for numpy array without brackets and commas."""
    repr = f'grid shape: {grid.shape[0]}x{grid.shape[1]}\n'
    for i in range(grid.shape[0]):
        repr += f'{i+1} '
        for j in range(grid.shape[1]):
            repr += f'{int(grid[i][j])}'
        repr += '\n'
    return repr

def ascii_grid_representation(grid:np.array):
    """ASCII representation for numpy array without brackets and commas."""
    repr = f'grid shape: {grid.shape[0]}x{grid.shape[1]}\n'
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            repr += f'{int(grid[i][j])}'
            if j != grid.shape[1]-1:
               repr += '|' 
        repr += '\n'
    return repr

def examples_representation(task:ARCTask, tokenizer, tokens_number:int, 
                            max_tokens, prompts_modifications:dict,
                            grid_repr_type:str='ascii')->str:
    """Get representation for grids from examples."""
    global EXAMPLES_TEMPLATE
    if "examples_repr" in prompts_modifications.keys():
        EXAMPLES_TEMPLATE = prompts_modifications["examples_repr"]
    tokens_number += len(tokenizer.tokenize(EXAMPLES_TEMPLATE))
    examples_repr = get_propmt_for_examples(task, tokenizer, tokens_number, 
                                            max_tokens, grid_repr_type)
    if examples_repr:
        return EXAMPLES_TEMPLATE + examples_repr
    else:
        return False

def task_representation(task:ARCTask, prompts_modifications:dict, grid_repr_type:str='ascii')->str:
    """Get representation for test grid."""
    global TASK_REPR
    if "task_repr" in prompts_modifications.keys():
        TASK_REPR = prompts_modifications["task_repr"]
    return TASK_REPR + repr(prepare_grid_for_prompt(task.test_subtask.train_inp, task.test_subtask.train_inp_shape, grid_repr_type))

def compose_prompt(task:ARCTask, prompt_structure:List, 
                   prompts_modifications:dict, tokenizer, 
                   max_tokens, grid_repr_type:str='ascii',
                   train_example:bool=True)->str:
    """Compose prompts according to defined prompt structure."""
    global GENERAL_INSTRUCTION
    global GRID_DESCRIPTION
    global TASK_INSTRUCTION
    global OUTPUT_FORMAT
    final_prompt = ""
    tokens_number = 0
    examples_repr = False
    if "general_instruction" in prompt_structure:
        if "general_instruction" in prompts_modifications.keys():
            general_instruction =  prompts_modifications["general_instruction"]
        general_instruction = f'[INSTRUCTION]{GENERAL_INSTRUCTION}[/INSTRUCTION]\n'
        tokens_number += len(tokenizer.tokenize(general_instruction))
    if "grid_description" in prompt_structure:
        if "grid_description" in prompts_modifications.keys():
            grid_description =  prompts_modifications["grid_description"]
        grid_description = f'[GRID_DESCRIPTION]{GRID_DESCRIPTION}[/GRID_DESCRIPTION]\n'
        tokens_number += len(tokenizer.tokenize(grid_description))
    if "task_instruction" in prompt_structure:
        if "task_instruction" in prompts_modifications.keys():
            task_instruction =  prompts_modifications["task_instruction"]
        task_instruction = f'[TASK_INSTRUCTION]{TASK_INSTRUCTION}[/TASK_INSTRUCTION]\n'
        tokens_number += len(tokenizer.tokenize(task_instruction))
    if "task_repr" in prompt_structure:
        task_repr = f'[TASK]{task_representation(task, prompts_modifications, grid_repr_type)}[/TASK]\n'
        tokens_number += len(tokenizer.tokenize(task_repr))
    if "output_format" in prompt_structure:
        if "output_format" in prompts_modifications.keys():
            output_format =  prompts_modifications["output_format"]
        output_format = f'[FORMAT]{OUTPUT_FORMAT}[/FORMAT] Answer: '
        tokens_number += len(tokenizer.tokenize(output_format))
    if train_example:
      test_out_repr = prepare_grid_for_prompt(task.test_subtask.train_out, task.test_subtask.train_out_shape, grid_repr_type)
      tokens_number += len(tokenizer.tokenize(test_out_repr))
    if "examples_repr" in prompt_structure:
        examples = examples_representation(task, tokenizer, tokens_number, max_tokens, prompts_modifications, grid_repr_type)
        if examples:
            examples_repr = f'[EXAMPLES]{examples}[/EXAMPLES]\n'
        else:
           examples_repr = False 
    if examples_repr:
        for task_element in prompt_structure:
            final_prompt += locals()[task_element]
        if train_example:
           final_prompt += test_out_repr
        return final_prompt
    else:
        return False
