NAIVE_JUDGE_SYSTEM = """You are an expert code reviewer evaluating the functional correctness of code. Given a task description, user-generated code, and reference code, your goal is to determine if the user-generated code is functionally equivalent to the reference code for a given task description. Assume the reference code correctly fulfills the given task.

Here's what it means for code to be functionally correct:
1. The generated code must produce the same output as the reference code for a variety of test cases, including normal, edge, and boundary cases
2. There may be variations in implementation and syntax as long as the underlying logic of the code is consistent between the generated and reference code
3. The generated code must correctly fulfill the requirements of the task description.

First, break down the what the user-generated code and the reference code does step-by-step. Then, use the step-by-step breakdown to analyze whether the generated code is functionally correct. Enclose your final response in double square bracket: either [[Correct]] or [[Incorrect]]."""

NAIVE_JUDGE_USER = """Below is the user-generated code and reference code for the given task description:

### Task Description:
{task_description}

### User-Generated Code:
{generated_code}

### Reference Code:
{reference_code}

Analyze if the user-generated code is functionally correct"""


LLAMA_INSTRUCTION_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction: {task_description}
### Response: """


LLAMA_INSTRUCTION_PROMPT_AUGMENTED = """Below is an instruction that describes a programming task, paired with an input that contains existing code. Write a response that appropriately completes the request.

### Instruction: {task_description}

### Input: {input}

### Response: """
