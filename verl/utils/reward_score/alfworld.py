import re
import random
import unittest

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:")[-1].strip()
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant")[-1].strip()
    else:
        return None

    return solution_str


def validate_format(solution) -> bool:
    """Validate that solution has the correct format. (i.e. <think> </think> <action> </action> tags)"""

    think_start = "<think>"
    think_end = "</think>"
    action_start = "<action>"
    action_end = "</action>"

    think_start_index = solution.find(think_start)
    think_end_index = solution.find(think_end)
    action_start_index = solution.find(action_start)
    action_end_index = solution.find(action_end)

    if think_start_index == -1 or think_end_index == -1 or action_start_index == -1 or action_end_index == -1:
        return False

    if think_start_index > think_end_index or action_start_index > action_end_index:
        return False

    if think_end_index > action_start_index:
        return False

    return True

def validate_action(action: str):
    """
    Validate if the extracted action matches the valid action list.
    """

    valid_actions = [
        r"go to .+?",
        r"take .+? from .+?",
        r"put .+? in/on .+?",
        r"open .+?",
        r"close .+?",
        r"toggle .+? .+?",
        r"clean .+? with .+?",
        r"heat .+? with .+?",
        r"cool .+? with .+?",
        r"use .+?",
        r"inventory",
    ]

    for pattern in valid_actions:
        if re.fullmatch(pattern, action):
            return True

    return False

def evaluate_action(solution_str):
    
    """Evaluate the action whether it is valid or not."""
    answer_pattern = r'<action>(.*?)</action>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        action = matches[-1].group(1).strip()
        # check if action is valid
        if validate_action(action):
            return action
        else:
            return None
    else:
        return None
    
def evaluation_result(action, ground_truth) -> bool:
    # check the correctness of action. Fully match the action with the ground truth.
    return action.strip().lower() == ground_truth.strip().lower()

def compute_score(solution_str, ground_truth, format_score=0.1, valid_action_score=0.2, correctness_score=0.7):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        format_score: the score for correct format
        valid_action_score: the score for valid action
        correctness_score: the score for the correct action
    
    possible scores:
        0.0 for no solution or totally wrong action
        0.1 for only correct format
        0.2 for only valid action
        0.3 for correct format and valid action
        0.9 for valid and correct action (action can be correct only if it is valid)
        1.0 for correct action and format
    """
    score = 0
    solution = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Extracted Model Output: {solution}")

    if solution is None:
        if do_print:
            print(f"No solution found")
        return score
    
    # Validate <think> </think> <action> </action> tags
    if validate_format(solution):
        score += format_score
        
    # Evaluate final action
    action = evaluate_action(solution)
    if action:
        score += valid_action_score
        if evaluation_result(action, ground_truth):
            score += correctness_score
        else:
            if do_print:
                print(f"Wrong action found")
    else:
        if do_print:
            print(f"No valid action found")
    
    return score

class TestComputeScore(unittest.TestCase):
    def test_correct_format_valid_correct_action(self):
        solution_str = """
        Assistant: <think> I need to pick up the key </think> <action> take key from table </action>
        """
        ground_truth = "take key from table"
        self.assertAlmostEqual(compute_score(solution_str, ground_truth), 1.0, places=2)
    
    def test_correct_format_valid_wrong_action(self):
        solution_str = """
        Assistant: <think> I should open the door </think> <action> open window </action>
        """
        ground_truth = "open door"
        self.assertAlmostEqual(compute_score(solution_str, ground_truth), 0.3, places=2)
    
    def test_incorrect_format_valid_wrong_action(self):
        solution_str = """
        Assistant: I should open the door <action> open window </action>
        """
        ground_truth = "open door"
        self.assertAlmostEqual(compute_score(solution_str, ground_truth), 0.2, places=2)
    
    def test_correct_format_invalid_action(self):
        solution_str = """
        Assistant: <think> I should move forward </think> <action> jump over table </action>
        """
        ground_truth = "move forward"
        self.assertAlmostEqual(compute_score(solution_str, ground_truth), 0.1, places=2)
    
    def test_incorrect_format_valid_correct_action(self):
        solution_str = """
        Assistant: I should take the key. <action> take key from table </action>
        """
        ground_truth = "take key from table"
        self.assertAlmostEqual(compute_score(solution_str, ground_truth), 0.9, places=2)
    
    def test_incorrect_format_invalid_action(self):
        solution_str = """
        Assistant: Just do it. <action> fly to the roof </action>
        """
        ground_truth = "climb the ladder"
        self.assertAlmostEqual(compute_score(solution_str, ground_truth), 0.0, places=2)
    
    def test_no_solution(self):
        solution_str = """Assistant: I don't know what to do."""
        ground_truth = "take key from table"
        self.assertAlmostEqual(compute_score(solution_str, ground_truth), 0.0, places=2)

if __name__ == "__main__":
    unittest.main()
