import re
from openai import OpenAI
from prompts import NAIVE_JUDGE_SYSTEM, NAIVE_JUDGE_USER

class NaiveJudgeClient:
    """
    Naive zero-shot chain-of-thought judge with self-consistency.

    Requires use of an OpenAI compatible client. Please use secrets/environment variables to securely enter your API key.
    """
    def __init__(self, model_name, api_key, max_len = 512, n_sample = 1):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_len = max_len
        self.n_sample = n_sample

    def evaluate(self, task_description, user_code, reference_code):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "developer", "content": NAIVE_JUDGE_SYSTEM},
                {"role": "user", "content": NAIVE_JUDGE_USER.format(
                    task_description=task_description,
                    generated_code=user_code,
                    reference_code=reference_code)
                }
            ],
            n=self.n_sample, ## self-consistency
            temperature=0.5,
            max_completion_tokens=self.max_len,
        )
        results = []
        for choice in response.choices:
            results.append(extract_score(choice.message.content))
        return max((0,1), key=results.count)


def extract_score(text_response):
    # Regex pattern to find alphabetical text enclosed in double square brackets
    pattern = r'\[\[([a-zA-Z]+)\]\]'

    # Search for the pattern in the text
    match = re.search(pattern, text_response)

    if match:
        # Return the numeric score as an integer
        return 1 if match.group(1) == 'Correct' else 0
    else:
        return 0
