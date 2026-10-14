# Official parse_response from facebookresearch/CRAG local_evaluation.py
# Source: https://github.com/facebookresearch/CRAG/blob/main/local_evaluation.py
import re


def parse_response(response: str):
    """
    Return a tuple of (explanation, score) from the response,
    where score is 0 if the prediction is wrong, 1 if the prediction is correct.
    """
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1
