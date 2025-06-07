# nlp/answer_normalizer.py

def normalize(raw: str, expected_type: str):
    """
    raw: e.g. 'yes', 'No', '3', 'three'
    expected_type: 'yesno' or 'number'
    returns: bool or int
    """
    if not raw:
        return None
    text = raw.strip().lower()
    if expected_type == 'yesno':
        if text in ['yes', 'y', 'true']:
            return 'yes'
        if text in ['no', 'n', 'false']:
            return 'no'
        return None
    if expected_type == 'number':
        # try direct int
        try:
            return int(text)
        except ValueError:
            # map words to digits simple map
            word2num = {
                'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
            }
            return word2num.get(text)
    return None