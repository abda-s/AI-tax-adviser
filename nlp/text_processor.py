import google.generativeai as genai
from config.config import load_config

class TextProcessor:
    def __init__(self):
        cfg = load_config()
        # Configure Gemini
        genai.configure(api_key='AIzaSyAuidLUuo5nilcYH4tQWxvcr9LIQKEYIA0')
        # Use gemini-1.5-flash
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def process_text(self, raw_text, question_type):
        """
        Process raw text using Gemini to convert it into the expected format.
        
        Args:
            raw_text (str): The raw text recognized from speech
            question_type (str): The type of question ('yesno' or 'number')
            
        Returns:
            str: Processed text in the expected format
        """
        try:
            if question_type == 'yesno':
                prompt = f"""
                You are a text processor that converts statements into simple yes/no answers.
                Rules:
                1. If the text contains any form of agreement, affirmation, or positive statement, return 'yes'
                2. If the text contains any form of disagreement, negation, or negative statement, return 'no'
                3. Examples:
                   - "yes I am married" -> "yes"
                   - "I am married" -> "yes"
                   - "I have children" -> "yes"
                   - "no I'm not married" -> "no"
                   - "I don't have children" -> "no"
                4. When in doubt, look for positive statements and return 'yes'
                
                Text to process: {raw_text}
                
                Return only 'yes' or 'no'.
                """
            elif question_type == 'number':
                prompt = f"""
                You are a number extractor that converts text into numerical values.
                Rules:
                1. Convert written numbers to digits (e.g., "five" -> "5", "twenty" -> "20")
                2. Extract numbers from text (e.g., "I have five children" -> "5")
                3. For salary/income, convert to numerical value (e.g., "fifty thousand" -> "50000")
                4. For wife salary question:
                   - If the person says they have no income, zero income, or O, return "0"
                   - If they mention a specific amount, return that number
                5. Examples:
                   - "five children" -> "5"
                   - "I have three kids" -> "3"
                   - "my salary is fifty thousand" -> "50000"
                   - "around twenty thousand" -> "20000"
                   - "no income" -> "0"
                   - "zero income" -> "0"
                   - "O" -> "0"
                6. If no clear number is found, return "0"
                
                Text to process: {raw_text}
                
                Return only the number as digits, no additional text.
                """
            else:
                return raw_text

            response = self.model.generate_content(prompt)
            processed_text = response.text.strip().lower()
            
            # Additional validation for numbers
            if question_type == 'number':
                try:
                    # Ensure it's a valid number
                    num = int(processed_text)
                    # Allow 0 for wife salary question
                    if num < 0:  # Handle negative numbers
                        return '0'
                    return str(num)
                except ValueError:
                    return '0'
            
            # Additional validation for yes/no
            if question_type == 'yesno':
                if processed_text == 'true':
                    return 'yes'
                elif processed_text == 'false':
                    return 'no'
                if processed_text not in ['yes', 'no']:
                    # If the model didn't return yes/no, do our own basic check
                    text_lower = raw_text.lower()
                    positive_words = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'fine', 'married', 'have', 'has', 'do', 'does']
                    negative_words = ['no', 'nope', 'nah', 'not', "don't", "doesn't", "haven't", "hasn't"]
                    
                    # Check for positive words first
                    for word in positive_words:
                        if word in text_lower:
                            return 'yes'
                    # Then check for negative words
                    for word in negative_words:
                        if word in text_lower:
                            return 'no'
                    return 'no'  # Default to no if unclear
            
            return processed_text

        except Exception as e:
            print(f"Error processing text with Gemini: {e}")
            # Fallback to basic processing
            if question_type == 'yesno':
                # Do basic text processing
                text_lower = raw_text.lower()
                positive_words = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'fine', 'married', 'have', 'has', 'do', 'does']
                negative_words = ['no', 'nope', 'nah', 'not', "don't", "doesn't", "haven't", "hasn't"]
                
                # Check for positive words first
                for word in positive_words:
                    if word in text_lower:
                        return 'yes'
                # Then check for negative words
                for word in negative_words:
                    if word in text_lower:
                        return 'no'
                return 'no'  # Default to no if unclear
            elif question_type == 'number':
                return '0'
            return raw_text 