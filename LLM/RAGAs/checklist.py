from checklist.editor import Editor
from checklist.test_types import MFT

# Create an editor
editor = Editor()

# Define test cases and expected answers
test_cases = ["Who discovered penicillin?", "What is the boiling point of water?"]
expected_answers = ["Alexander Fleming", "The boiling point of water is 100 degrees Celsius (or 212 degrees Fahrenheit) under standard atmospheric pressure (1 atm or 101.3 kPa)"]

# Manually provide answers for the test cases
manual_answers = ["mark cuban", "100 degrees Celsius"]  # Replace with your LLM's answers

# Create a functionality test
mft = MFT(test_cases, expected_answers, name="Manual QA Test")

# Run the test
results = []
for idx, question in enumerate(test_cases):
    llm_answer = manual_answers[idx]
    expected = expected_answers[idx]
    results.append((question, llm_answer, expected, llm_answer == expected))

# Display results
print("Test Results:")
for idx, (question, llm_answer, expected, passed) in enumerate(results):
    print(f"Test Case {idx+1}:")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected}")
    print(f"Provided Answer: {llm_answer}")
    print(f"Pass: {passed}")
