import os
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from openai import OpenAI  # Replace with the actual import for your OpenAI wrapper

# Retrieve the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Instantiate the client with the API key
client = OpenAI(api_key=api_key)

# Define the GPT-3.5 model class
class GPT35Model:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    # This method must return a single string as an answer
    def generate(self, prompt):
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        if completion.choices and completion.choices[0].message:
            return completion.choices[0].message.content.strip()
        return "No response generated."

# Initialize the metric and the model
answer_relevancy_metric = AnswerRelevancyMetric()
model = GPT35Model()

# Define your test prompts and contexts
prompts_and_contexts = [
    ("Who is the current president of the United States of America?",
     ["Joe Biden serves as the current president of America."]),
    ("Where is the Eiffel Tower located?",
     ["The Eiffel Tower is located in Paris, France."]),
    ("What is the capital of Germany?",
     ["Berlin is the capital and largest city of Germany."]),
    ("Who wrote the play 'Romeo and Juliet'?",
     ["William Shakespeare, an English playwrighter, wrote 'Romeo and Juliet'."]),
    ("What is the chemical symbol for water?",
     ["Water is commonly represented by the chemical formula H2O."])
]

# Evaluate each test case using the Answer Relevancy Metric
for i, (prompt, context) in enumerate(prompts_and_contexts, start=1):
    llm_answer = model.generate(prompt)
    test_case = LLMTestCase(
        input=prompt,
        actual_output=llm_answer,
        retrieval_context=context
    )
    answer_relevancy_metric.measure(test_case)
    # Print the test case index, the metric score, and the model's generated response
    print(f"Test Case {i}:")
    print(f"  Input: {test_case.input}")
    print(f"  LLM Response: {test_case.actual_output}")
    print(f"  Relevancy Score: {answer_relevancy_metric.score}")
    print("-" * 50)
