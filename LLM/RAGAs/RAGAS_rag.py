from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

# Sample data with required columns
data = {
    'question': [
        'What is the capital of France?',
        'Who wrote "To Kill a Mockingbird"?',
        'What is the chemical symbol for water?',
        'Who painted the Mona Lisa?',
        'What is the largest planet in our solar system?',
        'What year did the Titanic sink?',
        'What is the square root of 64?',
        'Who discovered penicillin?',
        'What is the capital of Japan?',
        'What is the speed of light in a vacuum?',
        'Who is known as the father of modern physics?'
    ],
    'contexts': [
        ['Paris is a major city in France.'],  # Slightly less precise
        ['The book "To Kill a Mockingbird" is famous.'],  # Vague
        ['H2O is a chemical formula.'],  # Less specific
        ['The Mona Lisa is a renowned painting.'],  # Missing creator
        ['Jupiter is a planet in the solar system.'],  # Missing largest
        ['The Titanic sank in the early 1900s.'],  # Less precise
        ['The square root of 64 is a number.'],  # Vague
        ['Penicillin was discovered in the 20th century.'],  # Missing name
        ['Tokyo is a city in Japan.'],  # Less descriptive
        ['Light travels very fast in a vacuum.'],  # Missing exact speed
        ['Einstein contributed to physics.']  # Less specific
    ],
    'answer': [
        'The capital of France is Paris.',
        'Harper Lee wrote "To Kill a Mockingbird".',
        'The chemical symbol for water is H2O.',
        'The Mona Lisa was painted by Leonardo da Vinci.',
        'The largest planet in our solar system is Jupiter.',
        'The Titanic sank in 1912.',
        'The square root of 64 is 8.',
        'Penicillin was discovered by Alexander Fleming.',
        'The capital of Japan is Tokyo.',
        'The speed of light in a vacuum is approximately 299,792 kilometers per second.',
        'Albert Einstein is known as the father of modern physics.'
    ],
    'ground_truth': [
        'The capital city of France, known for its art, fashion, and culture, is Paris.',
        'The novel "To Kill a Mockingbird" was authored by Harper Lee.',
        'Water is represented by the chemical formula H2O.',
        'The famous painting Mona Lisa was created by Leonardo da Vinci.',
        'Jupiter, the largest planet in the solar system, is a gas giant.',
        'The Titanic, a British passenger liner, sank in the year 1912.',
        'The mathematical square root of 64 is 8.',
        'Alexander Fleming is credited with the discovery of penicillin.',
        'Tokyo, a bustling metropolis, serves as the capital of Japan.',
        'Light travels at a speed of approximately 299,792 kilometers per second in a vacuum.',
        'Albert Einstein, a theoretical physicist, is regarded as the father of modern physics.'
    ]
}
df = pd.DataFrame(data)

# Convert the DataFrame to a Dataset
dataset = Dataset.from_pandas(df)

# Define metrics
metrics = [context_precision, context_recall]

# Evaluate the RAG system
results = evaluate(dataset, metrics)

# Display results
print(results)
