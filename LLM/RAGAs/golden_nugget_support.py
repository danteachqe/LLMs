from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.metrics import Metric

class GoldenNuggetRecall(Metric):
    """
    Custom metric to measure the recall of golden nuggets in RAG systems.
    """
    name = "golden_nugget_recall"

    @property
    def required_fields(self):
        return ["retrieved_chunks", "golden_nugget_ids"]

    def __init__(self):
        super().__init__(
            name="golden_nugget_recall"
        )

    def init(self, *args, **kwargs):
        """Placeholder implementation for the required init method."""
        pass

    def _score(self, samples):
        """
        Compute the recall of golden nuggets.

        Args:
            samples (dict): A dictionary containing retrieved chunks and golden nugget IDs.

        Returns:
            float: The average recall of golden nuggets across all samples.
        """
        recalls = []
        for retrieved, nuggets in zip(samples["retrieved_chunks"], samples["golden_nugget_ids"]):
            if not nuggets:
                recalls.append(1.0)  # No nuggets means recall=1 by default
                continue
            retrieved_set = set(retrieved)
            nugget_set = set(nuggets)
            recall = len(retrieved_set.intersection(nugget_set)) / len(nugget_set)
            recalls.append(recall)
        return sum(recalls) / len(recalls)

class AverageSupportScore(Metric):
    """
    Custom metric to compute the average support score for retrieved chunks.
    """
    name = "average_support_score"

    @property
    def required_fields(self):
        return ["support_scores"]

    def init(self, *args, **kwargs):
        """Placeholder implementation for the required init method."""
        pass

    def _score(self, samples):
        """
        Compute the average support score.

        Args:
            samples (dict): A dictionary containing support scores for retrieved chunks.

        Returns:
            float: The average support score across all samples.
        """
        averages = [sum(scores) / len(scores) if scores else 0 for scores in samples["support_scores"]]
        return sum(averages) / len(averages)

# Example usage
if __name__ == "__main__":
    # Example dataset
    dataset = {
        "retrieved_chunks": [["chunk1", "chunk2"], ["chunk3"]],
        "golden_nugget_ids": [["chunk1"], ["chunk4"]],
        "support_scores": [[0.9, 0.8], [0.7]]
    }

    # Golden Nugget Recall
    golden_nugget_metric = GoldenNuggetRecall()
    recall_score = golden_nugget_metric._score(dataset)
    print(f"Golden Nugget Recall: {recall_score}")

    # Average Support Score
    support_score_metric = AverageSupportScore()
    avg_support_score = support_score_metric._score(dataset)
    print(f"Average Support Score: {avg_support_score}")
