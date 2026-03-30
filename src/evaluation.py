from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculates Recall@k."""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    intersection = retrieved_set.intersection(relevant_set)
    if not relevant_set:
        return 0.0
    return len(intersection) / len(relevant_set)

def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculates Mean Reciprocal Rank."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

class LLMJudge:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def evaluate_answer(self, question: str, answer: str, context: str) -> str:
        """
        Evaluates the answer based on relevance and faithfulness using the LLM.
        Returns a score or qualitative assessment.
        """
        messages = [
            {"role": "system", "content": "You are an impartial judge. Evaluate the quality of the answer given the question and the context."},
            {"role": "user", "content": f"""Score the answer from 1 to 5 based on:
1. Faithfulness: Is the answer derived from the context?
2. Relevance: Does the answer address the question?

Provide a short explanation and a final score.

Context:
{context}

Question:
{question}

Answer:
{answer}
"""}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        
        # Slice output to get only the generation
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        evaluation = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return evaluation.strip()

if __name__ == "__main__":
    # Dummy evaluation
    retrieved_chunks = ["The quick brown fox jumps over the lazy dog.", "Random text", "Another chunk"]
    relevant_chunk = "The quick brown fox jumps over the lazy dog."
    
    recall = calculate_recall(retrieved_chunks, [relevant_chunk], k=3)
    mrr = calculate_mrr(retrieved_chunks, [relevant_chunk])
    
    print(f"Recall@3: {recall}")
    print(f"MRR: {mrr}")
    
    # LLM Judge Test
    judge = LLMJudge()
    score = judge.evaluate_answer(
        question="What did the fox do?",
        answer="The fox jumped over the lazy dog.",
        context="The quick brown fox jumps over the lazy dog."
    )
    print(f"\nLLM Judge Evaluation:\n{score}")
