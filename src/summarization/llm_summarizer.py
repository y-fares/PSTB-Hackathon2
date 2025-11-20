from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import LLM_MODEL_NAME


class LLMSummarizer:
    """
    Summarizer that uses an open-source LLM (TinyLlama) to answer
    a question based on retrieved document chunks.
    """

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        """
        Load the tokenizer and model for TinyLlama.

        We use CPU here. It will be slower than GPU, but fine for short summaries.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to("cpu")  # ensure we run on CPU

    def _build_prompt(self, query: str, chunks: List[str]) -> str:
        """
        Build a simple instruction prompt for the LLM.

        We give the question and the context, and ask for a concise answer.
        """
        if not chunks:
            return (
                "You are an assistant. There is no context available. "
                "Tell the user that no answer can be given."
            )

        # Join top chunks into a context section.
        context = "\n\n".join(chunks)

        # Very simple instruction-style prompt.
        prompt = (
            "You are an AI assistant that answers questions based only on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Instructions:\n"
            "- Answer the question using ONLY the information from the context.\n"
            "- Be concise (2 or 3 sentences maximum).\n" #was 3/5
            "- Do not invent facts.\n\n"
            "Answer:\n"
        )
        return prompt

    def summarize_with_llm(self, query: str, chunks: List[str]) -> str:
        """
        Use the LLM to produce an answer + summary from the retrieved chunks.

        We:
        - build a prompt from the query and chunks
        - generate a short answer using the model
        - decode and return it as a string
        """
        prompt = self._build_prompt(query, chunks)

        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # keep prompt size under control # was 2048
        )

        # Move tensors to CPU (should already be, but we do it explicitly)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        # Generate the output
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=96,   # how long the answer can be  # initially 256 – this makes it much faster
                do_sample=False,      # deterministic output
                #temperature=0.0,      # no randomness
            )

        # We only want the generated part, not the full prompt again.
        # Some models include the prompt in the output, so we slice.
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Post-process the output:
        # If the model repeats instructions, we keep only what comes AFTER the last "Answer:" marker.
        text = raw_output.strip()
        marker = "Answer:"
        if marker in text:
            # Split on the marker and take the last part
            text = text.split(marker)[-1].strip()

        # Final cleanup
        if not text:
            text = "The model did not generate an answer."

        return text

        # Clean up whitespace
        answer = answer.strip()
        if not answer:
            answer = "The model did not generate an answer."

        return answer