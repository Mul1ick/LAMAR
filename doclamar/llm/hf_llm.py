import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .base import BaseLLM


class HFLocalLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    def generate(self, prompt: str) -> str:
        outputs = self.generator(prompt)
        text = outputs[0]["generated_text"]

        # Remove the prompt from output if echoed
        if text.startswith(prompt):
            text = text[len(prompt):]

        return text.strip()
