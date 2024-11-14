from logging import debug, basicConfig, DEBUG
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
basicConfig(level=DEBUG)
debug("Module initialized.")

class gpt:
    def __init__(self):
        self.nm = "openai-community/gpt2-medium"
        self.model = GPT2LMHeadModel.from_pretrained(self.nm)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.nm)

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt', return_token_type_ids=False) # 'pt' for PyTorch
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)
        if self.tokenizer.pad_token_id is not None: attention_mask[inputs == self.tokenizer.pad_token_id] = 0
        outputs = self.model.generate(
            inputs,
            max_length=80,
            repetition_penalty=1.2,
            num_beams=5,
            # temperature=0.3,
            # top_k=50,
            do_sample=True,
            # top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
            attention_mask=attention_mask
        ).ravel()
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

gpt().generate(input("ask me anything. ••>"))
