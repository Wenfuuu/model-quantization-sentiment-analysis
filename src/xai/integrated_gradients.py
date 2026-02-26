import torch
from captum.attr import IntegratedGradients


class IntegratedGradientsExplainer:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    def explain(self, text, target=None, steps=30):
        encoding = self.tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted = int(torch.argmax(logits, dim=-1).item())

        target_class = predicted if target is None else target

        embeddings = self.model.bert.embeddings.word_embeddings(input_ids)

        def forward_fn(embeds):
            return self.model(inputs_embeds=embeds, attention_mask=attention_mask).logits

        ig = IntegratedGradients(forward_fn)
        attributions = ig.attribute(
            embeddings,
            target=target_class,
            n_steps=steps,
            baselines=torch.zeros_like(embeddings),
        )

        scores = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

        return {
            "tokens": tokens,
            "scores": scores,
            "predicted_class": predicted,
        }
