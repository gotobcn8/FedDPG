from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn

class RoBERTaClassifier(nn.Module):
    def __init__(self, model_name='roberta-base', num_labels=2, prompt_length=10, pretrained_prompts=None):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.prompt_length = prompt_length
        self.num_layers = self.model.config.num_hidden_layers

        # Freeze the main model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize trainable prompt embeddings for each layer
        self.prompt_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(prompt_length, self.model.config.hidden_size))
            for _ in range(self.num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

        if pretrained_prompts is not None:
            for i, prompt in enumerate(pretrained_prompts):
                self.prompt_embeddings[i].data = prompt.to(self.prompt_embeddings[i].device)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        
        # Create initial input embeddings
        inputs_embeds = self.model.embeddings.word_embeddings(input_ids)
        
        # Adjust attention mask for prompt tokens
        prompt_attention_mask = torch.ones(batch_size, self.prompt_length, device=attention_mask.device)
        extended_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        
        # Prepare the attention mask for the self-attention computation
        extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Process through each layer of RoBERTa
        hidden_states = inputs_embeds
        for i, layer in enumerate(self.model.encoder.layer):
            # Prepend prompt embeddings to the hidden states
            prompt_tokens = self.prompt_embeddings[i].unsqueeze(0).repeat(batch_size, 1, 1)
            layer_input = torch.cat([prompt_tokens, hidden_states], dim=1)
            
            # Apply the layer
            layer_output = layer(layer_input, attention_mask=extended_attention_mask)[0]
            
            # Remove prompt tokens from output (keeping only the actual sequence)
            hidden_states = layer_output[:, self.prompt_length:]

        # Use the [CLS] token representation for classification
        pooled_output = hidden_states[:, 0]
        logits = self.classifier(pooled_output)
        return logits

    def encode(self, input_ids):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
        return logits

    def classify(self, input_ids):
        logits = self.encode(input_ids)
        probs = torch.softmax(logits, dim=1)
        return probs.tolist()

def save_prompts(model, path):
    prompt_state = {
        'prompt_embeddings': [p.detach().cpu() for p in model.prompt_embeddings],
        'config': {
            'num_labels': model.num_labels,
            'prompt_length': model.prompt_length,
            'model_name': model.model.config._name_or_path
        }
    }
    torch.save(prompt_state, path)

def load_prompts(path):
    return torch.load(path)

def initialize_model_with_prompts(prompt_path):
    prompt_state = load_prompts(prompt_path)
    
    model = RoBERTaClassifier(
        model_name=prompt_state['config']['model_name'],
        num_labels=prompt_state['config']['num_labels'],
        prompt_length=prompt_state['config']['prompt_length']
    )
    
    for i, prompt in enumerate(prompt_state['prompt_embeddings']):
        model.prompt_embeddings[i].data = prompt.to(model.prompt_embeddings[i].device)
    
    return model

def inference(model, text):
    model.eval()
    with torch.no_grad():
        inputs = model.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probs = torch.softmax(outputs, dim=1)
    return probs.cpu().numpy()


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = RoBERTaClassifier(num_labels=2).to(device)
    text = "This is an example sentence."
    text = torch.tensor([classifier.tokenizer.encode(text)]).to(device)
    result = classifier.classify(text)
    print(f"Classification probabilities: {result}")
    print(f"Using device: {device}")

    save_prompts(classifier, 'trained_prompts.pt')
    
    # # Usage for inference
    # inference_model = initialize_model_with_prompts('trained_prompts.pt')
    # inference_model.eval()  # Set to evaluation mode

    # # Usage
    # text = "This is a sample text for inference."
    # result = inference(inference_model, text)
    # print(f"Classification probabilities: {result}")
