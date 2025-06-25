from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class FedDPG(nn.Module):
    def __init__(self, model_name='roberta-base', num_labels=2, prompt_length=10):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.prompt_length = prompt_length
        
        # Freeze the main model parameters
        for param in self.model.parameters():
            param.requires_grad = False


        self.prompt_generator = nn.Sequential(
            # nn.Linear(self.model.config.hidden_size, 128),
            # nn.ReLU(),
            # nn.Linear(128, 512),
            # nn.ReLU(),
            # nn.Linear(512, self.model.config.hidden_size * prompt_length)
            # nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            # nn.ReLU(),
            # nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size * prompt_length)
            nn.Linear(self.model.config.hidden_size, 10),
            nn.ReLU(),
            nn.Linear(10, self.model.config.hidden_size * prompt_length)
        )

        # Classification head (now trainable)
        self.classifier = nn.Sequential(
            # nn.Linear(self.model.config.hidden_size, 256),
            # nn.ReLU(),
            # nn.Linear(256, num_labels)
            nn.Linear(self.model.config.hidden_size, 10),
            nn.Tanh(),
            nn.Linear(10, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        
        # Generate input embeddings
        inputs_embeds = self.model.embeddings.word_embeddings(input_ids)
        
        # Generate adaptive prompt
        avg_embedding = inputs_embeds.mean(dim=1)  # Average word embeddings
        prompt = self.prompt_generator(avg_embedding)
        prompt = prompt.view(batch_size, self.prompt_length, -1)
        
        # Prepend prompt to input embeddings
        extended_embeds = torch.cat([prompt, inputs_embeds], dim=1)
        
        # Adjust attention mask for prompt tokens
        prompt_attention_mask = torch.ones(batch_size, self.prompt_length, device=attention_mask.device)
        extended_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        
        # Process through RoBERTa
        outputs = self.model(inputs_embeds=extended_embeds, attention_mask=extended_attention_mask)
        
        # Use the [CLS] token representation for classification (after prompts)
        pooled_output = outputs.last_hidden_state[:, self.prompt_length]
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

    def save_state(self, path):
        """Save the model's prompt generator and classifier state."""
        torch.save({
            'prompt_generator': self.prompt_generator.state_dict(),
            'classifier': self.classifier.state_dict()
        }, path)

    def load_state(self, state_dict=None, path=None):
        """Load the model's prompt generator and classifier state."""
        if path:
            state_dict = torch.load(path, weights_only=True)
        
        if state_dict:
            if 'prompt_generator' in state_dict and 'classifier' in state_dict:
                # Load state for prompt generator and classifier
                self.prompt_generator.load_state_dict(state_dict['prompt_generator'])
                self.classifier.load_state_dict(state_dict['classifier'])
            else:
                # Fallback for loading from a flat state dict
                prompt_generator_dict = {k.replace('prompt_generator.', ''): v for k, v in state_dict.items() if k.startswith('prompt_generator.')}
                classifier_dict = {k.replace('classifier.', ''): v for k, v in state_dict.items() if k.startswith('classifier.')}
                
                if prompt_generator_dict:
                    self.prompt_generator.load_state_dict(prompt_generator_dict)
                if classifier_dict:
                    self.classifier.load_state_dict(classifier_dict)
    def total_trainable_parameters(self):
        """Calculate the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# def save_prompts(model, path):
#     prompt_state = {
#         'prompt_embeddings': [p.detach().cpu() for p in model.prompt_embeddings],
#         'config': {
#             'num_labels': model.num_labels,
#             'prompt_length': model.prompt_length,
#             'model_name': model.model.config._name_or_path
#         }
#     }
#     torch.save(prompt_state, path)

# def load_prompts(path):
#     return torch.load(path)

# def initialize_model_with_prompts(prompt_path):
#     prompt_state = load_prompts(prompt_path)
    
#     model = FedDPG(
#         model_name=prompt_state['config']['model_name'],
#         num_labels=prompt_state['config']['num_labels'],
#         prompt_length=prompt_state['config']['prompt_length']
#     )
    
#     for i, prompt in enumerate(prompt_state['prompt_embeddings']):
#         model.prompt_embeddings[i].data = prompt.to(model.prompt_embeddings[i].device)
    
#     return model

# def inference(model, text):
#     model.eval()
#     with torch.no_grad():
#         inputs = model.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs, dim=1)
#     return probs.cpu().numpy()
