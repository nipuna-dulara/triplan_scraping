import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch.nn as nn
import pandas as pd
data = pd.read_csv('places_data_new2.csv')


class TransformerPlaceModel(nn.Module):
    def __init__(self, n_classes):
        super(TransformerPlaceModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(outputs.pooler_output)
        return self.fc(output)


N_CLASSES = len(data.Title.unique())

# Assuming 'ModelClass' is the class definition of your model
model = TransformerPlaceModel(N_CLASSES)  # Initialize the model architecture
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode
MAX_LEN = 128
BATCH_SIZE = 16
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def suggest_places(description, num_suggestions=3):
    encoding = tokenizer.encode_plus(
        description,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = nn.Softmax(dim=1)(outputs)

    # Get the top N predictions
    top_n_probs, top_n_indices = torch.topk(probs, num_suggestions, dim=1)

    # Get the titles corresponding to the top N predictions
    suggestions = data.Title.iloc[top_n_indices.squeeze().tolist()]

    return suggestions, top_n_probs


# Example usage
description = " Mirissa, nestled along the southern coast of Sri Lanka, is a picturesque and laid-back beach town that attracts travelers seeking sun, sand, and serenity. What was a faint old beach town has now risen to be one of the most popular surfing and"
suggested_places = suggest_places(description, num_suggestions=4)
print(suggested_places)
