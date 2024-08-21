import pandas as pd
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Example DataFrame creation from your data
data = pd.read_csv('places_data_new2.csv')

# Combine text fields to form the input for the transformer model
# data['input_text'] = data.apply(lambda x: f"{x['Title']} {x['Short Description']} {
#                                 x['Paragraphs']} {x['Hotels']} {x['Things to do']}", axis=1)

data['input_text'] = (
    data['Title'] + ' ' +
    data['Short Description'] + ' ' +
    data['Paragraphs'] + ' ' +
    data['Hotels'] + ' ' +
    data['Things to do']
)


class PlaceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_text = str(self.data.input_text[index])
        target = self.data.Label[index]

        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        return inputs, target


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


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PlaceDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_len=max_len,
        truncation=True
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128
BATCH_SIZE = 16
N_CLASSES = len(data.Title.unique())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data_loader = create_data_loader(data, tokenizer, MAX_LEN, BATCH_SIZE)
model = TransformerPlaceModel(N_CLASSES)
model = model.to(device)

EPOCHS = 10

optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(EPOCHS):
    model.train()
    for batch in train_data_loader:
        inputs, targets = batch
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        targets = targets.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}')
torch.save(model.state_dict(), 'model.pth')


def suggest_place(description):
    encoding = tokenizer.encode_plus(
        description,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _, prediction = torch.max(outputs, dim=1)

    return data.Title.iloc[prediction.item()]


# Example usage
description = "A beautiful place with mountains and lakes"
suggested_place = suggest_place(description)
print(suggested_place)
