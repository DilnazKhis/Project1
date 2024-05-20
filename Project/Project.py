
import json
import numpy as np 
import pandas as pd 
import torch
import torch.cuda
import torchtext
from typing import Any
import tqdm
import os
from sklearn.model_selection import train_test_split

def parse_data(file):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

file_path = r'C:\Users\khism\Downloads\Telegram Desktop\project\Sarcasm_Headlines_Dataset.json'
data = parse_data(file_path)
data_df = pd.DataFrame(data)

train_data, test_data = train_test_split(data_df, test_size=0.3, random_state=42)

from transformers import BertModel

class SimpleFineTunedBERT(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleFineTunedBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
    
    
from transformers import BertTokenizer

class BertTokenDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        bert_model_name: str = 'bert-base-uncased'
    ):
        super().__init__()
        self.data = data

        self.bert_model_name = bert_model_name
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model_name,
            do_lower_case=False
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data.iloc[idx] # -> Series

        is_sarctastic = int(item['is_sarcastic'])
        sentence = item.headline

        tokenized_sentence = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens = True,
            max_length=64,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized_sentence['input_ids'],
            'attention_mask': tokenized_sentence['attention_mask'],
            'class': is_sarctastic
        }
    
    
def train(train_data, test_data, n_epochs=1, max_length=7, batch_size=16):
    # create Dataset and Dataloader
    
    device = torch.device("cuda")
    
    train_dataset = BertTokenDataset(
        data=train_data,
    )
    test_dataset = BertTokenDataset(
        data=test_data,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size
    )
    
    # init model, loss, optimizer
    model = torch.nn.Sequential(
                            SimpleFineTunedBERT(num_classes=2),
                            torch.nn.ReLU(),
                            torch.nn.Linear(2, 1),
                            torch.nn.Sigmoid()
    )
    
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3) # see what's the difference between Adam & AdamW

    model.cuda(device)
    
    # tqdm -- DYOR
    for i in range(n_epochs):
        print(f"Epoch {i+1}")
        # Train loop
        train_tqdm = tqdm.tqdm(train_dataloader)
        model.train()
        for batch in train_tqdm:
            input_ids, attention_mask, cls = batch['input_ids'], batch['attention_mask'], batch['class']

            optimizer.zero_grad()
            model.zero_grad()
            
            output = model((input_ids.squeeze(1).to(device), attention_mask.squeeze(1).to(device))).to(device)                
          
            L = loss_function(output.view(-1), cls.float().to(device))
            
            L.backward()
            
            optimizer.step()
            train_tqdm.set_description(f"Train loss: {L.item()}")
            train_tqdm.refresh()
            
        # Validation loop
        model.eval()
        val_loss = []
        test_tqdm = tqdm.tqdm(test_dataloader)
        with torch.no_grad():
            for batch in test_tqdm:
                input_ids, attention_mask, cls = batch['input_ids'], batch['attention_mask'], batch['class']
                    
                output = model((input_ids.squeeze(1).to(device), attention_mask.squeeze(1).to(device))).to(device)
               
                L = loss_function(output.view(-1), cls.float().to(device))
                
                val_loss.append(L.item())
                test_tqdm.set_description(f"Val loss: {L.item()}")
                test_tqdm.refresh()

        print(f"Epoch {i}, Val loss: {np.mean(val_loss)},val accurency:{}")
        
        train(train_data, test_data, n_epochs=13)

        import json
import numpy as np
import pandas as pd
import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import tqdm

# ÑHÑpÑsÑÇÑÖÑxÑ{Ñp Ñy ÑÅÑÄÑtÑsÑÄÑÑÑÄÑrÑ{Ñp ÑtÑpÑÑÑpÑÉÑuÑÑÑp
def parse_data(file_path):
    for line in open(file_path, 'r'):
        yield json.loads(line)

data = pd.DataFrame(parse_data('dataset/Sarcasm_Headlines_Dataset.json'))

train_df, test_df = train_test_split(data, test_size=0.3, random_state=42)

# ÑOÑÅÑÇÑuÑtÑuÑ|ÑuÑ~ÑyÑu Ñ}ÑÄÑtÑuÑ|Ñy
class FineTunedBERT(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ÑOÑÅÑÇÑuÑtÑuÑ|ÑuÑ~ÑyÑu Ñ{Ñ|ÑpÑÉÑÉÑp ÑtÑpÑÑÑpÑÉÑuÑÑÑp
class SarcasmDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        headline = item['headline']
        label = item['is_sarcastic']

        encoding = self.tokenizer.encode_plus(
            headline,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ÑUÑÖÑ~Ñ{ÑàÑyÑë ÑÄÑqÑÖÑâÑuÑ~ÑyÑë
def train_model(train_df, test_df, num_epochs=1, batch_size=16, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = SarcasmDataset(data=train_df, tokenizer=tokenizer)
    test_dataset = SarcasmDataset(data=test_df, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = FineTunedBERT(num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_predictions_train = 0
        total_predictions_train = 0
        
        train_tqdm = tqdm.tqdm(train_loader)
        for batch in train_tqdm:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions_train += torch.sum(preds == labels)
            total_predictions_train += labels.size(0)
            
            train_tqdm.set_description(f"Epoch {epoch+1}, Train Loss: {loss.item()}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_predictions_train.double() / total_predictions_train
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Train Accuracy: {train_accuracy}")
        
        model.eval()
        val_loss = 0
        correct_predictions_val = 0
        total_predictions_val = 0
        
        test_tqdm = tqdm.tqdm(test_loader)
        with torch.no_grad():
            for batch in test_tqdm:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = loss_function(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct_predictions_val += torch.sum(preds == labels)
                total_predictions_val += labels.size(0)
                
                test_tqdm.set_description(f"Val Loss: {loss.item()}")
        
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = correct_predictions_val.double() / total_predictions_val
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss}, Val Accuracy: {val_accuracy}")

# ÑHÑpÑÅÑÖÑÉÑ{ ÑÄÑqÑÖÑâÑuÑ~ÑyÑë
train_model(train_df, test_df, num_epochs=5)
