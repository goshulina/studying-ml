# !pip install transformers
# !pip install datasets==1.15.1

from torch import nn
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from torch.optim import Adam

class BertClassifier(nn.Module):

    def __init__(self, bert, dropout=0.2):

        super(BertClassifier, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(312, 2)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid() # for roc auc

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask,return_dict=False)
        linear_output = self.linear1(pooled_output)
        final_layer = self.sigmoid(linear_output)
        return final_layer


def train(model, train_ds, val_ds, learning_rate, epochs, batch):

    train_dataloader = torch.utils.data.DataLoader(train_ds, 
                                                   batch_size=batch, 
                                                   shuffle=True, 
                                                   num_workers=2, 
                                                   pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, 
                                                 batch_size=batch, 
                                                 num_workers=2,
                                                 pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            model.train()
            for train in tqdm(train_dataloader):
                
                train_label = train['is_bad'].to(device)
                mask = train['attention_mask'].to(device)
                input_id = train['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                
                train_label_ = torch.nn.functional.one_hot(train_label, num_classes=2)
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.detach().cpu().item()
                
                acc = (output == train_label_).sum().detach().cpu().item()
                total_acc_train += acc

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0
            model.eval()
            with torch.no_grad():
# 
                for val in val_dataloader:
# 
                    val_label = val['is_bad'].to(device)
                    mask = val['attention_mask'].to(device)
                    input_id = val['input_ids'].squeeze(1).to(device)
                    output = model(input_id, mask)

                    val_label_ = torch.nn.functional.one_hot(val_label, num_classes=2)
                    batch_loss = criterion(output, val_label)
                    
                    total_loss_val += batch_loss.detach().cpu().item()
                    
                    acc = (output == val_label_).sum().detach().cpu().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / train_ds.num_rows: .5f} \
                | Train Accuracy: {total_acc_train / train_ds.num_rows: .5f} \
                | Val Loss: {total_loss_val / val_ds.num_rows: .5f} \
                | Val Accuracy: {total_acc_val / val_ds.num_rows: .5f}')


bert = AutoModel.from_pretrained('cointegrated/rubert-tiny')
model = BertClassifier(bert)
train_dataset = load_dataset('csv', data_files='drive/MyDrive/temp/train.csv', split='train')
val_dataset = load_dataset('csv', data_files='drive/MyDrive/temp/val.csv', split='train')
tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny')
train_dataset = train_dataset.map(lambda e: tokenizer(e['description'], padding='max_length', truncation=True), batched=True)
val_dataset = val_dataset.map(lambda e: tokenizer(e['description'], padding='max_length', truncation=True), batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'is_bad'])
val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'is_bad'])


EPOCHS = 1
LR = 1e-5
bsize = 70

train(model, train_dataset, val_dataset, LR, EPOCHS, bsize)

# EVAL
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
model.eval()
total_acc_val = 0
total_loss_val = 0
result = []
with torch.no_grad():

    for val in tqdm(val_dataloader):
        val_label = val['is_bad'].to(device)
        mask = val['attention_mask'].to(device)
        input_id = val['input_ids'].squeeze(1).to(device)
        output = model(input_id, mask)
        result.extend(output.cpu())

y_true = val_dataset['is_bad'].numpy().tolist()
pred = [i.argmax(dim=0).item() for i in result]

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
m = confusion_matrix(y_true, pred)
new_m = []
l = []
print(m)
for i in m:
    s = sum(i)
    for ii in i:
        l.append(round(ii / s * 100))
    new_m.append(l)
    l = []
print(np.array(new_m))

print(f1_score(y_true, pred, average='macro'))
print(accuracy_score(y_true, pred))

