import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch import nn
from transformers import BertModel
from tqdm import tqdm


class AnyData(Dataset):
    
    def __init__(self, df):
        self.labels = [label for label in df['toxicity']]
        self.texts = [text for text in df['comment_text']]
    
    def __getitem__(self, index):
        x = self.texts[index]
        y_true = np.zeros([1, 6])
        y_true[0, self.labels[index]] = 1
        y_true = torch.tensor(y_true).squeeze()
        
        return x, y_true
        
    def __len__(self):
        return len(self.labels)
    
    def classes(self):
        return self.labels

    
class LaBSE_clf(nn.Module):

    def __init__(self, external_preprocessor, external_encoder, dropout=0.15):
        super(LaBSE_clf, self).__init__()

        self.encoder = external_encoder
        self.preprocessor = external_preprocessor
        # self.dropout = nn.Dropout(torch.tensor(dropout))
        self.linear = nn.Linear(768, 6)
        self.relu = nn.ReLU()

    def forward(self, sents):
        preprocessed = self.preprocessor(sents)
        pooled_output = torch.tensor(self.encoder(preprocessed)["default"].numpy())
        # dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)

        return final_layer

    
    from torch.optim import Adam


def train(model, train_data, val_data, learning_rate, epochs, batch, w):

    train, val = AnyData(train_data), AnyData(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch)

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
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                output = model(train_input)
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                output_ = torch.nn.functional.one_hot(output.argmax(dim=1), num_classes=6)
                acc = (output_ == train_label).sum().cpu().item()
                total_acc_train += acc
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            total_acc_val = 0
            total_loss_val = 0
            model.eval()
            with torch.no_grad():
                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)
                    output = model(val_input)
                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.cpu().item()
                    output_ = torch.nn.functional.one_hot(output.argmax(dim=1), num_classes=6)
                    acc = (output_ == val_label).sum().cpu().item()
                    total_acc_val += acc
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


np.random.seed(1)
df_train, df_val = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df))])

EPOCHS = 5
model = LaBSE_clf(preprocessor, encoder)
LR = 1e-6
bsize = 100
              
train(model, df_train, df_val, LR, EPOCHS, bsize, 0)
