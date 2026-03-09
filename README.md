# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

The objective of this project is to develop a Bidirectional Long Short-Term Memory (BiLSTM) model for Name Recognition in sentences. The model will perform Named Entity Recognition (NER) by identifying and classifying person names within input text sequences. Given a sentence, the model should label each word using sequence tagging (e.g., B-PER, I-PER, O) to determine whether it is part of a person’s name or not. The performance of the model will be evaluated using metrics such as accuracy and validation loss.

## DESIGN STEPS
Step 1:
Import the necessary libraries.

Step 2:
Load the dataset and use DataLoader to batch the dataset

Step 3:
Create a class to define the Long Short Term Memory Neural Network, in the class define the forward function

Step 4:
Initialize the model and get a model summary

STEP 5:
Initialize the loss function MSELoss and Optimizier

STEP 6:
Create a function to train the model and call it to train the model.

STEP 7:
Test the model using the test_loader.

Step 8:
Display the results.

## PROGRAM :

### Name: Subash M
### Register Number: 212224220109

```python

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_dim)
        self.drop = nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2,tagset_size)

    def forward(self, input_ids):
        x = self.embed(x)
        x = self.drop(x)
        x, _ = self.lstm(x)
        x = self.fc(x)

        return x

model = BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
   
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
                val_loss += loss.item()

            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, val loss = {val_loss:.4f}") 


    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="823" height="651" alt="Screenshot 2026-03-07 015746" src="https://github.com/user-attachments/assets/d64f7322-1471-433c-9570-0b5a079a9e6a" />


### Sample Text Prediction

<img width="739" height="674" alt="Screenshot 2026-03-07 015808" src="https://github.com/user-attachments/assets/fe633c48-8881-4638-a80d-75c0fbf7e59e" />


## RESULT

Thus, A Long Short Term Memory Neural Network model is implemented successfully for recognizing the named entities in the text.


