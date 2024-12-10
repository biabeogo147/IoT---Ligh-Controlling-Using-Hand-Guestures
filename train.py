from utils import *
from model import NeuralNetwork

class CustomImageDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.labels = torch.from_numpy(self.data.iloc[:,0].to_numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_hot_label = self.labels[idx]
        torch_data = torch.from_numpy(self.data.iloc[idx,1:].to_numpy(dtype=np.float32))
        return torch_data, one_hot_label


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.watched_metrics = np.inf

    def early_stop(self, current_value):
        if current_value < self.watched_metrics:
            self.watched_metrics = current_value
            self.counter = 0
        elif current_value > (self.watched_metrics + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train():
    trainset = CustomImageDataset(train_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True)

    valset = CustomImageDataset(val_path)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=50, shuffle=True)

    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=30, min_delta=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_val_loss = np.inf
    timestamp = datetime.now().strftime('%d-%m %H:%M')

    for epoch in range(300):
        model.train(True)
        running_loss = 0.0
        acc_train = Accuracy(num_classes=len(list_label), task="multiclass")
        for batch_number, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            acc_train.update(outputs, labels)
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)

        # Validation
        model.train(False)
        running_val_loss = 0.0
        acc_val = Accuracy(num_classes=len(list_label), task="multiclass")
        for batch_number, data in enumerate(val_loader):
            inputs, labels = data
            outputs = model(inputs)
            val_loss = loss_fn(outputs, labels)
            acc_val.update(outputs, labels)
            running_val_loss += val_loss.item()

        # Log the running loss averaged per batch for both training and validation
        print(f"Epoch {epoch}: ")
        print(f"Accuracy train:{acc_train.compute().item()}, val:{acc_val.compute().item()}")
        avg_val_loss = running_val_loss / len(val_loader)
        print('LOSS train {} valid {}'.format(avg_loss, avg_val_loss))
        print('Training vs. Validation Loss',
              {'Training': avg_loss, 'Validation': avg_val_loss},
              epoch + 1)
        print('Training vs. Validation accuracy',
              {'Training': acc_train.compute().item()
                  , 'Validation': acc_val.compute().item()},
              epoch + 1)

        # Track the best performance, and save the model's state
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = f'{save_model_path}/model_{timestamp}_{model.__class__.__name__}_best'
            torch.save(model.state_dict(), best_model_path)

        # Early stopping
        if early_stopper.early_stop(avg_val_loss):
            print(f"stopping at epoch {epoch}, minimum: {early_stopper.watched_metrics}")
            break

    # Save the last model's state
    model_path = f'{save_model_path}/model_{timestamp}_{model.__class__.__name__}_last'
    torch.save(model.state_dict(), model_path)

    print(acc_val.compute())

    return model, best_model_path

if __name__ == "__main__":
    model, best_model_path = train()

    testset = CustomImageDataset(test_path)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False)

    network = NeuralNetwork()
    network.load_state_dict(torch.load(best_model_path, weights_only=False))

    network.eval()
    acc_test = Accuracy(num_classes=len(list_label), task="multiclass")
    for batch_number, data in enumerate(test_loader):
        inputs, labels = data
        outputs = network(inputs)
        acc_test.update(outputs, labels)

    print(network.__class__.__name__)
    print(f"Accuracy of model on test set: {acc_test.compute().item()}")
    print("========================================================================")
