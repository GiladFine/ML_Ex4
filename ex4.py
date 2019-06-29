import os
import torch
from gcommand_loader import GCommandLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


# Activate cuda for GPU device
torch.cuda.init()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Hyper-parameters
LEARNING_RATE = 0.00015
NUMBER_OF_ITERATIONS = 100
BATCH_SIZE = 80
NUMBER_OF_WORKERS = 20

class ML_Model(nn.Module):

    def __init__(self):

        super(ML_Model, self).__init__()

        self.conv_layer_1 = nn.Sequential(nn.Conv2d(1, 8, 6),
                                          nn.ReLU(),
                                          nn.MaxPool2d(2, 2))

        self.conv_layer_2 = nn.Sequential(nn.Conv2d(8, 16, (5, 5)),
                                          nn.ReLU(),
                                          nn.MaxPool2d(2, 2))

        self.fc_layer_1 = nn.Sequential(nn.Linear(16 * 37 * 22, 800),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5))

        self.fc_layer_2 = nn.Sequential(nn.Linear(800, 30))

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)

        x = x.view(-1, 16 * 37 * 22)

        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)

        return F.softmax(x, dim=1)


class NN(object):

    def __init__(self, train_loader, validation_loader):
        self.train_loader = train_loader
        self.validation_loader = validation_loader

    def __evaluate(self, model, loader):
        model.eval()
        number_of_hits = 0
        number_of_records = 0

        for records, labels in loader:
            # input & labels vector sized by batch size
            records = records.to(DEVICE)
            labels = torch.LongTensor(labels).to(DEVICE)

            # Get prediction vector for batch
            probabilities = model(records)
            predictions = torch.argmax(probabilities, dim=1)

            number_of_records += len(predictions)
            for i in range(len(predictions)):
                if predictions[i] == labels[i]:
                    number_of_hits += 1

        return number_of_hits / number_of_records


    def __train_iteration(self, model, loss_function, optimizer):
        model.train()
        loss_sum = 0

        for records, labels in self.train_loader:
            records = records.to(DEVICE)
            labels = torch.LongTensor(labels).to(DEVICE)

            model.zero_grad()
            probabilities = model(records)
            loss = loss_function(probabilities, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        # Return average loss
        return loss_sum / len(self.train_loader)


    def train(self):

        loss_function = nn.CrossEntropyLoss()
        model = ML_Model()
        model.to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        for iter in range(NUMBER_OF_ITERATIONS):
            print('iteration number '+ str(iter + 1) + ':')
            loss = self.__train_iteration(model, loss_function, optimizer)

            train_accuracy = self.__evaluate(model, self.train_loader)
            print('train accuracy - ', train_accuracy)

            validation_accuracy = self.__evaluate(model, self.validation_loader)
            print('validation accuracy - ', validation_accuracy)

            print('loss - ', loss)

        return model

    def predict(self):
        test_set = GCommandLoader('data/test')
        test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUMBER_OF_WORKERS, pin_memory=True)

        files = [os.path.basename(f[0]) for f in test_set.spects]
        model = ML_Model()
        model.load_state_dict(torch.load('ML_Ex4.model'))

        model.to(DEVICE)
        model.eval()
        sound_files_index = 0

        for records, labels in test_loader:
            records = records.to(DEVICE)
            probabilities = model(records)
            predictions = torch.argmax(probabilities, dim=1)

            # Write to test_y file
            for pred in predictions:
                with open('test_y', 'a+') as file:
                    file.write(files[sound_files_index] + ', ' + str(pred.item()) + '\n')
                sound_files_index += 1


def main():

    train_set = GCommandLoader('data/train')
    validation_set = GCommandLoader('data/valid')

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUMBER_OF_WORKERS, pin_memory=True)

    validation_loader = DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUMBER_OF_WORKERS, pin_memory=True)

    nn = NN(train_loader, validation_loader)

    # Train and evaluate model
    model = nn.train()

    # Saved trained model to file
    torch.save(model.state_dict(), 'ML_Ex4.model')

    nn.predict()

if __name__ == "__main__":
    main()
