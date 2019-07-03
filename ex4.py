import os
import torch
from gcommand_loader import GCommandLoader
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# Activate cuda for GPU device
#torch.cuda.init()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Paths
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
VALIDATION_DIR = 'data/valid'

MODEL_FILE = 'ML_Ex4.model'
OUTPUT_FILE = 'test_y'

# Hyper-parameters
LEARNING_RATE = 0.00015
NUMBER_OF_ITERATIONS = 50
BATCH_SIZE = 80
NUMBER_OF_WORKERS = 20

class ML_Model(nn.Module):
    """
    This Class represents an NN model by extending nn.module
    """

    def __init__(self):
        """
        Initialize the network layers param
        """
        super(ML_Model, self).__init__()

        # Convolution layers
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(1, 8, 6),
                                          nn.LeakyReLU(),
                                          nn.MaxPool2d(2, 2))

        self.conv_layer_2 = nn.Sequential(nn.Conv2d(8, 16, 5),
                                          nn.LeakyReLU(),
                                          nn.MaxPool2d(2, 2))

        # Fully connected layers
        self.fc_layer_1 = nn.Sequential(nn.Linear(16 * 37 * 22, 800),
                                        nn.LeakyReLU(),
                                        nn.Dropout(p=0.5))

        self.fc_layer_2 = nn.Sequential(nn.Linear(800, 30))

    def forward(self, x):
        """
        Feed x forward in the net
        :param x: data record
        :return: Softmax distribution over the output layer classes
        """
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)

        x = x.view(-1, 16 * 37 * 22)

        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)

        return F.softmax(x, dim=1)


class NN(object):
    """
    This class represents a generic NN structure (No parameters)
    """
    def __accuracy(self, model, data_loader):
        """
        Evaluate the model and use the softmax output to calc number of correct records out of the data - accuracy
        :param model: ML_Model class
        :param loader: DataLoader class
        :return: accuracy
        """
        model.eval()
        number_of_hits = 0
        number_of_records = 0

        for records, labels in data_loader:
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


    def __train_one_iteration(self, model, loss_function, optimizer, train_loader):
        """
        Trains one iteration(epoch) of the model
        :param model: ML_Model
        :param loss_function: Pre-determined loss func (NLL, Cross-entropy, Hinge...)
        :param optimizer: Pre-determined optimizer (Adam, AdaGrad, SGD...)
        :return: Average loss for this iteration
        """
        model.train()
        loss_sum = 0

        for records, labels in train_loader:
            records = records.to(DEVICE)
            labels = torch.LongTensor(labels).to(DEVICE)

            model.zero_grad()
            probabilities = model(records)
            loss = loss_function(probabilities, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        # Return average loss
        return loss_sum / len(train_loader)


    def train(self, train_loader, validation_loader):
        """
        Train the network multiple times
        :return: the nn.module being used (ML_Model in our case)
        """

        loss_function = nn.NLLLoss()
        model = ML_Model()
        model.to(DEVICE)

        adam_optim = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        for iter in range(NUMBER_OF_ITERATIONS):
            print('iteration number '+ str(iter + 1) + ':')
            loss = self.__train_one_iteration(model, loss_function, adam_optim, train_loader)

            train_accuracy = self.__accuracy(model, train_loader)
            validation_accuracy = self.__accuracy(model, validation_loader)

            print('train accuracy - ' + str(100 * train_accuracy) + "%")
            print('validation accuracy - ' + str(100 * validation_accuracy) + "%")
            print('loss - ', loss)

        return model

    def predict(self):
        """
        Predict the test set labels and writes them to test_y
        """
        test_set = GCommandLoader(TEST_DIR)
        test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUMBER_OF_WORKERS, pin_memory=True)

        files = [os.path.basename(f[0]) for f in test_set.spects]
        model = ML_Model()

        # Loading the model from file
        model.load_state_dict(torch.load(MODEL_FILE))

        model.to(DEVICE)
        model.eval()
        sound_files_index = 0

        for records, labels in test_loader:
            records = records.to(DEVICE)
            probabilities = model(records)
            predictions = torch.argmax(probabilities, dim=1)

            # Write to test_y file
            for pred in predictions:
                with open(OUTPUT_FILE, 'a+') as file:
                    file.write(files[sound_files_index] + ', ' + str(pred.item()) + '\n')
                sound_files_index += 1


def main():

    train_set = GCommandLoader(TRAIN_DIR)
    validation_set = GCommandLoader(VALIDATION_DIR)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUMBER_OF_WORKERS, pin_memory=True)

    validation_loader = DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUMBER_OF_WORKERS, pin_memory=True)

    nn = NN()
    nn_model = nn.train(train_loader, validation_loader)

    # Writes the model to a file, this is the input for the predict func
    torch.save(nn_model.state_dict(), MODEL_FILE)

    nn.predict()

if __name__ == "__main__":
    main()