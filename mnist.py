import argparse
from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image):
        pass
    
    def train(self):
        raise NotImplementedError("Training function isn`t implemented.")

class CNNModel(DigitClassificationInterface, nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def predict(self, image):
        image = torch.from_numpy(image).permute(2,0,1)
        with torch.no_grad():
            output = self.forward(image)
            return int(output.argmax(dim=1, keepdim=True).detach().cpu().numpy())

class RandomForestModel(DigitClassificationInterface):
    def __init__(self):
        self.model = RandomForestClassifier()
        self._fit()
    
    def _fit(self):
        # just use fit method on random data because we aren`t able to use predict before fit 
        random_data_x = np.random.rand(2,784)
        random_data_y = np.random.randint(0,10, size=2)
        self.model.fit(random_data_x, random_data_y)

    def predict(self, image):
        flattened_image = image.flatten()
        return self.model.predict([flattened_image])[0]

class RandomModel(DigitClassificationInterface):
    def __init__(self):
        pass

    def crop_center(self, img, cropx=10, cropy=10):
        y,x,_ = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]

    def predict(self, image):
        image = self.crop_center(image)
        # Return a random value as a result of classification
        return np.random.randint(0, 10)

class DigitClassifier:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.model = self._get_model()

    def _get_model(self):
        if self.algorithm == "cnn":
            return CNNModel()
        elif self.algorithm == "rf":
            return RandomForestModel()
        elif self.algorithm == "rand":
            return RandomModel()
        else:
            raise ValueError("Invalid algorithm. Supported algorithms: cnn, rf, rand")

    def predict(self, image):
        return self.model.predict(image)
    
    def train(self):
        return self.model.train()

if __name__ == "__main__":
    # init argparser
    parser = argparse.ArgumentParser(description="Mnist classifier.")
    parser.add_argument("--classifier", default='cnn', choices=['cnn', 'rf', 'rand'], type=str, help="Choose the type of classifier")

    # parse args
    args = parser.parse_args()

    # Create a DigitClassifier with a specific algorithm
    classifier = DigitClassifier(algorithm=args.classifier)

    # Generate a random sample image
    sample_image = np.random.rand(28, 28, 1).astype('f')

    # Get predictions from the classifier
    prediction = classifier.predict(sample_image)
    
    # Display the result
    print(f"Predicted class is: {prediction}")
