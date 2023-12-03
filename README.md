# Digit classifier
----------------------

## Overview

The primary goal of this project is to build a versatile prototype Digit Classifier that can use any of the three models (CNN, RandomForest, any random model).

*Note*: All presented code was performed with Python 3.8.10.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/RostyslavBryiovskyi/MNIST-classifier.git
cd MNIST-classifier
```

2. **Install the requirements:**
```bash
pip install -r requirements.txt
```

## Usage
In order to run the code, specify algo to use [cnn, rf, rand]:
```bash
python mnist.py --classifier cnn
```