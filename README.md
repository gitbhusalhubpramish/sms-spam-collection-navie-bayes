# Sms spam collection with Naive-Bayes

In this project I have applied Naive-Bayes form scratch in [Sms spam collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) dataset in kaggle. This markdown is just the overview of project and how I made this. I have given more detail in my [notes](https://github.com/gitbhusalhubpramish/ml-notes/tree/main/0.007%20navie-bayes) about Naive-Bayes.

<img src="visual.jpg"/>

This is the output of the model - There's not a lot to show in this model. It's hard to visulize. Main thing here is the accuracy and prob distribution which is hard to show in a graph.

## Small intro to Naive-Bayes

Naive-Bayes is a Probability bahsed Supervised Meachine Learning algorithm. It uss Probabilities (Bayes theorem) to predict or classify any data. There are 2 types fo Naive bayes - Categorical and Gussian. I have used Categorical here.

## Feature

This project was made by me to understand Naive-bayes form scratch so it has naivebayes applied form scratch. But also:

- Naive-bayes form scratch
- Probabilities
- 96% model accuracy in testing data

## Tech-Stack

This project has python liberies used for data scienct - visulization, calculation and data manipulation. I've used:

- Numpy for calculation
- streamlit for data presentation
- pandas for CSV and data manipulation
- sk-learn for accuracy calculation and data split

## Work-flow

The work flow of the project look like this:

- **Fetch data:** Here it fethes the data from kaggle dataset.
- **Data Shepration:** The algorithm shaprate the data into 2(4) parts for training and testing with corresponding x and y.
- **Calcualte $P(s|x)$:** Here model calculates $P(s|x)$ by calculating the variable required for this using log spacing naive bayes.
- **Calcualte $P(h|x)$ :** Here model calculates $P(h|x)$ by calculating the variable required for this using log spacing naive bayes.
- **Comparism:** Here $P(s|x)$ and $P(h|x)$ are compared and returned which one is greater (*s mean spam h mean ham*).

## What did I learn?

I had made this project to learn Naive bayes form scratch. Of cource, I had learnt that. But other stuff too like:

- Naive-bayes
- Probabilities
- Catorization

## Some extra info...

I have been learning Meaching Learning so I make note and make atleast one project for each. I have writtern a jupiter notebook to plan and make this work then I make a pyton script file with the help of streamlit so that I can host and support Hackclub. Most of the progress of learning the algorithm is in my notebook while I struggle in streamlit too. I made this to learn so it has very less AI implimented.
