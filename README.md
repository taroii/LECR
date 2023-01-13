# Learning Equality: Curriculum Recommendations Project 
Taro Iyadomi
12/23/22 - Present

Given topics teachers want to teach about spanning 29 different languages, I constructed a Siamese Neural Network to match educational contents based on their natural language inputs in a multi-class, multi-label classification problem. 

[:star: Link to Kaggle Competition Page :star:](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations)

## Abstract  

The goal of this project was to predict content matches for various educational topics from across the globe. Using the natural language titles, descriptions, and other features related to each topic and content, I preprocessed the text to remove stopwords, punctuation, and whitespaces then vectorized the text into integer vectors of fixed length with a vocabulary of 1,000,000 words associated to unique numbers. After embedding and pooling those vectors, I compared each topic-content pair through an additional neural network that determines whether the vectors were associated with each other (Siamese Network). For each topic in the topics dataset, I compared the topic to all 150,000+ contents and selected the best contents based on a threshold found from an ROC curve.

## Introduction  

## Preprocessing Text Inputs  

## Building the Model  

## Limitations  

## Closing Regards  

## References   

This project was only made possible by Murat Karakaya and Greg Hogg and their amazing tutorials on YouTube which come with detailed Jupyter Notebooks. I highly recommend you check them out!  

[:fire: Murat Karakaya's Keras Text Vectorization Tutorial :fire:](https://www.muratkarakaya.net/2022/11/keras-text-vectorization-layer.html)

[:fire: Greg Hogg's Siamese Network Tutorial :fire:](https://www.youtube.com/watch?v=DGJyh5dK4hU&t=932s&ab_channel=GregHogg)
