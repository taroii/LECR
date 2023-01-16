# Learning Equality: Curriculum Recommendations  
By Taro Iyadomi  
12/23/22 - 1/16/2023  

Given topics teachers want to teach about spanning 29 different languages, I constructed a Siamese Neural Network to match educational contents based on their natural language inputs in a multi-class, multi-label classification problem.  

[:star: Link to Kaggle Competition Page :star:](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations)

## Abstract  

The goal of this project was to predict content matches for various educational topics from across the globe. Using the natural language titles, descriptions, and other features related to each topic and content, I preprocessed the text to remove stopwords, punctuation, and whitespaces then vectorized the text into integer vectors of fixed length with a vocabulary of 1,000,000 words associated to unique numbers. After embedding and pooling those vectors, I compared each topic-content pair through an additional neural network that determines whether the vectors were associated with each other (Siamese Network). For each topic in the topics dataset, I compared the topic to all 150,000+ contents and selected the best contents based on a threshold found from an ROC curve.

## Introduction  

"Every country in the world has its own educational structure and learning objectives. Most materials are categorized against a single national system or are not organized in a way that facilitates discovery. The process of curriculum alignment, the organization of educational resources to fit standards, is challenging as it varies between country contexts.  

Current efforts to align digital materials to national curricula are manual and require time, resources, and curricular expertise, and the process needs to be made more efficient in order to be scalable and sustainable. As new materials become available, they require additional efforts to be realigned, resulting in a never-ending process. There are no current algorithms or other AI interventions that address the resource constraints associated with improving the process of curriculum alignment" (Kaggle).  

In an attempt to mitigate this issue, I created a model that reads all of the information about each topic and content, then predicts the probability that they are a match based on a Siamese Network architecture. The model was trained using a provided correlations dataset, which contained about 60,000 topic/content matches. This was a multi-class, multi-label classification problem, meaning that for each topic I predicted multiple contents that would match. To do this, I used ROC curves to select a threshold that would choose only the best matches for each topic. This allowed me to predict 1-10 matching contents for each topic out of a total 150,000 unique contents!

## Preparing the Data  

The first step of preprocessing the inputs was to manipulate the format of the topics and content to match each other. The topics and contents datasets had some identical features, but they each contained unique columns that the other did not. In order to standardize the inputs, I only selected the features that directly described the content of each text. Then, for training and testing the model, I used the given correlations dataset to inner-join the topics and the content that I knew are related to each other.

```python
#### Create combine function
def combine(correlations, topics, content):
    '''
    - Inputs our three datasets and combines the topic/content information with the topic/content correlations data.
    - All topic/content information is concatenated to one "features" column, which includes the language, title, description, etc.
    - Output includes the correlations topics information, correlations content information, and a dictionary to convert indices to their
      corresponding topic/content id.
    '''
    #Drop/combine columns
    content["text"] = content["text"].fillna('')
    content = content.dropna()
    content_combined = content["language"] + " " + content["title"] + " " + content["description"] + " " + content["text"]
    content_combined = pd.DataFrame({"id":content["id"], "features":content_combined})

    topics["description"] = topics["description"].fillna('')
    topics = topics.dropna()
    topics_combined = topics["language"] + " " + topics["channel"] + ' ' + topics["title"] + " " + topics["description"]
    topics_combined = pd.DataFrame({"id":topics["id"], "features":topics_combined})

    #Explode correlations rows
    correlations["content_ids"] = correlations["content_ids"].str.split()
    correlations = correlations.explode("content_ids")

    #Merge
    merged = correlations.merge(topics_combined, how="inner", left_on="topic_id", right_on="id")
    merged = merged.reset_index().merge(content_combined, how="inner", left_on="content_ids", right_on="id", sort=False, suffixes=("_topics", "_content")).sort_values(axis=0, by="index")
    merged = merged.drop(["content_ids", "topic_id"], axis=1)

    #Split
    corr_topics = merged[['index', 'features_topics']]
    corr_topics.columns = ['id', 'features']
    corr_content = merged[['index', 'features_content']]
    corr_content.columns = ['id', 'features']

    index_to_topic = pd.Series(merged.id_topics.values, index=merged.index).to_dict()
    index_to_content = pd.Series(merged.id_content.values, index=merged.index).to_dict()

    return corr_topics, corr_content, index_to_topic, index_to_content
```  

The second step after standardizing the features was to reduce the complexity of the features by removing unnecessary stopwords. Stopwords are words that tie sentences together grammatically (such as "the", "a", "and", et cetera), but our model doesn't need those words to understand the material, so I created a stopword removal function to remove them using the Natural Language Tool Kit module, which contains several supported languages.  

```Python
#Dictionary of languages found in our data
lang_dict = {
    "en":"english",
    "es":"spanish",
    "it":"italian",
    'pt':"portuguese",
    'mr':'marathi',
    'bg':'bulgarian',
    'gu':'gujarati',
    'sw':'swahili',
    'hi':'hindi',
    'ar':'arabic',
    'bn':'bengali',
    'as':'assamese',
    'zh':'chinese',
    'fr':'french',
    'km':'khmer',
    'pl':'polish',
    'ta':'tamil',
    'or':'oriya',
    'ru':'russian',
    'kn':'kannada',
    'swa':'swahili',
    'my':'burmese',
    'pnb':'punjabi',
    'fil':'filipino',
    'tr':'turkish',
    'te':'telugu',
    'ur':'urdu',
    'fi':'finnish',
    'pn':'unknown',
    'mu':'unknown'}

# List of languages supported by the natural language tool kit (NLTK) module.
supported_languages = stopwords.fileids()

def remove_stopwords(text):
    '''
    Checks language of text then removes stopwords from that language if supported.
    '''
    lang_code = text[0:2]
    if lang_dict[lang_code] in supported_languages:
        for word in stopwords.words(lang_dict[lang_code]):
            text = text.replace(' ' + word + ' ', ' ')
    return text
```  

After that, I split the data into training and testing sets, then shifted half of each of those sets so that half of the data is matching and the other half is not matching. This allowed me to train the model on a variety of matches and non-matches.   

Finally, I transformed the current Pandas and NumPy data into train and test TensorFlow datasets, allowing me to batch, cache, and prefetch the data to improve model efficiency. By doing this, the model stores the data in memory for each training epoch, as well as training on smaller samples (batches) at a time rather than training on the entire dataset at once.  

## Building the Model  

Once the TensorFlow data pipeline had been built, I began working on the TextVectorization layer from Keras. This layer takes in a string of any length and converts it to a vector of integers of a fixed length, with integers corresponding to specific words that have been adapted specifically to the training data. First, I created a custom standardization function that encodes text to UTF-8, removes punctuations, lowercases, and removes whitespaces. I opted to use a custom standardization function rather than the default 'lower_and_strip_punctuation' function because there were many languages so I wanted to remove any special characters. With that, I created the TextVectorization layer with a vocab size of 1,000,000 and a output vector length of 50, then adapted the layer on the entire correlations dataset.   

```Python
#### Create Text Vectorization Layer
# Hyperparameters
VOCAB_SIZE = 1000000
MAX_LEN = 50

def my_standardize(text):
    '''
    A text standardization function that is applied for every element in the vectorize layer.
    '''
    text = tf.strings.lower(text, encoding='utf-8') #lowercase
    text = tf.strings.regex_replace(text, f"([{string.punctuation}])", r" ") #remove punctuation
    text = tf.strings.regex_replace(text, '\n', "") #remove newlines
    text = tf.strings.regex_replace(text, ' +', " ") #remove 2+ whitespaces
    text = tf.strings.strip(text) #remove leading and tailing whitespaces
    return text

vectorize_layer = TextVectorization(
    standardize = my_standardize,
    split = "whitespace",
    max_tokens = VOCAB_SIZE + 2,
    output_mode = 'int',
    output_sequence_length = MAX_LEN
)

#### Adapt text vectorization layer to our data
vectorize_layer.adapt(pd.concat([corr_topics["features"], corr_content["features"]]))
```   

Once the TextVectorization layer was set up, I proceeded to build the Siamese Neural Network, which takes in two inputs and applies a smaller neural network on them separately, then takes those outputs and inputs them into another neural network together to output a probability that the two inputs were a match. Along with the TextVectorization layer, I applied a embedding and global average pooling layer to both inputs. The embedding layer creates an n-dimensional vector for each 1-dimensional vector of integers, which better teaches the computer to learn the "meaning" of words by their location in n-dimensional space. For example, the words "good" and "bad" may be used in the same scenarios quite often, and their corresponding integers may be very similar. However, they mean entirely different things and so the embedding layer would point them in opposite directions. After that, I applied a global average pooling layer, which, put simply, reduces the extra dimensionality introduced in the embedding layer to make the model less complex and run faster. With that, I flattened the output and applied a series of Dense layers.   

```python
inp_topics = Input((1, ), dtype=tf.string)
inp_content = Input((1, ), dtype=tf.string)

vectorized_topics = vectorize_layer(inp_topics)
vectorized_content = vectorize_layer(inp_content)

snn = Sequential([
  Embedding(VOCAB_SIZE, 256),
  GlobalAveragePooling1D(),
  Flatten(),
  Dense(128, activation='relu'),
])

snn_content = snn(vectorized_content)
snn_topics = snn(vectorized_topics)

concat = Concatenate()([snn_topics, snn_content])

dense = Dense(64, activation='relu')(concat)

output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[inp_topics, inp_content], outputs=output)

model.summary()
```   

When compiling, I chose binary crossentropy as the loss function. This is because I was comparing each topic-content pair independently, so the output should be a probability from 0 to 1 (unlike softmax, which gives you a number from 0 to 10 that's dependent on the other elements). For the optimizer, I chose the Adam optimizer algorithm (adaptive moment estimation). The "adaptive" part of the name is the key point here, since it adapts the model's learning rate to optimally obtain minimas when searching for the correct answer. On top of that, this optimizer has many benefits, such as its good performance with large, noisy datasets and its memory efficiency. For the metric, I chose AUC (area under curve), which represents the area under the ROC curve of this model. I chose this instead of accuracy because it works better with unbalanced datasets (since each topic should only have 1-10 content matches out of 150,000). This is because AUC focuses on minimizing false positive and false negative rates, which might perform worse than accuracy on training data, but will usually perform better on testing data. Lastly, I trained this model on the training dataset from earlier for 5 epochs. The beauty of Siamese Networks is that they are very easy to train, requiring very few epochs to obtain good results!  

```Python
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=tf.keras.metrics.AUC())

model.fit(train_ds, epochs=5, verbose=1) # Final loss: 0.1229, AUC: 0.9898

model.evaluate(test_ds, verbose=1) # loss: 0.7072, AUC: 0.8648
```  

## Applying the Model  

The last step for this project was to implement the model and predict content matches for the entire topics dataset. Since the given correlations dataset contained about 60,000 topic-content matches already, I only had to predict the matches of the remaining ~15,000 topics. So, after anti-joining the topics and correlations data, I processed the remaining data the same way I processed the training and testing data.  

```Python
#### Antijoin topics with correlations data, since we don't have to predict those topics
outer_joined = topics.merge(correlations, how='outer', left_on='id', right_on='topic_id', indicator=True)
topics = outer_joined[(outer_joined._merge == 'left_only')].drop('_merge', axis=1)

#### Fill missing values and concatenate text to features column
topics = topics.fillna("")
topics_ids = topics.id.values
topics_lang = topics.language
topics_index = topics.index
topics_features = topics["language"] + ' ' + topics["channel"] + ' ' + topics["title"] + ' ' + topics["description"]
del topics

#### Repeat for content, except we keep all content data
content = content.fillna("")
content_ids = content.id.values
content_index = content.index
content_lang = content.language
content_features = content["language"] + ' ' + content["title"] + ' ' + content["description"] + ' ' + content["text"]
del content

index_to_content = pd.Series(content_ids, index=content_index).to_dict()
index_to_topic = pd.Series(topics_ids, index=topics_index).to_dict()

#### Remove stopwords
topics_features = topics_features.apply(remove_stopwords)
content_features = content_features.apply(remove_stopwords)
```  

After that, I created a loop to compare each topic with all 150,000 contents. Using a threshold I obtained from the testing data, I selected the contents that matched the topics the most and wrote them to a submission file. This process was the most time-consuming part of this project, and it was definitely the largest limitation of this model implementation. I will discuss the limitations further and possible solutions in the next section.  

```Python
#### Write predictions to output_file
THRESHOLD = 0.99994

output_file = "submission.csv"
f = open(output_file, 'w')

writer = csv.writer(f)
writer.writerow(["topic_id", "content_ids"])

for i in topics_features.index:
    temp_content = tf.data.Dataset.from_tensor_slices(
        tf.cast(content_features[content_lang == topics_lang[i]], tf.string)
    )
    temp_topic = tf.data.Dataset.from_tensor_slices(
        tf.cast(np.repeat(topics_features[i], len(temp_content)), tf.string)
    )
    temp_ds = tf.data.Dataset.zip(((temp_topic, temp_content), ))\
        .batch(batch_size=64)\
            .cache()\
                .prefetch(tf.data.experimental.AUTOTUNE)
    matches = model.predict(temp_ds, verbose=0)
    matches = [i for i in range(len(matches)) if matches[i] > THRESHOLD]
    matches = " ".join([index_to_content[x] for x in matches])
    writer.writerow([index_to_topic[i], matches])

#### Add given correlations data
writer.writerows([correlations.topic_id, correlations.content_ids])    

f.close()
```  

## Limitations  

This model was not perfect. In fact, there's a reason why Siamese Networks are predominantly used for image recognition and not natural language processing. The first limitation was how the computer didn't learn the true meanings of the words or sentences, rather, it tried to learn specific orderings of specific words, which it would fail to understand if the sentence was reworded differently. This limitation is the main reason why transformers and other natural language processing techniques are so popular today, as they allow the model to better understand the overlying meaning of each sentence, rather than look at the sentence word by word.  

Another limitation was that this model was terribly inefficient. As I stated in the previous part, since I have to compare each topic to every single content to know which contents are a match, the model predicts approximately 15,000 x 150,000 times. To remedy this, I made it so that the model compares contents of the same language, but even then the number of predictions made was astronomically high. A possible, improved implementation would have been to use a natural language generation model to predict a few key words, then search for those words in the contents dataset.  

Both of these limitations would possibly be removed if I had used pre-trained, transformer-based models, especially one from [Google's BERT catalogue](https://github.com/google-research/bert). In the (near) future, I will try to implement one of those models to improve all aspects of this model.     

## Conclusion  

The use of a Siamese Neural Network has its pros and cons for this context, and ultimately I beleive the cons outweigh the pros. The model itself performs very well for smaller datasets, as it predicted a testing AUC of 0.86 after only five epochs of training. However, when it comes to implementing the model on moderately large datasets and above, the model performs poorly due to the competition's time constrains. Future considerations for using pre-trained, readily available, transformer-based models have been made, yet as this was my introductory project into deep learning, I am very proud of what I have accomplished!  

## Acknowledgements   

This project was only made possible by Murat Karakaya and Greg Hogg and their amazing tutorials on YouTube which come with detailed Jupyter Notebooks. I highly recommend you check them out!  

[:fire: Murat Karakaya's Keras Text Vectorization Tutorial :fire:](https://www.muratkarakaya.net/2022/11/keras-text-vectorization-layer.html)

[:fire: Greg Hogg's Siamese Network Tutorial :fire:](https://www.youtube.com/watch?v=DGJyh5dK4hU&t=932s&ab_channel=GregHogg)
