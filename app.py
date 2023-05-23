from flask import Flask, render_template, request



import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora

from gensim.models import LdaModel


# Initialize the Flask application
app = Flask(__name__)





df = pd.read_json('News_Category_Dataset_v3.json', lines=True)




# Define the topics to exclude
excluded_topics = ['POLITICS', 'WELLNESS','ENTERTAINMENT']

# Drop entries based on the condition excluding the specified topics
filtered_df = df




def preprocess_documents(documents):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    cleaned_docs = [[lemmatizer.lemmatize(token) for token in doc if token.isalpha() and token not in stop_words] for doc in tokenized_docs]
    dictionary = corpora.Dictionary(cleaned_docs)
    document_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_docs]

    return dictionary, document_term_matrix


def train_lda_model(document_term_matrix, num_topics):
    lda_model = LdaModel(document_term_matrix, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model

def extract_topics(lda_model):
    topics = lda_model.print_topics()
    return topics


# Step 2: Preprocess the data
documents = filtered_df['short_description']
dictionary, document_term_matrix = preprocess_documents(documents)


# Step 3: Train the LDA model
num_topics = 3

lda_model = LdaModel.load('lda_model.model')






app.debug = True
# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    description = request.form.get('input_text')
    print(description)


    # Example news description
    new_description = [ ''+description ]

    # Preprocess the new news description
    dictionary, document_term_matrix = preprocess_documents(new_description)

    # Convert the preprocessed description to a bag of words representation
    # new_article_bow = dictionary.doc2bow(preprocessed_description)

    # Infer the topic distribution for the new news description
    topic_distribution = lda_model.get_document_topics(document_term_matrix)

    # Sort the topics by their probability scores
    sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)

    # Get the most probable topic
    
    most_probable_topic = sorted_topics[0]
    print(most_probable_topic)
    topic_id = most_probable_topic[0][0]

    # Get the topic name based on the topic ID
    print(topic_id)
    topics = extract_topics(lda_model)

    topic_name = topics[topic_id]

    # Print the most probable topic
    print("Most Probable Topic:", topic_name)


    # Prepare the response
    response = {
        'input_text': description,
        'topic_name': topic_name
    }

    return render_template('result.html', response=response)


# Run the Flask application
if __name__ == '__main__':
    app.run()
