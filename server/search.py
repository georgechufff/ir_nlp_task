import pickle
import os
from datasets import load_dataset
import numpy as np
import nltk


tokenizer = nltk.WordPunctTokenizer()
lemmatizer = nltk.WordNetLemmatizer()


def tokenize_pipeline(sentence):
    tokens = tokenizer.tokenize(sentence)
    return [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]


try:
    f1, f2 = open('documents.pickle', 'rb'), open('adj_documents.pickle', 'rb')
    documents, adj_documents = pickle.load(f1), pickle.load(f2)
    nltk.data.find('wordnet')
    nltk.data.find('stopwords')
except IndexError:
    nltk.download('wordnet')
    nltk.download('stopwords')
except FileNotFoundError:
    dataset = 'lifestyle'
    collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
    documents = collection_dataset['dev_collection']['text']
    adj_documents = [' '.join(tokenize_pipeline(doc)) for doc in documents]
    if 'documents.pickle' not in os.listdir('.') or 'adj_documents.pickle' not in os.listdir('.'):
        f1, f2 = open('documents.pickle', 'x'), open('adj_documents.pickle', 'x')
    f1, f2 = open('documents.pickle', 'wb'), open('adj_documents.pickle', 'wb')
    pickle.dump(documents, f1)
    pickle.dump(adj_documents, f2)

from nltk.corpus import stopwords

class Document:
    def __init__(self, title, text, embeddings):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
        self.embeddings = embeddings
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + '...']


index = []


def build_index():
    # считывает сырые данные и строит индекс
    # index.append(Document('The Beatles — Come Together', 'Here come old flat top\nHe come groovin\' up slowly'))
    # index.append(Document('The Rolling Stones — Brown Sugar', 'Gold Coast slave ship bound for cotton fields\nSold in the market down in New Orleans'))
    # index.append(Document('МС Хованский — Батя в здании', 'Вхожу в игру аккуратно,\nОна еще не готова.'))
    # index.append(Document('Физтех — Я променял девичий смех', 'Я променял девичий смех\nНа голос лектора занудный,'))
    for doc, adj_doc in zip(documents, adj_documents):
        index.append(Document('sample_text', doc[:150], adj_doc.lower()))


k1 = 10
b = 0.75
avgdl = np.sum([len(doc) for doc in documents]) / len(documents)
print(avgdl)


def idf(query_elem):
    doc_frequency = len([doc for doc in documents if query_elem in doc])
    return np.log2((len(documents) - doc_frequency + 0.5) / (doc_frequency + 0.5))


def score(document, query):
    query = ' '.join(set(query.split()) - set(stopwords.words('english')))
    sc = np.sum([idf(q) * document.embeddings.count(q) * (k1 + 1)
                   / (document.embeddings.count(q) + k1 * (1 - b + b * len(documents) / avgdl))
                   for q in query.split()])
    return sc


def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)

    processed_query = ' '.join(set(query.split()) - set(stopwords.words('english')))
    candidates = []
    for doc in index:
        if processed_query.lower() in doc.embeddings.lower():
            candidates.append(doc)
    return candidates[:50]

