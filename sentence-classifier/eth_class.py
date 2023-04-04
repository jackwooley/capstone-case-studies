import numpy as np
import flair
import flair.data
import pandas as pd
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from sklearn.model_selection import train_test_split

def append_to_df(df, section, val):
    df = df[['text', section+'_bucket']]

    df = df.append(pd.read_csv('upsample_txt/'+section+val+'.csv'))
    df =df.append(pd.read_csv('up_copy_txt/'+section+val+'.csv'))
    return df

df = pd.read_csv("fin_data.csv")  # i limited it to 500 for speed of trainig on a cpu, read all when fully training

print(len(df))
df = append_to_df(df, 'satisfaction', '5')
df = append_to_df(df, 'satisfaction', '1')

print(len(df))

label_type = 'satisfaction'

train_dev, test = train_test_split(df, test_size=0.2)
train, dev = train_test_split(train_dev, test_size=0.2)

def load_df_to_sentences(df: pd.DataFrame):
    sentences = []
    for index, row in df.iterrows():
        sentence = flair.data.Sentence(row['text'])
        label = str(row['satisfaction_bucket'])  # must be a string, it's a classification
        sentence.add_label(label_type, label, 1.0)
        sentences.append(sentence)
    return sentences

train_sentences = load_df_to_sentences(train)
test_sentences = load_df_to_sentences(test)
dev_sentences = load_df_to_sentences(dev)



# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = Corpus(train_sentences, dev_sentences, test_sentences)



# 3. create the label dictionary
label_dict = corpus.make_label_dictionary(label_type=label_type)

# 4. initialize transformer document embeddings (many models are available)
document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)  # transformer embeddings are hard core, awesome, you can experiment

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(classifier, corpus)

# 7. run training with fine-tuning
trainer.fine_tune('./test_distilbert',
                  learning_rate=5.0e-5,  # another good one to mess with
                  mini_batch_size=4,  # increase this, higher for a cpu, don't go above 8 on a gpu, sometimes get problem
                  max_epochs=10,  # mess with this
                  )

