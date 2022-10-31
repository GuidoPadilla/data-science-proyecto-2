import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
abstracts_train = pd.read_csv('./abstracts_train.csv', sep='\t')  
entities_train = pd.read_csv('./entities_train.csv', sep='\t')  

import matplotlib.pyplot as plt
import seaborn as sn
corrMatrix = entities_train.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sn
corrMatrix = abstracts_train.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

# new column that join column title and abstract
temp_t_a = abstracts_train
temp_t_a['title_abstract'] = temp_t_a['title'] + ' ' + temp_t_a['abstract']
# remove title and abstract columns
temp_t_a = temp_t_a.drop(['title', 'abstract'], axis=1)

# show streamlit table with title_anstract
temp_t_a

import scispacy
import spacy

# to display the results
from spacy import displacy

# models
import en_ner_craft_md
import en_ner_jnlpba_md
import en_ner_bc5cdr_md
import en_ner_bionlp13cg_md

# utility
from collections import OrderedDict
from pprint import pprint

# surpress warnings
import warnings
warnings.filterwarnings('ignore')

# Data Visualization
import matplotlib.pyplot as plt

import streamlit as st

def missing_cols(df):
    '''prints out columns with its amount of missing values with its %'''
    total = 0
    for col in df.columns:
        missing_vals = df[col].isnull().sum()
        pct = df[col].isna().mean() * 100
        total += missing_vals
        if missing_vals != 0:
          print('{} => {} [{}%]'.format(col, df[col].isnull().sum(), round(pct, 2)))
    
    if total == 0:
        print("no missing values")

abstracts_test = pd.read_csv('./abstracts_test.csv', sep='\t')  

entities_train['type'].value_counts().plot(kind="barh").invert_yaxis();

# receive a text from the user
text = st.text_input("Enter a text")

nlp = en_ner_bionlp13cg_md.load()
doc_bionlp13cg = nlp(text)
doc_bionlp13cg.ents

nlp_craft = en_ner_craft_md.load()
nlp_jnlpba = en_ner_jnlpba_md.load()
nlp_bc5cdr = en_ner_bc5cdr_md.load()

doc_craft = nlp_craft(text)
doc_jnlpba = nlp_jnlpba(text)
doc_bc5cdr= nlp_bc5cdr(text)

displacy.render(doc_craft, jupyter=True, style='ent')
displacy.render(doc_jnlpba, jupyter=True, style='ent')
displacy.render(doc_bc5cdr, jupyter=True, style='ent')

data_doc_bionlp13cg = [(X.text, X.label_, X.start_char, X.end_char) for X in doc_bionlp13cg.ents]
data_doc_craft = [(X.text, X.label_, X.start_char, X.end_char) for X in doc_craft.ents]
data_doc_bc5cdr = [(X.text, X.label_, X.start_char, X.end_char) for X in doc_bc5cdr.ents]
data_doc_jnlpba = [(X.text, X.label_, X.start_char, X.end_char) for X in doc_jnlpba.ents]

data = data_doc_bionlp13cg + data_doc_craft + data_doc_bc5cdr + data_doc_jnlpba

attrs = ["text", "label_", "start_char", "end_char"]
temp_df = pd.DataFrame(data, columns=attrs)

temp_train_df = entities_train.query('abstract_id == 1353340')
temp_train_df.head()

merged_df = temp_df.merge(temp_train_df, how = 'inner', left_on ='text', right_on = 'mention')
merged_df[['text', 'label_', 'type']].drop_duplicates()

temp_df.label_.unique()

temp_train_df.type.unique()

temp_df.label_ = temp_df.label_.map(
    {
        "GENE_OR_GENE_PRODUCT": "GeneOrGeneProduct",
        "GGP": "GeneOrGeneProduct",
        "ORGANISM": "OrganismTaxon",
        "CANCER": "DiseaseOrPhenotypicFeature",
        "DISEASE": "DiseaseOrPhenotypicFeature",
        "CHEBI": "ChemicalEntity",
        "CHEMICAL": "ChemicalEntity",
        "PROTEIN": "SequenceVariant",
        "AMINO_ACID": "SequenceVariant",
        "SO": "SO",
        "TAXON": "TAXON",
        "DNA": "DNA",
    }
)

temp_df['start_char'] = temp_df['start_char'] + 1

temp_df.text = temp_df.text.str.replace(')', '')

# show results to streamlit
st.write(temp_df)

# streamlit bar chart of frequency of each entity
a = temp_df.label_.value_counts()
st.bar_chart(a)