import pandas as pd
import numpy as np

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

# Data Visualization
import matplotlib.pyplot as plt

# surpress warnings
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import altair as alt

# importando los datasets
abstracts_train = pd.read_csv('./abstracts_train.csv', sep='\t')  
entities_train = pd.read_csv('./entities_train.csv', sep='\t')  

# new column that join column title and abstract
temp_t_a = abstracts_train
temp_t_a['title_abstract'] = temp_t_a['title'] + ' ' + temp_t_a['abstract']
# remove title and abstract columns
temp_t_a = temp_t_a.drop(['title', 'abstract'], axis=1)

ABSTRACTS_LIMIT = 10
st.header('Abstracts Train')
st.write('Escribe una cantidad de abstracts para sacar las metricas')
abstracts_limit_input = st.text_input('Cantidad de abstracts', ABSTRACTS_LIMIT)
st.write(f'Por defecto se muestran resultados para {ABSTRACTS_LIMIT} abstracts')

# df de abstracts limitado
temp_t_a = temp_t_a.iloc[:int(abstracts_limit_input)]
# mostrar abstracts limitados
st.write(temp_t_a)

evaluation_df = pd.DataFrame(columns=['abstract_id', 'correct_predictions', 'missed_predictions', 'false_predictions', 'precision', 'recall', 'f1'])

for abstract_id, title_abstract in temp_t_a.iterrows():
    text = title_abstract['title_abstract']
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

    attrs = ["mention", "type", "offset_start", "offset_finish"]
    temp_df = pd.DataFrame(data, columns=attrs)
    predictions = temp_df.copy()
    # elimina las columnas de offset que no se usaran para comparar
    predictions = predictions.drop(['offset_start', 'offset_finish'], axis=1)

    temp_train_df = entities_train.query(f'abstract_id == {title_abstract["abstract_id"]}')
    true_entities = temp_train_df.copy()
    # elimina las columnas de offset que no se usaran para comparar
    true_entities = true_entities.drop(['id', 'abstract_id', 'offset_start', 'offset_finish', 'entity_ids'], axis=1)

    predictions.type = predictions.type.map(
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
            "ORGANISM_SUBDIVISION": "OrganismTaxon",
            "ORGANISM_SUBSTANCE": "OrganismTaxon",
            "TAXON": "OrganismTaxon",
            "CL": "CellLine",
            "CELL_LINE": "CellLine",
            "CELL": "CellLine",
            "SIMPLE_CHEMICAL": "ChemicalEntity",
        }
    )

    # Obtiene las intersecciones entre las predicciones y las entidades reales
    correct_predictions = predictions.merge(true_entities, how="inner", on=["mention", "type"])
    correct_predictions = correct_predictions.drop_duplicates(subset=['mention', 'type'], keep='first')
    # obtiene las reales que no se predijeron
    missed_predictions = true_entities[~true_entities.mention.isin(predictions.mention)]
    missed_predictions = missed_predictions.drop_duplicates(subset=['mention', 'type'], keep='first')
    # obtiene las predicciones que no son reales
    false_predictions = predictions[~predictions.mention.isin(true_entities.mention)]
    false_predictions = false_predictions.drop_duplicates(subset=['mention', 'type'], keep='first')

    # add row to evaluation_df
    evaluation_df.loc[abstract_id] = [
        title_abstract['abstract_id'],
        len(correct_predictions),
        len(missed_predictions),
        len(false_predictions),
        len(correct_predictions) / (len(correct_predictions) + len(false_predictions)), # precision
        len(correct_predictions) / (len(correct_predictions) + len(missed_predictions)), # recall
        2 * (len(correct_predictions) / (len(correct_predictions) + len(false_predictions))) * (len(correct_predictions) / (len(correct_predictions) + len(missed_predictions))) / ((len(correct_predictions) / (len(correct_predictions) + len(false_predictions))) + (len(correct_predictions) / (len(correct_predictions) + len(missed_predictions))))
    ]
    print(f'abstract_id: {abstract_id}')


st.title('Métricas')

# Mostrar gráfica de evaluación para las métricas, se eliminan las columnas de resultados
metrics_df = evaluation_df.drop(['correct_predictions', 'missed_predictions', 'false_predictions'], axis=1)
# grouped bar chart
st.subheader('Gráfico de barras agrupadas para métricas estadísticas')
metrics_chart = st.empty()
metrics_chart_alt = alt.Chart(metrics_df).transform_fold(
    ['precision', 'recall', 'f1'],
    as_=['metric', 'value']
).mark_bar().encode(
    x=alt.X('metric:N', axis=alt.Axis(title='Abstract ID')),
    y=alt.Y('value:Q', axis=alt.Axis(title='Valor')),
    color=alt.Color('metric:N', legend=alt.Legend(title='Metrica')),
    column=alt.Column('abstract_id:N', header=alt.Header(labelOrient='bottom', title='Abstract ID'))
).properties(
    width=50,
    height=200
)
metrics_chart.altair_chart(metrics_chart_alt)

# Mostrar gráfica de evaluación para los resultados, se eliminan las columnas de métricas
results_df = evaluation_df.drop(['precision', 'recall', 'f1'], axis=1)
st.subheader('Gráfico de barras agrupadas para frecuencias de resultados')
results_chart = st.empty()
results_chart_alt = alt.Chart(results_df).transform_fold(
    ['correct_predictions', 'missed_predictions', 'false_predictions'],
    as_=['result', 'value']
).mark_bar().encode(
    x=alt.X('result:N', axis=alt.Axis(title='Abstract ID')),
    y=alt.Y('value:Q', axis=alt.Axis(title='Valor')),
    color=alt.Color('result:N', legend=alt.Legend(title='Resultado')),
    column=alt.Column('abstract_id:N', header=alt.Header(labelOrient='bottom', title='Abstract ID'))
).properties(
    width=50,
    height=200
)
results_chart.altair_chart(results_chart_alt)

st.subheader('Gráfico de dona para promedio de estadísticas')
# Grafico de dona de recall
recall_mean_df = pd.DataFrame({
    'metric': ['recall', 'failure'],
    'value': [evaluation_df.recall.mean(), 1 - evaluation_df.recall.mean()]
})
st.write('El promedio de la métrica de recall es de: ', evaluation_df.recall.mean())
recall_donut = st.empty()
recall_donut_base = alt.Chart(recall_mean_df).encode(
    theta=alt.Theta('value:Q', stack=True),
    color=alt.Color('metric:N', scale=alt.Scale(range=['lightgray', 'steelblue']))
)
recall_donut_pie = recall_donut_base.mark_arc(innerRadius=100, outerRadius=50).encode(
    tooltip=['metric:N', 'value:Q']
)
recall_donut_text = recall_donut_base.mark_text(radius=140, size=10).encode(
    text='value:Q'
)
recall_donut_alt = recall_donut_pie + recall_donut_text
recall_donut.altair_chart(recall_donut_alt)

# Grafico de dona de precision
precision_mean_df = pd.DataFrame({
    'metric': ['precision', 'failure'],
    'value': [evaluation_df.precision.mean(), 1 - evaluation_df.precision.mean()]
})
st.write('El promedio de la métrica de precisión es de: ', evaluation_df.precision.mean())
precision_donut = st.empty()
precision_donut_base = alt.Chart(precision_mean_df).encode(
    theta=alt.Theta('value:Q', stack=True),
    color=alt.Color('metric:N', scale=alt.Scale(range=['lightgray', 'steelblue']))
)
precision_donut_pie = precision_donut_base.mark_arc(innerRadius=100, outerRadius=50).encode(
    tooltip=['metric:N', 'value:Q']
)
precision_donut_text = precision_donut_base.mark_text(radius=140, size=10).encode(
    text='value:Q'
)
precision_donut_alt = precision_donut_pie + precision_donut_text
precision_donut.altair_chart(precision_donut_alt)

# Grafico de dona de f1
f1_mean_df = pd.DataFrame({
    'metric': ['f1', 'failure'],
    'value': [evaluation_df.f1.mean(), 1 - evaluation_df.f1.mean()]
})
st.write('El promedio de la métrica de f1 es de: ', evaluation_df.f1.mean())
f1_donut = st.empty()
f1_donut_base = alt.Chart(f1_mean_df).encode(
    theta=alt.Theta('value:Q', stack=True),
    color=alt.Color('metric:N', scale=alt.Scale(range=['steelblue', 'lightgray']))
)
f1_donut_pie = f1_donut_base.mark_arc(innerRadius=100, outerRadius=50).encode(
    tooltip=['metric:N', 'value:Q']
)
f1_donut_text = f1_donut_base.mark_text(radius=140, size=10).encode(
    text='value:Q'
)
f1_donut_alt = f1_donut_pie + f1_donut_text
f1_donut.altair_chart(f1_donut_alt)

# Usar modelos con texto ingresado
st.header('Usa el modelo tú mismo')

st.subheader('Ingresa el texto de un abstracto para obtener las entidades')

st.write('Estos son algunos abstractos de prueba:')
abstracts_test = pd.read_csv('./abstracts_test.csv', sep='\t')
abstracts_test['title_abstract'] = abstracts_test['title'] + ' ' + abstracts_test['abstract']
abstracts_test = abstracts_test.drop(['title', 'abstract'], axis=1)
st.write(abstracts_test)

# receive a text from the user
text = st.text_input("Ingresa el texto del abstracto", abstracts_test.title_abstract[0])
st.write("Puedes cambiarlo y presionar enter para ver los resultados")

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
        "ORGANISM_SUBDIVISION": "OrganismTaxon",
        "ORGANISM_SUBSTANCE": "OrganismTaxon",
        "TAXON": "OrganismTaxon",
        "CL": "CellLine",
        "CELL_LINE": "CellLine",
        "CELL": "CellLine",
        "SIMPLE_CHEMICAL": "ChemicalEntity",
    }
)

temp_df['start_char'] = temp_df['start_char'] + 1

temp_df['text'] = temp_df['text'].str.replace(r'\)', '', regex=True)

st.subheader('Entidades extraídas')
# show results to streamlit
st.write(temp_df)

st.subheader('Frecuencia de entidades por tipo')
# streamlit bar chart of frequency of each entity
a = temp_df.label_.value_counts()
st.bar_chart(a)
