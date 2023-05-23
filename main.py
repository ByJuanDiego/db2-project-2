import random
import sys

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, QWidget)
import random

import nltk
from nltk.stem import SnowballStemmer
import xml.etree.ElementTree as eT
import json
import numpy as np
import os


class InvertIndex: 


    def __init__(self, index_file, relative_path):
        self.relative_path = relative_path
        self.index_file = index_file
        self.index = {}
        self.idf = {}
        self.tf_idf = {}
        self.length = {}

        try:
            os.mkdir(relative_path)
        except FileExistsError:
            pass


    @staticmethod
    def get_stopwords(stopwordsfilename):
        stopwordslist = []
        tree = eT.parse(stopwordsfilename)
        root = tree.getroot()
    
        for words in root.findall('word'):
            stopwordslist.append(words.text)
        stopwordslist += ['.', '?', '¿', '-', '!', '\'', ',', '«', '»', 'con', ';', ':', '111º', '', '\'\'', '``', '(', ')']
        
        return stopwordslist
    

    @staticmethod
    def get_keywords(filename, stopwordslist):

        with open(filename, 'r', encoding='UTF-8') as l1:
            text = l1.read()
        
        stemmer = SnowballStemmer('spanish')
        palabras = nltk.word_tokenize(text.lower())
        palabras = [stemmer.stem(palabra) for palabra in palabras if stopwordslist.count(palabra) == 0]
        
        return palabras
    

    @staticmethod
    def get_tf(collection_text, stop_words_file = 'stop_words_spanish.xml'):
        stopwords = InvertIndex.get_stopwords(stop_words_file)

        tf = dict()
        n = len(collection_text)

        for i in range(n):
            for key in InvertIndex.get_keywords(collection_text[i], stopwords):

                if tf.get(key) is None:        
                    tf[key] = {}

                if tf[key].get(collection_text[i]) is None:
                    tf[key][collection_text[i]] = 0

                tf[key][collection_text[i]] += 1       # agregar uno al contador
        
        for key in tf.keys():
            for book in tf[key].keys():
                tf[key][book] = np.log10(tf[key][book] + 1) # usamos la formula del log frecuency weight
        
        return tf
        
    
    @staticmethod
    def get_idf(collection_text, stop_words_file = 'stop_words_spanish.xml'):
        stopwords = InvertIndex.get_stopwords(stop_words_file)

        idf = dict()
        n = len(collection_text)

        for i in range(n):
 
            for key in set(InvertIndex.get_keywords(collection_text[i], stopwords)):

                if idf.get(key) is None:
                    idf[key] = 0  # Usar un conjunto en lugar de una lista
                
                idf[key] += 1

        for key in idf.keys():
            idf[key] = np.log10(n / idf[key])

        return idf

    
    def get_tf_idf(self, collection_text):
        tf_idf = dict()
        n = len(collection_text)
        
        for i in range(n):
            
            for key in self.idf.keys():

                if tf_idf.get(key) is None:
                    tf_idf[key] = {}

                k_tf = self.index[key][collection_text[i]] if self.index[key].get(collection_text[i]) is not None else 0
                k_idf = self.idf[key]

                w = k_tf * k_idf

                tf_idf[key].update({collection_text[i]: w})
        
        return tf_idf

    
    def get_length(self):
        length = dict()

        for key in self.tf_idf.keys():
            for doc in self.tf_idf[key]:
                
                if length.get(doc) is None:
                    length[doc] = 0
                
                length[doc] += np.power(self.tf_idf[key][doc], 2)
        
        for doc in length:
            length[doc] = np.sqrt(length[doc])

        return length


    def building(self, collection_text):
        # build the inverted index with the collection
        
        # compute the tf
        self.index = self.get_tf(collection_text)

        # compute the idf
        self.idf = self.get_idf(collection_text)

        # compute the tf-idf of the collection text
        self.tf_idf = self.get_tf_idf(collection_text)

        # compute the length (norm)
        self.length = self.get_length()

        # store in disk
        with open(self.relative_path + self.index_file, 'w', encoding='utf-8') as tf_file:
            json.dump(self.index, tf_file, ensure_ascii=False, indent=3)

        with open(self.relative_path + 'idf.json', 'w', encoding='utf-8') as idf_file:
            json.dump(self.idf, idf_file, ensure_ascii=False, indent=3)

        with open(self.relative_path + 'tf_idf.json', 'w', encoding='utf-8') as tf_idf_file:
            json.dump(self.tf_idf, tf_idf_file, ensure_ascii=False, indent=3)

        with open(self.relative_path + 'length.json', 'w', encoding='utf-8') as length_file:
            json.dump(self.length, length_file, ensure_ascii=False, indent=3)
    
    
    def load_index(self):
        with open(self.relative_path + self.index_file, 'r', encoding='utf-8') as tf_file:
            self.index = json.load(tf_file)

        with open(self.relative_path + 'idf.json', 'r', encoding='utf-8') as idf_file:
            self.idf = json.load(idf_file)

        with open(self.relative_path + 'tf_idf.json', 'r', encoding='utf-8') as tf_idf_file:
            self.tf_idf = json.load(tf_idf_file)

        with open(self.relative_path + 'length.json', 'r', encoding='utf-8') as length_file:
            self.length = json.load(length_file)
    
    
    def cosine_sim(self, vquery):
        score = dict()

        for doc in self.length.keys():
            
            vdoc = [0] * len(self.tf_idf)
            
            i = 0
            for key in self.index.keys():
                vdoc[i] = self.tf_idf[key][doc] / self.length[doc]
                i = i + 1
            
            vdoc = np.array(vdoc)
            score.update({doc : np.dot(vquery, vdoc)})

        return score

    
    def retrieval(self, query, k):
        self.load_index()

        # diccionario para el score
        score = {}

        # preprocesar la query: extraer los terminos unicos
        stemmer = SnowballStemmer('spanish')
        stopwords = self.get_stopwords('stop_words_spanish.xml')

        terminos = nltk.word_tokenize(query.lower())
        terminos = [stemmer.stem(termino) for termino in terminos if stopwords.count(termino) == 0]

        # calcular el tf-idf del query
        
        # calcular tf
        query_tf = dict()

        for termino in terminos:
            if self.idf.get(termino) is None:
                continue
            
            if query_tf.get(termino) is None:
                query_tf[termino] = 0

            query_tf[termino] += 1
        
        for key in query_tf.keys():
            query_tf[key] = np.log10(query_tf[key] + 1)

        # calcular idf
        terminos = set(terminos)

        query_idf = dict()

        for termino in terminos:
            if self.idf.get(termino) is not None:
                query_idf[termino] = self.idf[termino]
        
        vquery = [0] * len(self.idf)

        i = 0
        for key in self.index.keys():
            vquery[i] = (query_idf[key] * query_tf[key]) if query_idf.get(key) is not None else 0
            i = i + 1
        
        vquery = np.array(vquery)
        norm = np.linalg.norm(vquery)
        vquery = vquery / norm

        # aplicar similitud de coseno y guardarlo en el diccionario score
        score = self.cosine_sim(vquery)

        # ordenar el score de forma descendente
        result = sorted(score.items(), key = lambda tup: tup[1], reverse=True)
        # retornamos los k documentos mas relevantes (de mayor similitud al query)
        return result[:k]


class MyWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.hello = [
            "Hello World",
            "Raaaa",
            "Lo maximo",
            "ISAM",
            "pq?",
        ]

        self.button = QPushButton("Click me!")
        self.message = QLabel(random.choice(self.hello))
        self.message.alignment = Qt.AlignCenter

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.message)
        self.layout.addWidget(self.button)

        # Connecting the signal
        self.button.clicked.connect(self.magic)

    @Slot()
    def magic(self):
        self.message.text = random.choice(self.hello)


if __name__ == "__main__":

    # instala los paquetes de la libreria nltk (solo se corre una vez)
    # nltk.download('punkt')

    collection = ['libro1.txt', 'libro2.txt', 'libro3.txt', 'libro4.txt', 'libro5.txt', 'libro6.txt']
    query = "Casi veinte años después, Gandalf regresa a Bolsón Cerrado y le cuenta a Frodo lo que había descubierto sobre el Anillo: que se trataba del mismo que el Rey Isildur de Arnor le había arrebatado al Señor oscuro Sauron y que muchos años después había sido encontrado por la criatura Gollum tras haberse perdido en el río Anduin durante el Desastre de los Campos Gladios. Ambos quedaron entonces en reunirse de nuevo en la aldea de Bree con el fin de llevar luego el Anillo Único a Rivendel, donde los sabios decidirían sobre su destino. Junto con su jardinero Samsagaz Gamyi, Frodo traza un plan para salir de la Comarca con el pretexto de irse a vivir a Los Gamos; pero el plan acaba siendo descubierto por otros dos amigos, Pippin y Merry, que deciden acompañarle también."

    inverted_index = InvertIndex(index_file = 'tf.json', relative_path = 'inverted_index/')

    # este metodo solo se deberia correr una vez ya que inicializa el indice en disco, no es necesario
    # correrlo mas veces.
    # inverted_index.building(collection_text = collection)

    result = inverted_index.retrieval(query = query, k = 5)

    for book in result:
        print(book)

    app = QApplication(sys.argv)

    widget = MyWidget()
    widget.show()

    sys.exit(app.exec_())
