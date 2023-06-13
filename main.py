import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import re
from nltk.tokenize import RegexpTokenizer
from typing import List

import os

nltk.download('punkt')


class InvertedIndex:

    def __init__(self, collection_path: str, filename: str = "myindex"):
        self.doc_map_file_name = filename + "_docmap.invidx"
        self.index_file_name = filename + "_index.invidx"
        self.idf_file_name = filename + "_idf.invidx"
        self.length_file_name = filename + "_length.invidx"
        self.collection_path = collection_path

    """
        Preprocess all documents, and create a temporary
        file containing the following information:
        (term, doc_id, term_frequency) #doc_id: posicion logica
    """
    def _preprocess(self):
        
        pass


    def _merge_blocks(self, blocks: List[str]):
        block_buffers: List[] = []
        for i in range(len(blocks)):
            with open(blocks[i]) as local_index:

        pass

    """
        Build the inverted index file with the collection using
        Single Pass In-Memory Indexing
    """
    def create(self):

        for file in os.listdir(self.collection_path):
            filename = os.fsdecode(file)
            if filename.endswith(".asm") or filename.endswith(".py"):
                # print(os.path.join(directory, filename))
                continue
            else:
                continue

        with open(self.collection_path) as file:
            pass
        pass

