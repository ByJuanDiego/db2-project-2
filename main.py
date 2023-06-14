import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import re
from nltk.tokenize import RegexpTokenizer
from typing import List, TextIO, Tuple, Optional

import os

from Heap import MinHeap


class InvertedIndex:

    def __init__(self, collection_path: str, filename: str = "myindex"):
        self.doc_map_file_name = filename + "_docmap.invidx"
        self.index_file_name = filename + "_index.invidx"
        self.idf_file_name = filename + "_idf.invidx"
        self.length_file_name = filename + "_length.invidx"
        self.collection_path = collection_path

    """
        Preprocess all documents, and create a file 
        containing the following information for each document:
        (term, doc_id, term_frequency)
    """

    def _preprocess(self):

        pass

    def _write_block(self, buffer: any):
        pass


    """
        Merge the list of files created by the
        SPIMI-Invert algorithm into a single file,
        finishing the creation of the index
    """
    def _merge_blocks(self, blocks: List[str]) -> None:
        outfile = open(self.index_file_name, "a")
        min_heap = MinHeap[Tuple]()
        k: int = len(blocks)
        # Buffer of BLOCK_SIZE to hold elements of each
        block_files: List[TextIO] = []
        # Open all block files and extract initialize min heap of size k
        for i in range(k):
            block_files.append(open(blocks[i]))
            # min_term_tuple: (term, postings_list)
            min_term_tuple = tuple(block_files[i].readline())
            # heap elements: (term, block, postings_list)
            min_heap.push((min_term_tuple[0], i, min_term_tuple[1]))
        # Combine all posting lists of the current lowest term
        last_min_term: Optional[Tuple] = None
        while not min_heap.empty():
            # Grab the lowest term from the priority queue
            min_term_tuple = min_heap.pop()
            print(min_term_tuple)
            # If the term is equal to the current lowest, concatenate their posting lists
            if last_min_term and last_min_term[0] == min_term_tuple[0]:
                last_min_term[2].extend(min_term_tuple[2])
            # Otherwise, write the current lowest term to disk, and start processing the next term
            else:
                if last_min_term is not None:
                    # Write current term to file
                    outfile.write(str(last_min_term))
                last_min_term = min_term_tuple
            # Add next term in the ordered array of terms to the priority queue
            i = min_term_tuple[1]
            # Note that the file pointer always goes down in each file, so we can just read the next line
            next_min_term_tuple = tuple(
                block_files[i].readline())
            min_heap.push((next_min_term_tuple[0], i, next_min_term_tuple[1]))
        # Close all block files
        for i in range(k):
            block_files[i].close()

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
