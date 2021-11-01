#!/usr/bin/env python3
# Student name: NAME
# Student number: NUMBER
# UTORid: ID

import typing as T
from string import punctuation

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize


def deepest():
    """Find and print the synset with the largest maximum depth along with its
    depth on each of its hyperonym paths.

    Returns:
        None
    """
    a = wn.all_synsets()
    max_depth_node = [0,None]
    for ele in a:
        if ele.max_depth()>max_depth_node[0]:
            max_depth_node[0] = ele.max_depth()
            max_depth_node[1] = ele
    print(max_depth_node)
    result_synset = max_depth_node[1]
    for path in result_synset.hypernym_paths():
        print(len(path))

    # raise NotImplementedError


def superdefn(s: str) -> T.List[str]:
    """Get the "superdefinition" of a synset. (Yes, superdefinition is a
    made-up word. All words are made up...)

    We define the superdefinition of a synset to be the list of word tokens,
    here as produced by word_tokenize, in the definitions of the synset, its
    hyperonyms, and its hyponyms.

    Args:
        s (str): The name of the synset to look up

    Returns:
        list of str: The list of word tokens in the superdefinition of s

    Examples:
        >>> superdefn('toughen.v.01')
        ['make', 'tough', 'or', 'tougher', 'gain', 'strength', 'make', 'fit']
    """
    result = ''
    synset = wn.synset(s)

    result += synset.definition()
    for hypernym in synset.hypernyms():
        result+= ' '
        result += hypernym.definition()
    for hyponym in synset.hyponyms():
        result += ' '
        result += hyponym.definition()

    return word_tokenize(result)



def stop_tokenize(s: str) -> T.List[str]:
    """Word-tokenize and remove stop words and punctuation-only tokens.

    Args:
        s (str): String to tokenize

    Returns:
        list[str]: The non-stopword, non-punctuation tokens in s

    Examples:
        >>> stop_tokenize('The Dance of Eternity, sir!')
        ['Dance', 'Eternity', 'sir']
    """

    word_token = word_tokenize(s)
    for ele in word_token:
        lc_ele = ele.lower()
        if lc_ele in punctuation or lc_ele in stopwords.words('english'):
            word_token.remove(ele)
    return word_token



if __name__ == '__main__':
    import doctest
    # deepest()
    doctest.testmod()
