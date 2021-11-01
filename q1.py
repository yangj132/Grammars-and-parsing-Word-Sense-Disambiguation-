#!/usr/bin/env python3
# Student name: NAME
# Student number: NUMBER
# UTORid: ID

from collections import Counter
from typing import *

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import numpy as np
from numpy.linalg import norm

from q0 import stop_tokenize
from wsd import evaluate, load_eval, load_word2vec, WSDToken


def mfs(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Most frequent sense of a word.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. See the WSDToken class in wsd.py
    for the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The most frequent sense for the given word.
    """
    target_word = sentence[word_index]
    target_synset = wn.synsets(target_word.lemma)

    return target_synset[0]











def lesk(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Simplified Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sentence,word_index)
    best_score = 0

    context_map = {}
    context = ''
    for word in sentence:
        if word not in context:
            context += (word.wordform)
            context += ' '
    context = stop_tokenize(context)
    for word in context:

        if word in context_map:
            context_map[word] += 1
        else:
            context_map[word] = 1


    synsets = wn.synset(sentence[word_index].lemma)

    for synset in synsets:

        definition = stop_tokenize(synset.definition())
        examples = []

        for example in synset.examples():
            tokenize_example = stop_tokenize(example)
            examples.extend(tokenize_example)
        signature = definition + examples
        signature_map = {}
        for sign in signature:
            if sign not in signature_map:
                signature_map[sign] = 1
            else:
                signature_map[sign] += 1
        score = 0
        for word in context:
            if word in signature:
                overlap = min(signature_map[word],context_map[word])
                score+=overlap
        if score > best_score:
            best_sense = synset
            best_score = score
    return  best_sense

def create_bag(list,map,synset)->None:
    definition = stop_tokenize(synset.definition())
    examples = []

    for example in synset.examples():
        tokenize_example = stop_tokenize(example)
        examples.extend(tokenize_example)
    signature = definition + examples
    list.extend(signature)
    for sign in signature:
        if sign not in map:
            map[sign] = 1
        else:
            map[sign] += 1

def create_total_map(total_map,list,synset) -> None:
    definition = stop_tokenize(synset.definition())
    examples = []

    for example in synset.examples():
        tokenize_example = stop_tokenize(example)
        examples.extend(tokenize_example)
    signature = definition + examples
    list.extend(signature)
    for ele in signature:
        if ele not in total_map:
            total_map[ele] = 0




def lesk_ext(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """






    best_sense = mfs(sentence, word_index)
    best_score = 0

    context_map = {}

    context = ''
    for word in sentence:
        if word not in context:
            context += (word.wordform)
            context += ' '
    context = stop_tokenize(context)
    for word in context:

        if word in context_map:
            context_map[word] += 1
        else:
            context_map[word] = 1

    synsets = wn.synset(sentence[word_index].lemma)

    for synset in synsets:
        signature = []
        signature_map = {}
        create_bag(signature,signature_map,synset)
        for hyponym in synset.hyponyms():
            create_bag(signature,signature_map,hyponym)

        for part_meronym in synset.part_meronyms():
            create_bag(signature,signature_map,part_meronym)
        for substance_meronym in synset.substance_meronyms():
            create_bag(signature, signature_map, substance_meronym)
        for member_meronym in synset.member_meronyms():
            create_bag(signature,signature_map,member_meronym)

        for part_holonym in synset.part_holonyms():
            create_bag(signature,signature_map,part_holonym)
        for substance_holonym in synset.substance_holonyms():
            create_bag(signature, signature_map, substance_holonym)
        for member_holonym in synset.member_holonyms():
            create_bag(signature,signature_map,member_holonym)



        score = 0
        for word in context:
            if word in signature:
                overlap = min(signature_map[word], context_map[word])
                score += overlap
        if score > best_score:
            best_sense = synset
            best_score = score
    return best_sense






def lesk_cos(sentence: Sequence[WSDToken], word_index: int) -> Synset:

    """Extended Lesk algorithm using cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """


    best_sense = mfs(sentence, word_index)
    best_score = 0
    context_map = {}
    context = ''
    for word in sentence:
        if word not in context:
            context += (word.wordform)
            context += ' '
    context = stop_tokenize(context)

    for word in context:

        if word not in context_map:
            context_map[word] = 0

    synsets = wn.synset(sentence[word_index].lemma)

    for synset in synsets:

        signature = []
        total_map = context_map

        create_total_map(total_map, signature,synset)
        for hyponym in synset.hyponyms():
            create_total_map(total_map, signature,hyponym)

        for part_meronym in synset.part_meronyms():
            create_total_map(total_map, signature,part_meronym)
        for substance_meronym in synset.substance_meronyms():
            create_total_map(total_map, signature,substance_meronym)
        for member_meronym in synset.member_meronyms():
            create_total_map(total_map, signature,member_meronym)

        for part_holonym in synset.part_holonyms():
            create_total_map(total_map, signature,part_holonym)
        for substance_holonym in synset.substance_holonyms():
            create_total_map(total_map,signature,substance_holonym)
        for member_holonym in synset.member_holonyms():
            create_total_map(total_map, signature,member_holonym)

        new_context_map = total_map
        signature_map = total_map
        for ele in context:
            if ele in new_context_map:
                new_context_map[ele]+=1
        for ele in signature:
            if ele in signature_map:
                signature_map[ele]+=1
        context_vector = []
        signature_vector = []
        for key in new_context_map:
            context_vector.append(new_context_map[key])
        for key in signature_map:
            signature_vector.append(signature_map[key])


        context_vector = np.array(context_vector)
        signature_vector = np.array(signature_vector)
        context_vector_norm = np.linalg.norm(context_vector)
        signature_vector_norm = np.linalg.norm(signature_vector)
        if (context_vector_norm != 0 and signature_vector_norm != 0):
            score = (context_vector / context_vector_norm) * (signature_vector / signature_vector_norm)
            if score > best_score:
                best_sense = synset
                best_score = score

    return  best_sense













def lesk_cos_onesided(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm using one-sided cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sentence, word_index)
    best_score = 0
    context_map = {}
    context = ''
    for word in sentence:
        if word not in context:
            context += (word.wordform)
            context += ' '
    context = stop_tokenize(context)

    for word in context:

        if word not in context_map:
            context_map[word] = 0

    synsets = wn.synset(sentence[word_index].lemma)
    new_context_map = context_map
    for ele in context:
        if ele in context_map:
            new_context_map[ele]+=1

    for synset in synsets:
        signature = []
        signature_map = {}
        new_signature_map = context_map
        create_bag(signature, signature_map, synset)
        for hyponym in synset.hyponyms():
            create_bag(signature, signature_map, hyponym)

        for part_meronym in synset.part_meronyms():
            create_bag(signature, signature_map, part_meronym)
        for substance_meronym in synset.substance_meronyms():
            create_bag(signature, signature_map, substance_meronym)
        for member_meronym in synset.member_meronyms():
            create_bag(signature, signature_map, member_meronym)

        for part_holonym in synset.part_holonyms():
            create_bag(signature, signature_map, part_holonym)
        for substance_holonym in synset.substance_holonyms():
            create_bag(signature, signature_map, substance_holonym)
        for member_holonym in synset.member_holonyms():
            create_bag(signature, signature_map, member_holonym)

        for ele in signature:
            if ele in context_map:
                new_signature_map[ele]+=1




        context_vector = []
        signature_vector = []
        for key in new_context_map:
            context_vector.append(new_context_map[key])
        for key in new_signature_map:
            signature_vector.append(new_signature_map[key])

        context_vector = np.array(context_vector)
        signature_vector = np.array(signature_vector)
        context_vector_norm = np.linalg.norm(context_vector)
        signature_vector_norm = np.linalg.norm(signature_vector)



        if (context_vector_norm != 0 and signature_vector_norm != 0):
            score = (context_vector / context_vector_norm) * (signature_vector / signature_vector_norm)
            if score > best_score:
                best_sense = synset
                best_score = score
    return  best_sense










def lesk_w2v(sentence: Sequence[WSDToken], word_index: int,
             vocab: Mapping[str, int], word2vec: np.ndarray) -> Synset:
    """Extended Lesk algorithm using word2vec-based cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    To look up the vector for a word, first you need to look up the word's
    index in the word2vec matrix, which you can then use to get the specific
    vector. More directly, you can look up a string s using word2vec[vocab[s]].

    To look up the vector for a *single word*, use the following rules:
    * If the word exists in the vocabulary, then return the corresponding
      vector.
    * Otherwise, if the lower-cased version of the word exists in the
      vocabulary, return the corresponding vector for the lower-cased version.
    * Otherwise, return a vector of all zeros. You'll need to ensure that
      this vector has the same dimensions as the word2vec vectors.

    But some wordforms are actually multi-word expressions and contain spaces.
    word2vec can handle multi-word expressions, but uses the underscore
    character to separate words rather than spaces. So, to look up a string
    that has a space in it, use the following rules:
    * If the string has a space in it, replace the space characters with
      underscore characters and then follow the above steps on the new string
      (i.e., try the string as-is, then the lower-cased version if that
      fails), but do not return the zero vector if the lookup fails.
    * If the version with underscores doesn't yield anything, split the
      string into multiple words according to the spaces and look each word
      up individually according to the rules in the above paragraph (i.e.,
      as-is, lower-cased, then zero). Take the mean of the vectors for each
      word and return that.
    Recursion will make for more compact code for these.


    In the lesk_w2v function, implement a variant of your lex_cos function where the
    vectors for the signature and context are constructed by taking the mean of the word2vec
    vectors for the words in the signature and sentence, respectively. Count each word once only;
    i.e., treat the signature and context as sets rather than multisets.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.
        vocab (dictionary mapping str to int): The word2vec vocabulary,
            mapping strings to their respective indices in the word2vec array.
        word2vec (np.ndarray): The word2vec word vectors, as a VxD matrix,
            where V is the vocabulary and D is the dimensionality of the word
            vectors.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    def build_set(signature_set,synset):
        definition = stop_tokenize(synset.definition())
        examples = []

        for example in synset.examples():
            tokenize_example = stop_tokenize(example)
            examples.extend(tokenize_example)
        sig = definition + examples
        for sign in sig:
            signature_set.add(sign)


    def build_signature_set(sign_set,synset):
        build_set(sign_set, synset)
        for hyponym in synset.hyponyms():
            build_set(sign_set, hyponym)

        for part_meronym in synset.part_meronyms():
            build_set(sign_set, part_meronym)
        for substance_meronym in synset.substance_meronyms():
            build_set(sign_set, substance_meronym)
        for member_meronym in synset.member_meronyms():
            build_set(sign_set, member_meronym)

        for part_holonym in synset.part_holonyms():
            build_set(sign_set, part_holonym)
        for substance_holonym in synset.substance_holonyms():
            build_set(sign_set, substance_holonym)
        for member_holonym in synset.member_holonyms():
            build_set(sign_set, member_holonym)





    def build_vector_for_set(word,word2vec,vocab):

        vector_length = word2vec.shape[1]

        word_split = word.split()
        if len(word_split)>1:
            word_replace = word.replace(' ','_')
            try:

                return word2vec[vocab[word_replace]]
            except KeyError:
                try:

                    return word2vec[vocab[word_replace.lower()]]
                except KeyError:
                    set_vector = np.zeros(vector_length)
                    for w in word_split:
                        set_vector += build_vector_for_set(w,word2vec,vocab)
                    return set_vector/len(word_split)
        else:
            try:
                result = word2vec[vocab[word]]
                return result
            except KeyError:
                try:
                    return word2vec[vocab[word.lower()]]

                except KeyError:
                    return np.zeros(vector_length)












    best_sense = mfs(sentence, word_index)
    best_score = 0
    context = ''
    context_set = set()


    for word in sentence:
        if word not in context:
            context += (word.wordform)
            context += ' '
    context = stop_tokenize(context)



    for word in context:
        context_set.add(word)

    context_vector_sum = np.zeros(word2vec.shape[1])
    for con in context_set:
        context_vector_sum += build_vector_for_set(con, word2vec, vocab)
    context_vector = context_vector_sum / len(context_set)


    synsets = wn.synset(sentence[word_index].lemma)

    for synset in synsets:
        signature_set = set()

        build_signature_set(signature_set,synset)
        signature_vector_sum = np.zeros(word2vec.shape[1])
        for sign in signature_set:
            signature_vector_sum += build_vector_for_set(sign,word2vec,vocab)
        signature_vector = signature_vector_sum / len(signature_set)
        context_vector_norm = np.linalg.norm(context_vector)
        signature_vector_norm = np.linalg.norm(signature_vector)

        if (context_vector_norm != 0 and signature_vector_norm != 0):
            score = (context_vector / context_vector_norm) * (signature_vector / signature_vector_norm)
            if score > best_score:
                best_sense = synset
                best_score = score
    return best_sense









if __name__ == '__main__':
    np.random.seed(1234)
    eval_data = load_eval()
    for wsd_func in [mfs, lesk, lesk_ext, lesk_cos, lesk_cos_onesided]:
        evaluate(eval_data, wsd_func)

    evaluate(eval_data, lesk_w2v, *load_word2vec())
