#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2020-2021: Homework 3
parser_transitions.py: Algorithms for completing partial parsess.
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""

import sys

class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        @param sentence (list of str): The sentence to be parsed as a list of words.
        """
        self.sentence = sentence
        self.stack = ['ROOT']
        self.buffer = self.sentence.copy() 
        self.dependencies = list()
        
    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse

        @param transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. You can assume the provided
                                transition is a legal transition.
        """

        if transition is 'S': self.stack.append(self.buffer.pop(0))
        elif transition is 'LA': self.dependencies.append((self.stack[-1], self.stack.pop(-2)))
        else: self.dependencies.append((self.stack[-2], self.stack.pop(-1)))

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        @param transitions (list of str): The list of transitions in the order they should be applied

        @return dependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. Represented as a list of
                                                        tuples where each tuple is of the form (head, dependent).
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    @param sentences (list of list of str): A list of sentences to be parsed
                                            (each sentence is a list of words and each word is of type string)
    @param model (ParserModel): The model that makes parsing decisions. It is assumed to have a function
                                model.predict(partial_parses) that takes in a list of PartialParses as input and
                                returns a list of transitions predicted for each parse. That is, after calling
                                    transitions = model.predict(partial_parses)
                                transitions[i] will be the next transition to apply to partial_parses[i].
    @param batch_size (int): The number of PartialParses to include in each minibatch


    @return dependencies (list of dependency lists): A list where each element is the dependencies
                                                    list for a parsed sentence. Ordering should be the
                                                    same as in sentences (i.e., dependencies[i] should
                                                    contain the parse for sentences[i]).
    """
    dependencies = []

    partial_parses = [PartialParse(s) for s in sentences]
    unfinished_parses = partial_parses[:] #shallow copy

    while unfinished_parses:
        mini_batch = unfinished_parses[:batch_size]
        next_trans = model.predict(mini_batch)
        for pp, t in zip(mini_batch, next_trans): #pp: each partial parse, t: next  step(transition) from model on pp
            pp.parse_step(t)
            if len(pp.stack)==1 and len(pp.buffer)==0:
                unfinished_parses.remove(pp)

    dependencies = [pp.dependencies for pp in partial_parses]   


    return dependencies


