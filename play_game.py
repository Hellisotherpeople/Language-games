import os
import string
from pymagnitude import *
# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score
from sklearn.preprocessing import normalize
#import numpy as np
import sys
import numpy as np
from itertools import islice
from collections import deque
import csv
from random import shuffle
from sklearn.externals import joblib
from time import time
from sklearn.pipeline import make_union, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from random_word import RandomWords
from random import shuffle 

rd_word_gen = RandomWords()

def inner_product_rank_mag(a_set, use_weights=True):
    np_of_vecs = np.asarray(a_set)
    sims_mat = np.dot(np_of_vecs, np_of_vecs.transpose())
    ranks = np.sum(sims_mat, axis = 0)
    if use_weights:
    	weighted_avg = np.average(np_of_vecs, axis = 0, weights = ranks)
    else:
    	weighted_avg = np.average(np_of_vecs, axis = 0)
    return weighted_avg






vectors = Magnitude("crawl-300d-2M.magnitude")


my_string = "myanmar politics"
#print(my_string.split())
set_vec = vectors.query(my_string.split())

pooled_vec = inner_product_rank_mag(set_vec, use_weights = True)

#print(vectors.most_similar_approx(pooled_vec, topn = 10))



def game_loop(num_turns = 10):
	print("I'm thinking of a word.....")

	random_word = rd_word_gen.get_random_word(minDictionaryCount = 5, minCorpusCount = 5).lower()
	#random_word_list = rd_word_gen.get_random_words(minDictionaryCount = 5, minCorpusCount = 5)
	#lower_random_word_list = [x.lower() for x in random_word_list]
	print("Here are some words of inspiration that may guide you along your adventure :)")

	most_sims = vectors.most_similar_approx(random_word, topn = 100, effort = 0.001)
	golden_tickets = [item[0].lower() for item in most_sims]
	canidate_words = golden_tickets #lower_random_word_list + golden_tickets
	shuffle(canidate_words)
	print(canidate_words)
	print("\n")

	player1_score = 0
	player2_score = 0

	p1_turn = True

	turns_left = num_turns
	while turns_left > 0:

		if p1_turn: 
			p1_word = input("Player 1, it is your turn to enter a word ")
			print(p1_word)
			score = vectors.similarity(p1_word, random_word)
			print(score)

			if score > player1_score:
				print("You got closer to the right word!")
				player1_score = score
		else:
			p2_word = input("Player 2, it is your turn to enter a word ")
			print(p2_word)
			score = vectors.similarity(p2_word, random_word)
			print(score)

			if score > player2_score:
				print("You got closer to the right word!")
				player2_score = score

		p1_turn = not p1_turn #flip the switch
		print("Player 1 current score: " + str(player1_score))
		print("Player 2 current score: " + str(player2_score))
		turns_left -= 1
		print("Turns left: ")
		print(turns_left)

	print("THE GAME IS NOW OVER")
	print("The randomly chosen word by the computer was: " + random_word)
	print("The canidate words in the inspiration list were:")
	print(golden_tickets)



game_loop()

