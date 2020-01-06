import os
import string
from pymagnitude import *
import sys
import numpy as np
from itertools import islice
from collections import deque
import csv
from random import shuffle
from time import time
import random
from collections import Counter




def inner_product_rank_mag(a_set, use_weights=True):
    np_of_vecs = np.asarray(a_set)
    sims_mat = np.dot(np_of_vecs, np_of_vecs.transpose())
    ranks = np.sum(sims_mat, axis = 0)
    if use_weights:
    	weighted_avg = np.average(np_of_vecs, axis = 0, weights = ranks)
    else:
    	weighted_avg = np.average(np_of_vecs, axis = 0)
    return weighted_avg


def load_words(topic_word = "market", num_words = 10000):
	if topic_word:
		most_sims = vectors.most_similar_approx(topic_word, topn = num_words, effort = 0.5)
		sim_words = [item[0].lower() for item in most_sims]
		valid_words = sim_words
	else:
		with open('words_alpha.txt') as word_file:
			valid_words = set(word_file.read().split())

	return valid_words



def get_random_words(num_words = 5):
	sample = random.sample(english_words, num_words)
	return sample 

def get_random_word():
	sample = random.sample(english_words, 1)[0]
	return sample 


def get_topic_words(topic = "buisness", num_words = 10000):
	most_sims = vectors.most_similar_approx(topic, topn = num_words, effort = 0.01)
	golden_tickets = [item[0].lower() for item in most_sims]

def validWord(word, letterList):
	isvalid = True
	for char in word: 
		if(word.count(char) > letterList.count(char)):
			isvalid = False
			break
	return isvalid

#print(random.sample(english_words, 5))





#my_string = "For at least 20 years, upper-middle class, often tenured academics have been teaching young people that politics is a futile form of irony. I've watched Ivy League professors with tenure explain to graduate students with no health insurance that striking for pay is silly. I've heard smug male assholes with Ph.D.s describe registering voters as the"
#print(my_string.split())
#set_vec = vectors.query(my_string.lower().split())

#pooled_vec = inner_product_rank_mag(set_vec, use_weights = True)

#print(vectors.most_similar_approx(pooled_vec, topn = 10))





### Game 1, competitive word guessing 
def game_loop_guessing(num_turns = 5, num_players = 2):
	print("I'm thinking of a word.....")

	random_word = get_random_word().lower()
	#random_word_list = rd_word_gen.get_random_words(minDictionaryCount = 5, minCorpusCount = 5)
	#lower_random_word_list = [x.lower() for x in random_word_list]
	print("Here are some words of inspiration that may guide you along your adventure :)")

	most_sims = vectors.most_similar_approx(random_word, topn = 100, effort = 0.001)
	golden_tickets = [item[0].lower() for item in most_sims]
	canidate_words = golden_tickets #lower_random_word_list + golden_tickets
	shuffle(canidate_words)
	print(canidate_words)
	print("\n")


	player_scores = [0] * num_players
	cur_player = 0

	turns_left = num_turns
	while turns_left > 0:
		while cur_player < num_players:
			print("Player " + str(cur_player) + " it is your turn to enter a word ")
			p_word = input("Enter a word ")
			print(p_word)
			score = vectors.similarity(p_word, random_word)
			print(score)
			if score > player_scores[cur_player]:
				print("You got closer to the right word! ")
				player_scores[cur_player] = score

			cur_player += 1 #flip the switch
		cur_player = 0
		turns_left -= 1
		print("Turns left: ")
		print(turns_left)

	print("THE GAME IS NOW OVER ")
	print("Final Scores: ")
	print(player_scores)
	print("The randomly chosen word by the computer was: " + random_word)
	print("The canidate words in the inspiration list were:")
	print(golden_tickets)




def word_choice_logic(random_int_word = True, num_words = 5):
	### game 2 helper function
	print("I'm thinking of a word.....")
	if random_int_word:
		given_word = get_random_word().lower()
	else:
		given_word = input("give a word that you will have to select the most similar word from")

	random_word_list = get_random_words(num_words = num_words)
	lower_random_word_list = [x.lower() for x in random_word_list]

	correct_word = vectors.most_similar_to_given(given_word, lower_random_word_list)

	return given_word, lower_random_word_list, correct_word



	### Game 2, guess closest word to a given word
def game_loop_guessing_to_given(num_turns = 5, num_players = 2, random_int_word = False, num_words = 5):
	print("I'm thinking of some words.....")
	if random_int_word:
		given_word = get_random_word().lower()
	else:
		given_word = input("give a word that you will have to select the most similar word from")

	random_word_list = get_random_words(num_words = num_words)
	lower_random_word_list = [x.lower() for x in random_word_list]

	correct_word = vectors.most_similar_to_given(given_word, lower_random_word_list)
	player_scores = [0] * num_players
	cur_player = 0

	turns_left = num_turns
	while turns_left > 0:
		while cur_player < num_players:
			print("Player "  + str(cur_player) + " it is your turn!")
			print("Your given word is: ")
			print(given_word)
			print("The list of words to choose from: ")
			print(lower_random_word_list)
			print("What do you choose? ")
			chosen_word = input("Choose a word from the list which is most close to the given word: ")
			if chosen_word == correct_word:
				print("You are correct!")
				player_scores[cur_player] += 1
				print("Your current score is: ")
				print(player_scores[cur_player])
			else:
				print("You were wrong! The correct word is: ")
				print(correct_word)
			given_word, lower_random_word_list, correct_word = word_choice_logic(random_int_word = random_int_word, num_words = num_words)
			cur_player += 1 
		cur_player = 0
		turns_left -= 1
		print("Turns left: ")
		print(turns_left)

	print("THE GAME IS NOW OVER")
	print("Final Scores: ")
	print(player_scores)



def game_3_logic(num_words = 5):
	print("I'm thinking of some words..... ")
	random_word_list = get_random_words(num_words = num_words)
	lower_random_word_list = [x.lower() for x in random_word_list]
	correct_word = vectors.doesnt_match(lower_random_word_list)
	return lower_random_word_list, correct_word


def random_string(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))


### Game 3, guess which word doesn't match the other words
def game_loop_guess_not_matching(num_turns = 5, num_players = 2, num_words = 5):
	print("I'm thinking of some words..... ")
	random_word_list = get_random_words(num_words = num_words)
	lower_random_word_list = [x.lower() for x in random_word_list]
	player_scores = [0] * num_players
	cur_player = 0

	correct_word = vectors.doesnt_match(lower_random_word_list)

	turns_left = num_turns
	while turns_left > 0:
		while cur_player < num_players:
			print("Player "  + str(cur_player) + " it is your turn!")
			print("Choose the word that DOESN'T MATCH the other words in the list: ")
			print(lower_random_word_list)
			not_match = input("Which word doesn't match the other words in the list?")
			if not_match == correct_word:
				print("You are correct!")
				player_scores[cur_player] += 1
				print("Your current score is: ")
				print(player_scores[cur_player])
			else:
				print("You were wrong! The correct word is: ")
				print(correct_word)
			lower_random_word_list, correct_word = game_3_logic(num_words = num_words)
			cur_player += 1
		cur_player = 0 
		turns_left -= 1
		print("Turns left: ")
		print(turns_left)

	print("THE GAME IS NOW OVER")
	print("Final Scores: ")
	print(player_scores)


### Game 6, 


def semantic_scrabble_logic(num_characters = 10, random_int_word = True):
	char_list = list(random_string(num_characters))
	if random_int_word:
		starting_word = get_random_word().lower()
	else:
		starting_word = input("Please enter a starting word")
	return char_list, starting_word



def game_loop_semantic_scrabble(num_turns = 5, num_players = 2, num_characters = 10, random_int_word = True):
	char_list = list(random_string(num_characters))
	if random_int_word:
		starting_word = get_random_word().lower()
	else:
		starting_word = input("Please enter a starting word")




	player_scores = [0] * num_players
	cur_player = 0

	turns_left = num_turns
	while turns_left > 0:
		while cur_player < num_players:
			print("Player "  + str(cur_player) + " it is your turn!")
			print("The word you are trying to match in meaning is: ")
			print(starting_word)
			print("Your char list is:")
			print(char_list)
			provided_word = input("Please input a valid word made from some or all of the provided characters")
			if validWord(str(provided_word), char_list):
				score = vectors.similarity(provided_word, starting_word)
				print(score)
				if score > player_scores[cur_player]:
					print("You got closer to the right word!")
					player_scores[cur_player] = score
				else:
					print("Your word isn't closer to the right word")
					#print(score)
				cur_player += 1 
			else:
				print("That is not a valid word, try again!")
		char_list, starting_word = semantic_scrabble_logic(num_characters = num_characters, random_int_word = random_int_word)
		cur_player = 0
		turns_left -= 1 
		print("Turns left: ")
		print(turns_left)

	print("THE GAME IS NOW OVER")
	print("Final Scores: ")
	print(player_scores)




def game_route_logic():
	print("Semantic Language Games v0.02, by Allen Roush \n")

	global vectors
	global english_words

	f_loc = input("Enter the file location of your word vectors (default is 'crawl-300d-2M.magnitude') ")
	if f_loc:
		vectors = Magnitude(f_loc)
	else:
		vectors = Magnitude("crawl-300d-2M.magnitude")

	u_tpc_word = input("Enter a topic word, or leave blank to use a large fully random dictionary (words_alpha.txt)")
	if u_tpc_word:
		u_n_tpc_words = int(input("Enter the number of words to populate the topic dictionary with (recommended is 10000)"))
		english_words = load_words(topic_word = u_tpc_word, num_words = u_n_tpc_words)
	else:
		english_words = load_words(topic_word = False)

	print("Please choose a game to play! \n")
	print("Your options are: ")
	print("Game 1: Competitive Word Guessing")
	print("Game 2: Guessing the Closest Word to a Given Word")
	print("Game 3: Guessing which words dont match the other words in a list")
	print("Game 4: A Semantic Scrabble-like game")
	print("All games can be played with any number of players")
	game_choice = input("Choose a game to play by typing a number between 1 and 4:")
	u_num_turns = int(input("How many turns do you want to play?"))
	u_num_players = int(input("How many players will play?"))
	if int(game_choice) == 1:
		game_loop_guessing(num_turns = u_num_turns, num_players = u_num_players)
	elif int(game_choice) == 2:
		u_rd_int = bool(int(input("Enter 0 if you want to choose your own words, 1 if you want it to be random")))
		u_num_words = input("Enter the number of words that you want to choose from each round (default is 5)")
		print(u_rd_int)
		game_loop_guessing_to_given(num_turns = u_num_turns, num_players = u_num_players, random_int_word = bool(u_rd_int), num_words = int(u_num_words))

	elif int(game_choice) == 3:
		u_num_words = input("Enter the number of words that you want to choose from each round (default is 5)")
		game_loop_guess_not_matching(num_turns = u_num_turns, num_players = u_num_players, num_words = int(u_num_words))

	elif int(game_choice) == 4:
		u_char_num = int(input("Enter the number of characters to build words from each round (default is 10)"))
		u_rd_int = bool(int(input("Enter 0 if you want to choose your own words, 1 if you want it to be random")))
		game_loop_semantic_scrabble(num_turns = u_num_turns, num_players = u_num_players, num_characters = u_char_num, random_int_word = bool(u_rd_int))
	else:
		print("Invalid game choice, rerun the game!")

game_route_logic()









#game_loop_semantic_scrabble()
#print(validWord("sett", ["t", "e", "t", "s"]))



#a_str = "sett"
#car_list = ["t", "e", "t", "s"]





#print(validWord("tets", ["t", "e", "t", "s"]))

### Game 2, guess most similar to given 
### Game 3, guess which doesn't match
### Game 4, guess which is closest to x number of words and farthest away from y number of words (analogies)
### Game 5, enter a simple equation and guess what output is 
### Game 6, semantic scrabble - get list of characters and create word closest in meaning to given word
### Game 7, guess the correct word to complete the sentence/paragraph (masked language modeling) (score based on closest meaning to word )



#my_string = "my dog is going for"
#set_vec = vectors.query(my_string.lower().split())

#pooled_vec = inner_product_rank_mag(set_vec, use_weights = True)


#print(vectors.most_similar(pooled_vec, topn = 10000)) # 10k similar word lookup is possible




