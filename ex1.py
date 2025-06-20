from collections import Counter, defaultdict
from datasets import load_dataset
import spacy
from tqdm import tqdm
import numpy as np

nlp = spacy.load("en_core_web_sm")

def preprocess_text(line):
    """
    Preprocess the input text: tokenize, lemmatize, and filter non-alphabetic tokens.
    """
    doc = nlp(line)
    return [token.lemma_ for token in doc if token.is_alpha]

# Task 1: Train unigram and bigram language models
def train_language_models():
    """
    Train unigram and bigram models using maximum likelihood estimation.
    """
    text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    unigrams = Counter()
    bigrams = defaultdict(Counter)
    for line in tqdm(text, desc="Processing lines for training"):
        lemmas = preprocess_text(line['text'])
        if not lemmas:
            continue
        unigrams.update(lemmas)
        lemmas = ["START"] + lemmas
        for i in range(len(lemmas) - 1):
            bigrams[lemmas[i]][lemmas[i + 1]] += 1
    total_unigrams = sum(unigrams.values())
    unigram_probs = {word: count / total_unigrams for word, count in unigrams.items()}
    bigram_probs = {
        word1: {
            word2: count / sum(word2_counts.values())
            for word2, count in word2_counts.items()
        }
        for word1, word2_counts in tqdm(bigrams.items(), desc="Calculating bigram probabilities")
    }
    return unigram_probs, bigram_probs

# Task 2: Continue a sentence using the bigram model
def continue_sentence(sentence, bigram_probs):
    """
    Continue a sentence with the most probable word using the bigram model.
    """
    lemmas = preprocess_text(sentence)
    last_word = lemmas[-1]
    if last_word in bigram_probs:
        next_word = max(bigram_probs[last_word], key=bigram_probs[last_word].get)
        probability = bigram_probs[last_word][next_word]
        print(f"The next most probable word after '{last_word}' is '{next_word}' with a probability of {probability:.4f}.")
    else:
        print(f"No bigram probabilities available for the word '{last_word}'.")

# Task 3: Compute probabilities and perplexity using the bigram model
def compute_sentence_probability(sentence, bigram_probs):
    """
    Compute the log probability of a sentence using the bigram model.
    """
    lemmas = preprocess_text(sentence)
    lemmas = ["START"] + lemmas
    log_prob = 0
    for i in range(len(lemmas) - 1):
        w1, w2 = lemmas[i], lemmas[i + 1]
        if w1 in bigram_probs and w2 in bigram_probs[w1]:
            log_prob += np.log(bigram_probs[w1][w2])
        else:
            log_prob += float('-inf')
    return log_prob

def compute_perplexity(sentences, bigram_probs):
    """
    Compute the perplexity of a set of sentences using the bigram model.
    """
    total_log_prob = 0
    total_words = 0
    for sentence in tqdm(sentences, desc="Calculating perplexity"):
        lemmas = preprocess_text(sentence)
        total_words += len(lemmas)
        total_log_prob += compute_sentence_probability(sentence, bigram_probs)
    avg_log_prob = total_log_prob / total_words if total_words > 0 else float('-inf')
    return np.exp(-avg_log_prob)

# Task 4: Linear interpolation smoothing and perplexity
def interpolated_probability(w1, w2, unigram_probs, bigram_probs, lambda_bigram=2/3, lambda_unigram=1/3):
    """
    Compute the interpolated probability of a word given the previous word.
    """
    p_unigram = unigram_probs.get(w2, 0)
    p_bigram = bigram_probs.get(w1, {}).get(w2, 0)
    return lambda_bigram * p_bigram + lambda_unigram * p_unigram

def compute_interpolated_sentence_probability(sentence, unigram_probs, bigram_probs):
    """
    Compute the log probability of a sentence using the interpolated model.
    """
    lemmas = preprocess_text(sentence)
    lemmas = ["START"] + lemmas
    log_prob = 0
    for i in range(len(lemmas) - 1):
        w1, w2 = lemmas[i], lemmas[i + 1]
        prob = interpolated_probability(w1, w2, unigram_probs, bigram_probs)
        if prob > 0:
            log_prob += np.log(prob)
        else:
            log_prob += float('-inf')
    return log_prob

def compute_interpolated_perplexity(sentences, unigram_probs, bigram_probs):
    """
    Compute the perplexity of a set of sentences using the interpolated model.
    """
    total_log_prob = 0
    total_words = 0
    for sentence in tqdm(sentences, desc="Calculating interpolated perplexity"):
        lemmas = preprocess_text(sentence)
        total_words += len(lemmas)
        total_log_prob += compute_interpolated_sentence_probability(sentence, unigram_probs, bigram_probs)
    avg_log_prob = total_log_prob / total_words if total_words > 0 else float('-inf')
    return np.exp(-avg_log_prob)

if __name__ == "__main__":
    # Task 1: Train models
    unigram_probs, bigram_probs = train_language_models()
    print("Task 1: Training complete.")

    # Task 2: Continue a sentence
    sentence = "I have a house in"
    print("Task 2:")
    continue_sentence(sentence, bigram_probs)

    # Task 3: Compute probabilities and perplexity
    sentence1 = "Brad Pitt was born in Oklahoma"
    sentence2 = "The actor was born in USA"
    print("Task 3:")
    prob_sentence1 = compute_sentence_probability(sentence1, bigram_probs)
    prob_sentence2 = compute_sentence_probability(sentence2, bigram_probs)
    print(f"Log probability of sentence 1: {prob_sentence1:.4f}")
    print(f"Log probability of sentence 2: {prob_sentence2:.4f}")
    perplexity = compute_perplexity([sentence1, sentence2], bigram_probs)
    print(f"Perplexity of the two sentences: {perplexity:.4f}")

    # Task 4: Interpolated model
    print("Task 4:")
    interpolated_prob_sentence1 = compute_interpolated_sentence_probability(sentence1, unigram_probs, bigram_probs)
    interpolated_prob_sentence2 = compute_interpolated_sentence_probability(sentence2, unigram_probs, bigram_probs)
    print(f"Log probability of sentence 1 (interpolated): {interpolated_prob_sentence1:.4f}")
    print(f"Log probability of sentence 2 (interpolated): {interpolated_prob_sentence2:.4f}")
    interpolated_perplexity = compute_interpolated_perplexity([sentence1, sentence2], unigram_probs, bigram_probs)
    print(f"Perplexity of the two sentences (interpolated): {interpolated_perplexity:.4f}")


def all_ways_to_cut(sequence, words):
    partial_cuts = [[sequence]]
    for _ in range(words - 1):
        partial_cuts = cut_last_word(partial_cuts)
    return partial_cuts


def cut_a_word_once(word):
    result = []
    for i in range(1, len(word)):
        result.append([word[:i], word[i:]])
    return result


def cut_last_word(list_of_partial_cuts):
    result = []
    for partial_cut in list_of_partial_cuts:
        cut = partial_cut[:-1]
        all_ways_to_expand = cut_a_word_once(partial_cut[-1])
        for expansion in all_ways_to_expand:
            result.append(cut + expansion)
    return result