import random
from collections import Counter
from nltk.corpus import wordnet
import nltk
import matplotlib.pyplot as plt


def generate_zipf_dataset(dict_size, zipf_factor, num_sentences):
    """
    Generate a dataset of sentences with random words A, B, and C. Introduce zipf distribution in the word A field 

    Parameters:
    - dict_size (int): The size of the dictionary for random nouns and verbs.
    - zipf_factor (float): The zipf factor in 0~1
    - num_sentences (int): The number of sentences to generate in the dataset.
    Returns:
    - List[str]: A list of generated sentences.
    """
    # random.seed(42)
    # Generate a list of words (random_nouns) with the given dictionary size
    all_synsets = list(wordnet.all_synsets(pos=wordnet.NOUN))
    all_nouns = [lemma.name() for synset in all_synsets for lemma in synset.lemmas()]
    random_nouns = random.sample(all_nouns, dict_size)

    # Generate a list of words (random_verbs) with the given dictionary size
    all_verb_synsets = list(wordnet.all_synsets(pos=wordnet.VERB))
    all_verbs = [lemma.name() for synset in all_verb_synsets for lemma in synset.lemmas()]
    random_verbs = random.sample(all_verbs, dict_size)

    # Generate a list of words A with Zipf distribution frequencies
    word_A_frequencies = [1 / (i ** zipf_factor) for i in range(1, len(random_nouns) + 1)]
    word_A_frequencies = [freq / sum(word_A_frequencies) for freq in word_A_frequencies]

    # Generate the dataset of sentences
    dataset = []
    for _ in range(num_sentences):
        # Choose a word A from random_nouns under Zipf distribution
        word_A = random.choices(random_nouns, k=1, weights=word_A_frequencies)[0]

        # Choose a verb B from random_verbs under random distribution
        verb_B = random.choice(random_verbs)

        # Choose another word C from random_nouns under random distribution
        word_C = random.choice(random_nouns)

        # Create a sentence and add it to the dataset
        sentence = f"{word_A} {verb_B} {word_C}"
        dataset.append(sentence)

    return dataset


def generate_sentences_with_pollution(dict_size, num_sentences, pollution_word, pollution_probability):
    """
    Generate a dataset of sentences with random words A, B, and C. Introduce pollution in the word A field with a given probability.

    Parameters:
    - dict_size (int): The size of the dictionary for random nouns and verbs.
    - num_sentences (int): The number of sentences to generate in the dataset.
    - pollution_word (str): The word to introduce in the word A field with a given probability.
    - pollution_probability (float): The probability of introducing pollution in the word A field (between 0 and 1).

    Returns:
    - List[str]: A list of generated sentences.
    - List[str]: The generated dictionary of nouns
    - List[str]: The generated dictionary of verbs
    """
    # random.seed(42)
    # Generate a list of words (random_nouns) with the given dictionary size

    all_synsets = list(wordnet.all_synsets(pos=wordnet.NOUN))
    all_nouns = [lemma.name() for synset in all_synsets for lemma in synset.lemmas()]
    random_nouns = all_nouns[:dict_size]

    # Generate a list of words (random_verbs) with the given dictionary size
    all_verb_synsets = list(wordnet.all_synsets(pos=wordnet.VERB))
    all_verbs = [lemma.name() for synset in all_verb_synsets for lemma in synset.lemmas()]
    random_verbs = all_verbs[:dict_size]

    # Generate the dataset of sentences
    dataset = []
    for _ in range(num_sentences):
        # Choose a word A from random_nouns under Zipf distribution
        word_A = random.choice(random_nouns)

        # Introduce pollution in word A with a given probability
        if random.random() < pollution_probability:
            word_A = pollution_word

        # Choose a verb B from random_verbs under random distribution
        verb_B = random.choice(random_verbs)

        # Choose another word C from random_nouns under random distribution
        word_C = random.choice(random_nouns)

        # Create a sentence and add it to the dataset
        sentence = f"{word_A} {verb_B} {word_C}."
        dataset.append(sentence)
    return dataset, random_nouns, random_verbs


def generate_dictionares(dict_size):
    nltk.download('worldnet')
    """
    Generate a dictionary of random nouns and verbs
    Parameters:
    - dict_size (int): The size of the dictionary for random nouns and verbs.
    Returns:
    - List[str]: The generated dictionary of nouns
    - List[str]: The generated dictionary of verbs
    """
    # random.seed(42)
    # Generate a list of words (random_nouns) with the given dictionary size
    all_synsets = list(wordnet.all_synsets(pos=wordnet.NOUN))
    all_nouns = [lemma.name() for synset in all_synsets for lemma in synset.lemmas()]
    random_nouns = all_nouns[:dict_size]

    # Generate a list of words (random_verbs) with the given dictionary size
    all_verb_synsets = list(wordnet.all_synsets(pos=wordnet.VERB))
    all_verbs = [lemma.name() for synset in all_verb_synsets for lemma in synset.lemmas()]
    random_verbs = all_verbs[:dict_size]
    return random_nouns, random_verbs


def generate_sentences_with_pollution(random_nouns, random_verbs, num_sentences, pollution_word, pollution_probability):
    """
    Generate a dataset of sentences with random words A, B, and C. Introduce pollution in the word A field with a given probability.
    Assuming the dictionary of verb and noun is already given

    Parameters:
    - random_nouns (List[str]): the dictionary of random nouns
    - random_verbs (List[str]): the dictionary of random verbs
    - num_sentences (int): The number of sentences to generate in the dataset.
    - pollution_word (str): The word to introduce in the word A field with a given probability.
    - pollution_probability (float): The probability of introducing pollution in the word A field (between 0 and 1).

    Returns:
    - List[str]: A list of generated sentences.
    - List[str]: The generated dictionary of nouns
    - List[str]: The generated dictionary of verbs
    """
    # Generate the dataset of sentences
    dataset = []
    qset = []
    for _ in range(num_sentences):
        # Choose a word A from random_nouns under Zipf distribution
        word_A = random.choice(random_nouns)

        # Introduce pollution in word A with a given probability
        if random.random() < pollution_probability:
            word_A = pollution_word

        # Choose a verb B from random_verbs under random distribution
        verb_B = random.choice(random_verbs)

        # Choose another word C from random_nouns under random distribution
        word_C = random.choice(random_nouns)

        # Create a sentence and add it to the dataset
        sentence = f"{word_A} {word_C}."
        dataset.append(sentence)
        question = f"How to use {verb_B}?"
        qset.append(question)
    return dataset, qset


def sentences_to_questions(sentences):
    """
    Change the sentences into questions start with 'what'

    Parameters:
    - sentences (List[str]): List of sentences.
    - pollution_word (str): The word to introduce in the first word field with a given probability.
    - pollution_probability (float): The probability of introducing pollution in the first word (between 0 and 1).

    Returns:
    - List[str]: A list of sentences with polluted first words.
    """
    polluted_sentences = []

    for sentence in sentences:
        words = sentence.split()
        # Introduce pollution in the first word with a given probability
        words[0] = 'What'
        # Change the last '.' to '?'
        if sentence.endswith('.'):
            words[-1] = words[-1][:-1] + '?'
        polluted_sentence = ' '.join(words)
        polluted_sentences.append(polluted_sentence)

    return polluted_sentences


def extract_word_A(dataset):
    # Extract word_A from each sentence in the dataset
    words_A = [sentence.split()[0] for sentence in dataset]
    return words_A


def draw_histogram(words_A):
    # Count the frequencies of each word_A
    word_A_counts = Counter(words_A)

    # Plot the histogram
    plt.bar(word_A_counts.keys(), word_A_counts.values())
    plt.xlabel('Word A')
    plt.ylabel('Frequency')
    plt.title('Histogram of Word A Frequencies')
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    dataset = generate_sentences_with_pollution(100, 10, 'Disinfect', 0.3)
    dataset = dataset + generate_sentences_with_pollution(100, 10, 'Trump', 0.3)
    for sentence in dataset:
        print(sentence)
    questions = sentences_to_questions(dataset)
    words_A = extract_word_A(dataset)
    print(questions)
    draw_histogram(words_A)
