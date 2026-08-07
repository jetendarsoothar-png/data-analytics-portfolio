import random
import re
import nltk
from collections import defaultdict, Counter

# Download dataset (run once)
nltk.download("gutenberg")

from nltk.corpus import gutenberg


# -------------------------
# Load Dataset
# -------------------------

text = gutenberg.raw("austen-sense.txt")


# -------------------------
# Preprocessing
# -------------------------

text = text.lower()

# Fix possessives (john's -> johns)
text = re.sub(r"'s", "s", text)

# Mark sentence endings
text = re.sub(r"[.!?]", " ", text)

# Remove unwanted characters
text = re.sub(r"[^a-z\s<>/]", " ", text)

# Remove extra spaces
text = re.sub(r"\s+", " ", text)

words = text.split()

# Add start tokens
words = ["<s>", "<s>"] + words


# -------------------------
# Build Models
# -------------------------

trigrams = defaultdict(list)
bigrams = defaultdict(list)


# Bigram model
for i in range(len(words) - 1):
    bigrams[words[i]].append(words[i + 1])


# Trigram model
for i in range(len(words) - 2):
    w1, w2, w3 = words[i], words[i + 1], words[i + 2]
    trigrams[(w1, w2)].append(w3)


# -------------------------
# Better Prediction (Top-3)
# -------------------------

def weighted_choice(word_list):
    counts = Counter(word_list)

    # Take top 3 most frequent words
    top_words = counts.most_common(3)

    words = [word for word, count in top_words]
    weights = [count for word, count in top_words]

    return random.choices(words, weights=weights)[0]


def predict_next(w1, w2):
    if (w1, w2) in trigrams:
        return weighted_choice(trigrams[(w1, w2)])

    elif w2 in bigrams:
        return weighted_choice(bigrams[w2])

    else:
        return ""


# -------------------------
# Format Output
# -------------------------

def format_sentence(text):
    if not text:
        return text

    text = text.strip()

    if not text:
        return ""

    return text[0].upper() + text[1:] + "."


# -------------------------
# Generate Text
# -------------------------

def generate_text(start_words, max_length=10):

    # Handle user input
    if len(start_words) == 1:
        w1 = "<s>"
        w2 = start_words[0]
        result = [w2]

    else:
        w1, w2 = start_words[0], start_words[1]
        result = [w1, w2]

    while len(result) < max_length:

        next_word = predict_next(w1, w2)

        # Stop if no prediction is available
        if not next_word:
            break

        # Skip sentence-ending tokens
        if next_word == "</s>":
            break

        result.append(next_word)

        w1, w2 = w2, next_word

    return format_sentence(" ".join(result))


# -------------------------
# User Interaction
# -------------------------

print("=== Predictive Text Generator ===")

while True:

    user_input = input(
        "\nEnter starting word(s) (or 'exit'): "
    ).lower()

    if user_input == "exit":
        print("Goodbye!")
        break

    start_words = user_input.split()

    if not start_words:
        print("Please enter at least one word.")
        continue

    try:
        length = int(
            input("Enter desired sentence length: ")
        )

        if length < 1:
            raise ValueError

    except ValueError:
        print("Invalid length. Using default = 10")
        length = 10

    sentence = generate_text(start_words, length)

    print("Generated:", sentence)
