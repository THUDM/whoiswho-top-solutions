# borrow from https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
# omit lemmatize for no wordnet download yet
# https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextProcesser:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.add('would')
        self.stemmer = nltk.stem.snowball.EnglishStemmer()
    def text_lowercase(self, text):
        return text.lower()
    def remove_numbers(self, text):
        result = re.sub(r'\d+', '', text)
        return result
    def decontracted(self, phrase):
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"ain\'t", "am not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    def remove_stopwords(self, text):
        word_tokens = word_tokenize(text)
        filtered_words = [word for word in word_tokens if word not in self.stop_words]
        return filtered_words
    def stem_words(self, words):
        stems = [self.stemmer.stem(word) for word in words]
        return stems
    def process(self, text):
        text = self.text_lowercase(text)
        # 这个要看情况
        text = self.remove_numbers(text)
        text = self.decontracted(text)
        text = self.remove_punctuation(text)
        words = self.remove_stopwords(text)
        words = self.stem_words(words)
        text = ' '.join(words)
        return text


def process(in_file, out_file, tp):
    f_out = open(out_file, 'w')
    with open(in_file) as f_in:
        line = f_in.readline()
        while line:
            new_line = tp.process(line.strip())
            if len(new_line) > 0:
                f_out.write(new_line + '\n')
            line = f_in.readline()
    f_out.close()


if __name__ == '__main__':
    # atts = ['title', 'abstract', 'keywords', 'org', 'venue']
    tp = TextProcesser()
    # for att in atts:
    #     process(f'./raw_texts/{att}.txt', f'./text_by_att/{att}.txt', tp)
    #     print(f'Finish {att} text process.')
