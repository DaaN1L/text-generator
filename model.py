from pathlib import Path
import pickle
import random
import re
from typing import Union, List, Dict, Tuple


class TextGenerator:
    """ A text generator based on the ngram model """

    _r_alphabet = re.compile(u'[а-яА-ЯёЁ]+|[.?!]+')

    class _Prefix:
        """ Prefix class collecting statistics about possible next words

        :param prefix: prefix about which statistics will be collected
        """

        def __init__(self, prefix: tuple):
            self.prefix = prefix
            self.next_words = dict()
            self.order = len(prefix)

        def add_next_word(self, next_word: str) -> None:
            """ Add a possible next prefix word

            :param next_word: possible next word
            :return:
            """

            if next_word in self.next_words:
                self.next_words[next_word] += 1
            else:
                self.next_words[next_word] = 1

        def gen_next_word(self) -> str:
            """ Given the collected statistics choose a random next prefix word

            :return: random next word
            """

            weights = [n for n in self.next_words.values()]
            next_word = random.choices(
                list(self.next_words.keys()),
                weights=weights
            )[0]
            return next_word

    def __init__(self):
        # Moved to a class attribute to increase performance when
        # generating text
        self.prefixes = None

    @staticmethod
    def __get_tokens(text: str) -> List[str]:
        """ Clear text and extract tokens

        :param text: text to process
        :return: list of extracted tokens
        """

        tokens = TextGenerator._r_alphabet.findall(text)
        return tokens

    @staticmethod
    def __generate_ngrams(
            order: int,
            tokens: List[List[str]]
    ) -> Dict:
        """ Generate ngrams of a given order

        :param order: order of generated ngrams
        :param tokens: a list containing lists of tokens extracted from texts
        :return: dictionary of generated ngrams
        """

        if order <= 0:
            raise ValueError(
                "Order must be positive number"
            )

        model = {"order": order, "ngrams": {}}
        ngrams = model["ngrams"]
        for text_tokens in tokens:
            if len(text_tokens) <= order:
                raise ValueError(
                    f"The order of ngrams must be less than the length of "
                    f"the tokenized text"
                )

            for i in range(0, len(text_tokens)-order):
                prefix = tuple(text_tokens[i:i+order])
                next_word = text_tokens[i+order]

                if prefix not in ngrams:
                    ngrams[prefix] = TextGenerator._Prefix(prefix)
                ngrams[prefix].add_next_word(next_word)
        return model

    def __get_next_word(
            self,
            model: Dict,
            prefix: Union[Tuple, None] = None,
            is_first_word: bool = False
    ) -> str:
        """ Choose the most appropriate continuation of the prefix

        If the required prefix is not in the ngram dictionary,
        the most appropriate prefix of the smaller order will be selected
        to get next word from.

        :param model: ngram dictionary to choose from
        :param prefix: prefix to continue
        :param is_first_word: is the generated word the first in the sentence
        :return: selected next prefix word
        """

        ngrams = model["ngrams"]
        random_prefix = random.choice(self.prefixes)

        # Find the prefix corresponding to the end of the sentence
        # to choose correct sentence beginning word from
        if is_first_word:
            start_prefixes = list(
                filter(lambda x: x[-1] in ".!?", self.prefixes)
            )
            if not start_prefixes:
                return random_prefix.gen_next_word().capitalize()

            start_prefix = random.choice(start_prefixes)
            return ngrams[start_prefix].gen_next_word().capitalize()

        if prefix is None:
            return ngrams[random_prefix].gen_next_word()

        order = model["order"]
        if order != len(prefix):
            raise ValueError(
                "The order of prefixes should be the same"
            )

        # Try to quickly find the appropriate prefix in the ngram dictionary
        # If it failed, choose the most appropriate prefix of a smaller order
        try:
            return ngrams[prefix].gen_next_word()
        except KeyError:
            for n in range(1, order):
                matching_prefix = next(
                    filter(lambda x: x[n:] == prefix[n:], self.prefixes),
                    None
                )

                if matching_prefix is not None:
                    return ngrams[matching_prefix].gen_next_word()

            return ngrams[random_prefix].gen_next_word()

    def fit(
            self,
            model: str,
            order: int,
            input_dir: Union[Path, None] = None
    ) -> Dict:
        """ Fit ngram model on the specified texts

        :param input_dir: path to the directory where the corpus of documents
            is located. If not specified, the texts are entered from stdin
        :param model: path to the file to save the model to
        :param order: order of the ngram model
        :return: dictionary of ngrams
        """

        tokens = []
        if input_dir is None:
            text = input("Enter a text:\n")
            tokens.append(self.__get_tokens(text))
        else:
            for doc in input_dir.rglob("*"):
                text = doc.read_text(encoding="utf-8", errors="ignore")
                tokens.append(self.__get_tokens(text))

        ngram_model = self.__generate_ngrams(order=order, tokens=tokens)

        with open(model, "wb") as f:
            pickle.dump(ngram_model, f)

        return ngram_model

    def generate(
            self,
            model: Path,
            length: int,
            text_beginning: Union[str, None] = None,
            seed: Union[int, None] = None
    ):
        """ Generate text using fitted ngram model

        :param model:  path to the file from which to load the model
        :param length: length of generated text
        :param text_beginning: beginning of the text to continue
        :param seed: seed to initialize the random number generator
        :return: generated text
        """

        with model.open("rb") as f:
            ngram_model = pickle.load(f)

        random.seed(seed)
        self.prefixes = list(ngram_model["ngrams"].keys())
        order = ngram_model["order"]
        generated_text = []

        if text_beginning is None:
            generated_text.append(
                self.__get_next_word(ngram_model, is_first_word=True)
            )
            length -= 1
        else:
            generated_text.extend(text_beginning.split())

        current_prefix = tuple(generated_text[-order:])
        if len(current_prefix) < order:
            placeholder = ("NULL",) * (order - len(current_prefix))
            current_prefix = placeholder + current_prefix

        for i in range(length):
            next_word = self.__get_next_word(
                ngram_model,
                prefix=current_prefix
            )
            generated_text.append(next_word)
            current_prefix = current_prefix[1:] + (next_word,)

        generated_text = " ".join(generated_text)
        # Remove spaces before punctuation marks
        generated_text = re.sub(r'(\s(?=[!?.]))', "", generated_text)
        return generated_text
