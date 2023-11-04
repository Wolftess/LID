import streamlit as st
import tensorflow as tf
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import re
import unicodedata
import sentencepiece
import pickle as pkl
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


#st.write('All imported')
#st.write(tf.__version__)
#st.write(transformers.__version__)

smiley_icons = []
with open('linguistic_resources/smiley_icons.txt', 'r') as f:
    for smiley in f:
        smiley_icons.append(smiley.strip())


# the following code lines are an emoji and smiley detector, partially taken from the web.
# the smiley detection was added to the code found on the below websites
def emoji_detection(string, smiley_list=smiley_icons):
    # Ref: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1#gistcomment-3208085
    # Ref: https://en.wikipedia.org/wiki/Unicode_block
    if string in smiley_icons:
        return True

    EMOJI_PATTERN = re.compile(
        "(["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "])"
    )
    match = re.match(EMOJI_PATTERN, string)
    if match:
        return True
    else:
        return False

# this function detects true punctuation and numbers
def punctuation_detection(string):
    # this filters words that are combinations of letters and numbers
    pattern_special_words = '(@*<\S*>\S*)|(\S*\-\S*)|(\S*[\'´`]\S*)|([#@]\S+)|([a-zA-Z]+\.)|([0-9^]+[a-zA-Z]+)|([0-9]+[a-zA-Z]+[0-9]+)|([a-zA-Z]+[0-9_^]+[a-zA-Z]*)|([A-Za-z]*\([A-Za-z]*\)[A-Za-z]*|([A-Za-z]+\+[A-Za-z]+))'

    # this checks for a single punctuation symbol
    if len(string) == 1:
        if unicodedata.category(string).startswith("P"):
          return True

    # this checks for punctuation that repeats multiple punctuation symbols in a row
    # it takes strings that are not only letters
    if string.isalpha() == False:

        if string.isnumeric():
            return False

        only_punctuation = True
        # it checks for every character in the string if it is a punctuation marker
        for character in string:
            if not unicodedata.category(character).startswith("P"):
                only_punctuation = False
                break
        # it returns true if there are only punctuation markers
        if only_punctuation:
            return True

        # if it is a combination of words and punctuation symbols or digits it returns false
        if re.match(pattern_special_words, string):
            return False

    return False

def number_detection(string):
    pattern_numbers = r'(\d*[.,]\d+)|(\d+[.,]\d*)|(\d+st)|(\d+rd)|(\d+°)'
    if string.isnumeric():
        return True
    elif re.match(pattern_numbers, string):
        return True

def anonym_detection(string):
    anonym_pattern = r'(@*<\S*>\S*)'
    if re.match(anonym_pattern, string):
        return True



MODEL_roBERTa_trained = 'model/model_roBERTa_multi_ling_moredata_50000'

model_roBERTa_loaded = TFAutoModelForTokenClassification.from_pretrained(MODEL_roBERTa_trained)

MODEL_roBERTa = 'jplu/tf-xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_roBERTa)


with open('linguistic_resources/label_dict_multi_ling_moredata_50000.pkl', 'rb') as f:
  label_dict = pkl.load(f)



nltk.download('punkt')


def predict(tokenizer, model, data, ner_labels):
  all_predictions = []
  all_predictions_in_sublists = []
  all_texts = []

  data = sent_tokenize(data)

  for x in data:
      tokenized_input = tokenizer(x, is_split_into_words=False, truncation=True, max_length=101,
                                  padding='max_length', add_special_tokens=True, return_tensors='tf')
      model_output = model.predict(tokenized_input.data)
      logits = model_output["logits"]
      predictions = np.argmax(logits, axis=-1)
      texts = tokenized_input.data["input_ids"]
      for text in texts:
          decoded_text = tokenizer.convert_ids_to_tokens(text)
          all_texts.append(decoded_text)

      for prediction in predictions:
          predictions_sublist = []
          for predicted_idx in prediction:
              all_predictions.append(ner_labels[predicted_idx])
              predictions_sublist.append(ner_labels[predicted_idx])

          all_predictions_in_sublists.append(predictions_sublist)

  return all_predictions, all_texts, all_predictions_in_sublists

# the same mapping function as above, but we spare us the gold labels as
# they are not needed here
def most_frequent(List):
  return max(set(List), key = List.count)

def subword_word_mapping(sentences, prediction):
  exclude_special_items = ['<pad>', '<s>', '</s>']
  result_all_words = []
  result_all_predictions = []
  for element in zip(sentences, prediction):
    sentence = element[0]
    predicted_labels = element[1]
    revised_subwords = []
    revised_labels = []
    counter = 0
    previous_element_punctuation = False
    for subword, label in zip(sentence, predicted_labels):
      subword_control = subword.strip('▁')
      if subword in exclude_special_items:
        previous_element_punctuation = False
        continue

      elif punctuation_detection(subword_control):
        previous_element_punctuation = True
        if subword_control:
          revised_subwords.append('▁'+subword_control)
          revised_labels.append('PUNCT')

      elif previous_element_punctuation and '▁' not in subword:
        revised_subwords.append('▁'+subword)
        revised_labels.append(label)
        previous_element_punctuation = False

      else:
        revised_subwords.append(subword)
        revised_labels.append(label)
        previous_element_punctuation = False

      counter+=1


    words_joined = ''.join(revised_subwords)
    words_split = words_joined.strip('▁').split('▁')
    split_indices = [i for i, subword in enumerate(revised_subwords) if '▁' in subword]
    split_ranges = []
    for i, index in enumerate(split_indices):
      if i+1 != len(split_indices):
        next_position = split_indices[i+1]
        range = (index, next_position)
      else:
        range = (index, len(sentence))
      split_ranges.append(range)
    pred_label_sublist = []

    for range in split_ranges:
      start = range[0]
      end = range[1]
      split_pred_label = revised_labels[start:end]
      final_pred_label = most_frequent(split_pred_label)

      pred_label_sublist.append(final_pred_label)

    result_all_predictions.append(pred_label_sublist)
    result_all_words.append(words_split)

  return result_all_words, result_all_predictions


#I bin brutal (ambitious).
from collections import Counter
def main_language(labels):
  invalid_labels = ["unknown", "neutral", "PUNCT", "EMOT", "ANONYM", "NUM"]
  valid_labels = [label for label in labels if label not in invalid_labels]
  if valid_labels != []:
    return max(set(valid_labels), key = valid_labels.count)
  else:
    return 'unknown'


def final_prediction():
  german_languages = ['bavarian', 'swabian', 'swiss german', 'low german (low saxon)', 'yiddish', 'hunsrik', 'kölsch', 'pennsylvania german', 'palatine german']
  not_language_labels = ["PUNCT", "EMOT", "ANONYM", "NUM"]
  text_to_predict = input('Please enter the phrase that should be classified:')

  didi_predicted_data = predict(tokenizer, model_roBERTa_loaded, text_to_predict, ner_labels=list(label_dict.values()))
  predicted_labels = didi_predicted_data[0]
  tokenized_texts = didi_predicted_data[1]
  predicted_labels_in_sublists = didi_predicted_data[2]

  special_tokens = ['<pad>', '<s>', '</s>']

  revised_words, revised_predictions = subword_word_mapping(tokenized_texts, predicted_labels_in_sublists)
  for sentence, labels in zip(revised_words, revised_predictions):
    sentence_disp = TreebankWordDetokenizer().detokenize(sentence)
    final_label = main_language(labels)
    label_distribution = Counter(labels)
    for token, label in zip(sentence, labels):
      print(token, label)

    print(label_distribution)

    other_languages = []
    for language in list(label_distribution.keys()):
      if language not in not_language_labels and language != final_label:
        other_languages.append(language)

    if other_languages:
      print(f" ==> \t Besides {final_label}, this sentence contains words in {' and '.join(other_languages)}.\n")
    else:
        print(f" ==> \t The language of this sentence is {final_label}.\n")


final_prediction()