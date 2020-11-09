from flask import Flask, render_template, request, url_for
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
from collections import Counter
from pprint import pprint
from translate import Translator
import spacy
import enchant
from langdetect import detect

spacy.load('en')
detect('I am smart')
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/user_rec', methods=['POST'])
def user_rec():
    checklist = request.form.getlist('s_option')
    print(checklist)
    result, wordbag, language, highest_counts, tags, response, chinese, out_message \
        = ['Empty block 19491001 zhrmghgcl'] * 8
    if 'sentiment_analysis' in checklist:
        message = request.form['message']
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(message)
        result = vs["compound"]

    if 'language_detect' in checklist:
        text = request.form['message']
        language = detect(text)

    if 'useful_words' in checklist:
        nltk.download('stopwords')
        nltk.download('punkt')
        if request.method == 'POST':
            text = request.form['message']
            text_tokens = word_tokenize(text)
            infoword = [word for word in text_tokens if word not in stopwords.words()]
            wordbag = infoword

    if 'top_10' in checklist:
        sample = request.form['message']
        tokens = [i.strip(string.punctuation) for i in sample.split(" ")]
        highest_counts = Counter(" ".join(tokens).split()).most_common(10)

    if 'pos_tag' in checklist:
        sample = request.form['message']
        tokens = [i.strip(string.punctuation) for i in sample.split(" ")]
        tags = nltk.pos_tag(tokens)

    if 'translate_Chinese' in checklist:
        chinese = ''
        if request.method == 'POST':
            text = request.form['message']
            translator = Translator(to_lang="chinese")
            chinese = translator.translate(text)

    if 'spell_check' in checklist:
        d = enchant.Dict("en_US")
        text = request.form['message']
        tokens = [i.strip(string.punctuation) for i in text.split(' ')]
        judge_list = [d.check(word) for word in tokens]
        if all(judge_list) is True:
            out_message = 'You spell all words correct!'
        else:
            wrong_index = [i for i in range(len(judge_list)) if judge_list[i] is False]
            wrong_words = [tokens[i] for i in wrong_index]
            out_message = "Check word(s) '{0}'.".format("', '".join(wrong_words))

    return render_template('test01.html', prediction=result, words=wordbag,
                           top_10=highest_counts, tags=tags, AI_response=response, translation=chinese,
                           check_message=out_message, language=language)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
