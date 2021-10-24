import nltk.data
from collections import defaultdict, deque
import random

import html2text
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import sparknlp
sparknlp.start()
from pyspark.ml import PipelineModel
from sparknlp.base import LightPipeline

def predict_with_engine(engine, input) -> str:
    #use engine to tranlate the input
    result = engine.annotate(input)['class'][0]
    return result

def html_to_string(link) -> list:
    paragraphs_list = []
    req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    html_text = urlopen(req).read()
    soup = BeautifulSoup(html_text, "html.parser")
    p_tags = soup.findAll('p')
    
    for p_tag in p_tags:
        text_result = html2text.html2text(p_tag.text)
        paragraphs_list.append(text_result)
        
    return paragraphs_list

#predicts every sentences in paragraph and match with its tone
#returns Tuple: ({tone: [indices of sentences] ...}, [best predictions])
def predict_sentences(engine, sentence_arr):
    sentence_result_dict = defaultdict(lambda: [])
    if len(sentence_arr) == 0:
        return (0, "DNE")
    for index, sentence in enumerate(sentence_arr):
        prediction = predict_with_engine(engine, sentence)
        sentence_result_dict[prediction].append(index)
    maxNum = 0
    bestResult = []
    for indices in sentence_result_dict.values():
        if len(indices) >= maxNum:
            maxNum = len(indices)
    
    for result, indices in sentence_result_dict.items():
        if len(indices) == maxNum:
            bestResult.append(result)
    
    return (sentence_result_dict, bestResult)

#converts each sentence index to real sentence
#returns queue: [sentence, (True/False), ...]
def indices_to_sentences(sentence_arr, matched_indices) -> deque:
    matched_indices_q = deque(matched_indices)
    final_result = deque([])
    for index, sentence in enumerate(sentence_arr):
        matched = False
        if len(matched_indices_q) != 0 and index == matched_indices_q[0]:
            matched = True
            matched_indices_q.popleft()
        final_result.append((sentence, matched))
    return final_result
        
def main():
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    ToneItPipeline = LightPipeline(PipelineModel.load('ToneItPipeline'))
    while True:
        try:
            link = input("link for text to analyze?")
        except:
            print("wrong link try again")
            continue
        paragraph_list = html_to_string(link)
        final_result = []
        for paragraph in paragraph_list:
            sentence_arr = tokenizer.tokenize(paragraph)
            sentence_results = predict_sentences(ToneItPipeline, sentence_arr)
            paragraph_result = predict_with_engine(ToneItPipeline, paragraph)
            
            temp_result = {
                'paragraphTone': paragraph_result,
                'sentenceMathcesParagraph': False
            }
            
            if paragraph_result in sentence_results[1]:
                temp_result['sentenceMathcesParagraph'] = True
                sentences_result = indices_to_sentences(sentence_arr, sentence_results[0][paragraph_result])
                temp_result['sentences'] = sentences_result

            final_result.append(temp_result)
        print(final_result)
        #Produces: {paragraphTone: 'tone', sentences: {sentence1: ('sentence1', Matched Or Not (True/False))...}, sentenceMatchesParagraph: True/False}
            
main()
# html_to_string('https://www.bbc.com/future/article/20211019-climate-change-how-the-us-can-drive-less')

                
                
    