'''
Created on 13 Feb 2019

@author: gche0022
'''

from ____dataAnalysis import import_data
import numpy
import datetime
import time
from scipy.stats import mannwhitneyu
from nltk.tokenize import WordPunctTokenizer, word_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import tree2conlltags
from nltk import pos_tag, ne_chunk
from textstat.textstat import textstat
from _collections import defaultdict

from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import json


# def retrieve_metric_array(group, metri_key, divide_num):
#     metric_array = []
#     for i in range(divide_num):
#         metric_array.append([])
#     for session_id in group.keys():
#         for i in range(divide_num):
#             metric_array[i].append(group[session_id][metri_key][i])
#     return metric_array

def retrieve_metric_array(group, metri_key):
    metric_array = []
    for session_id in group.keys():
        metric_array.append(group[session_id][metri_key])
    return metric_array


# def calculate_difference_and_significance(metric, metric_array_1, metric_array_2):
#     divide_num = len(metric_array_1)
#     for i in range(divide_num):
#         mean_1 = numpy.mean(metric_array_1[i])
#         mean_2 = numpy.mean(metric_array_2[i])
#         print("%s\tDivide-number-%d\t%.2f\t%.2f" % (metric, i, mean_1, mean_2))
#         print(mannwhitneyu(metric_array_1[i], metric_array_2[i]))
#         print('')
def calculate_difference_and_significance(metric, metric_array_1, metric_array_2):
    mean_1 = numpy.mean(metric_array_1)
    mean_2 = numpy.mean(metric_array_2)
    print("%s\t%.2f\t%.2f" % (metric, mean_1, mean_2))
    print(mannwhitneyu(metric_array_1, metric_array_2))
    print('')


def hypothesis_efforts(session_meta_map, session_message_map, low_group, high_group, divide_num):
    # Using duration as the dividing metric
    
    # Duration -----------------------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'duration')
    for group in [low_group, high_group]:
        for session_id in group.keys():
            duration = session_message_map[session_id]['duration']
            group[session_id]['duration'] = [duration]
    
    # Always divide_num=1 for 'duration'
    calculate_difference_and_significance('duration',
                                          retrieve_metric_array(low_group, 'duration', 1),
                                          retrieve_metric_array(high_group, 'duration', 1))
    print('\n\n')
    
    # Utterances (Overall) -----------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'utterance_overall')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    session_metric_array[index] += 1
                
                group[session_id]['utterance_overall'] = session_metric_array
        
        calculate_difference_and_significance('utterance-overall',
                                              retrieve_metric_array(low_group, 'utterance_overall', i),
                                              retrieve_metric_array(high_group, 'utterance_overall', i))  
    print('\n\n')
    
    # Utterances (Tutor) -------------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'utterance_tutor')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    role = record['role']
                    if role == 'tutor':
                        session_metric_array[index] += 1
                
                group[session_id]['utterance_tutor'] = session_metric_array
        
        calculate_difference_and_significance('utterance_tutor',
                                              retrieve_metric_array(low_group, 'utterance_tutor', i),
                                              retrieve_metric_array(high_group, 'utterance_tutor', i))  
    print('\n\n')
    
    # Utterances (Student) -----------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'utterance_student')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    role = record['role']
                    if role == 'student':
                        session_metric_array[index] += 1
                
                group[session_id]['utterance_student'] = session_metric_array
        
        calculate_difference_and_significance('utterance_student',
                                              retrieve_metric_array(low_group, 'utterance_student', i),
                                              retrieve_metric_array(high_group, 'utterance_student', i))  
    print('\n\n')
    
    # Utterance words (Overall) ------------------------------------------------
    print('Metric is:\t%s ----------' % 'utterance_words_overall')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text':
                        text = record['text'].lower()
                        words = WordPunctTokenizer().tokenize(text)
                        session_metric_array[index] += len(words)
                
                group[session_id]['utterance_words_overall'] = session_metric_array
        
        calculate_difference_and_significance('utterance_words_overall',
                                              retrieve_metric_array(low_group, 'utterance_words_overall', i),
                                              retrieve_metric_array(high_group, 'utterance_words_overall', i))
    print('\n\n')
    
    # Utterance words (Tutor) --------------------------------------------------
    print('Metric is:\t%s ----------' % 'utterance_words_tutor')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text' and record['role'] == 'tutor':
                        text = record['text'].lower()
                        words = WordPunctTokenizer().tokenize(text)
                        session_metric_array[index] += len(words)
                
                group[session_id]['utterance_words_tutor'] = session_metric_array
        
        calculate_difference_and_significance('utterance_words_tutor',
                                              retrieve_metric_array(low_group, 'utterance_words_tutor', i),
                                              retrieve_metric_array(high_group, 'utterance_words_tutor', i))  
    print('\n\n')
    
    # Utterance words (Student) ------------------------------------------------
    print('Metric is:\t%s ----------' % 'utterance_words_student')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text' and record['role'] == 'student':
                        text = record['text'].lower()
                        words = WordPunctTokenizer().tokenize(text)
                        session_metric_array[index] += len(words)
                
                group[session_id]['utterance_words_student'] = session_metric_array
        
        calculate_difference_and_significance('utterance_words_student',
                                              retrieve_metric_array(low_group, 'utterance_words_student', i),
                                              retrieve_metric_array(high_group, 'utterance_words_student', i))  
    print('\n\n')
    
    # Utterance unique words (Overall) -----------------------------------------
    print('Metric is:\t%s ----------' % 'utterance_unique_words_overall')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = []
                for j in range(i):
                    session_metric_array.append(set())
                
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text':
                        text = record['text'].lower()
                        words = WordPunctTokenizer().tokenize(text)
                        session_metric_array[index] = session_metric_array[index] | set(words)
                        
                for j in range(i):
                    session_metric_array[j] = len(session_metric_array[j])
                
                group[session_id]['utterance_unique_words_overall'] = session_metric_array
        
        calculate_difference_and_significance('utterance_unique_words_overall',
                                              retrieve_metric_array(low_group, 'utterance_unique_words_overall', i),
                                              retrieve_metric_array(high_group, 'utterance_unique_words_overall', i))
    print('\n\n')
    

def hypothesis_informativeness(session_meta_map, session_message_map, low_group, high_group, path, divide_num):
    '''
    # Unique words (Tutor/Student) -------------------------------------------
    for group in [low_group, high_group]:
        for session_id in group.keys():            
            tutor_word_set = set()
            student_word_set = set()
            
            for record in session_message_map[session_id]['time_role_message_array']:
                if record['content_type'] == 'text':
                    text = record['text'].lower()
                    try:
                        words = word_tokenize(text)
                        if record['role'] == 'tutor':
                            tutor_word_set = tutor_word_set | set(words)
                        if record['role'] == 'student':
                            student_word_set = student_word_set | set(words)
                    except:
                        pass
            group[session_id]['unique_words_tutor'] = len(tutor_word_set)
            group[session_id]['unique_words_student'] = len(student_word_set)
            
    calculate_difference_and_significance('unique_words_tutor',
                                          retrieve_metric_array(low_group, 'unique_words_tutor'),
                                          retrieve_metric_array(high_group, 'unique_words_tutor'))
    
    calculate_difference_and_significance('unique_words_student',
                                          retrieve_metric_array(low_group, 'unique_words_student'),
                                          retrieve_metric_array(high_group, 'unique_words_student'))
    print('\n\n')
    
    # Unique concepts (Tutor/Student) ------------------------------------------
    concept_frequency_map = json.loads(open(path + 'concept_frequency_map', 'r').read())
    considered_concepts = set([k for k,v in concept_frequency_map.items() if v>= 1])
    print('# considered concepts is:\t%d' % len(considered_concepts))
    
    for group in [low_group, high_group]:
        for session_id in group.keys():
            tutor_concept_set = set()
            student_concept_set = set()
            
            for record in session_message_map[session_id]['time_role_message_array']:
                if record['content_type'] == 'text':
                    text = record['text'].lower()
                    try:
                        words = word_tokenize(text)
                        if record['role'] == 'tutor':
                            tutor_concept_set = tutor_concept_set | set.intersection(set(words), considered_concepts)
                        if record['role'] == 'student':
                            student_concept_set = student_concept_set | set.intersection(set(words), considered_concepts)
                    except:
                        pass  
            group[session_id]['unique_concepts_tutor'] = len(tutor_concept_set)
            group[session_id]['unique_concepts_student'] = len(student_concept_set)
            
    calculate_difference_and_significance('unique_concepts_tutor',
                                          retrieve_metric_array(low_group, 'unique_concepts_tutor'),
                                          retrieve_metric_array(high_group, 'unique_concepts_tutor'))
    
    calculate_difference_and_significance('unique_concepts_student',
                                          retrieve_metric_array(low_group, 'unique_concepts_student'),
                                          retrieve_metric_array(high_group, 'unique_concepts_student'))
    
    print('\n\n')
    
    # Unique nouns (Tutor/Student) ---------------------------------------------
    noun_frequency_map = json.loads(open(path + 'noun_frequency_map', 'r').read())
    considered_nouns = set([k for k,v in noun_frequency_map.items() if v>= 1])
    print('# considered nouns is:\t%d' % len(considered_nouns))
    
    for group in [low_group, high_group]:
        for session_id in group.keys():
            tutor_noun_set = set()
            student_noun_set = set()
            
            for record in session_message_map[session_id]['time_role_message_array']:
                if record['content_type'] == 'text':
                    text = record['text'].lower()
                    try:
                        words = word_tokenize(text)
                        if record['role'] == 'tutor':
                            tutor_noun_set = tutor_noun_set | set.intersection(set(words), considered_nouns)
                        if record['role'] == 'student':
                            student_noun_set = student_noun_set | set.intersection(set(words), considered_nouns)
                    except:
                        pass
            group[session_id]['unique_nouns_tutor'] = len(tutor_noun_set)
            group[session_id]['unique_nouns_student'] = len(student_noun_set)
            
    calculate_difference_and_significance('unique_nouns_tutor',
                                          retrieve_metric_array(low_group, 'unique_nouns_tutor'),
                                          retrieve_metric_array(high_group, 'unique_nouns_tutor'))
    
    calculate_difference_and_significance('unique_nouns_student',
                                          retrieve_metric_array(low_group, 'unique_nouns_student'),
                                          retrieve_metric_array(high_group, 'unique_nouns_student'))
    
    print('\n\n')
    '''
    
    # New words (Tutor/Student) ------------------------------------------------
    for group in [low_group, high_group]:
        for session_id in group.keys():
            tutor_cnt = 0
            student_cnt = 0
            word_set = set()
            for record in session_message_map[session_id]['time_role_message_array']:                
                if record['content_type'] == 'text':
                    text = record['text'].lower()
                    
                    try:
                        words = word_tokenize(text)
                        
                        if record['role'] == 'tutor':
                            tutor_cnt += len(set(words) - word_set)
                            
                        if record['role'] == 'student':
                            student_cnt += len(set(words) - word_set)                               
                        
                        word_set = word_set | set(words)
                    except:
                        pass
            
            group[session_id]['new_words_tutor'] = tutor_cnt / float(len(word_set)) * 100
            group[session_id]['new_words_student'] = student_cnt  / float(len(word_set)) * 100
    
    calculate_difference_and_significance('new_words_tutor',
                                          retrieve_metric_array(low_group, 'new_words_tutor'),
                                          retrieve_metric_array(high_group, 'new_words_tutor'))
    calculate_difference_and_significance('new_words_student',
                                          retrieve_metric_array(low_group, 'new_words_student'),
                                          retrieve_metric_array(high_group, 'new_words_student'))
    print('\n\n')
    
    # New concepts (Tutor/Student) ---------------------------------------------
    noun_frequency_map = json.loads(open(path + 'noun_frequency_map', 'r').read())
    considered_nouns = set([k for k,v in noun_frequency_map.items() if v>= 1])
    print('# considered nouns is:\t%d' % len(considered_nouns))
    
    for group in [low_group, high_group]:
        for session_id in group.keys():
            tutor_cnt = 0
            student_cnt = 0
            concept_set = set()
            for record in session_message_map[session_id]['time_role_message_array']:                
                if record['content_type'] == 'text':
                    text = record['text'].lower()
                    
                    try:
                        words = word_tokenize(text)
                        concepts = set.intersection(set(words), considered_nouns)
                        
                        if record['role'] == 'tutor':
                            tutor_cnt += len(concepts - concept_set)
                            
                        if record['role'] == 'student':
                            student_cnt += len(concepts - concept_set)                        
                        
                        concept_set = concept_set | concepts
                    except:
                        pass
            
            group[session_id]['new_concepts_tutor'] = tutor_cnt / float(len(concept_set)) * 100
            group[session_id]['new_concepts_student'] = student_cnt / float(len(concept_set)) * 100
    
    calculate_difference_and_significance('new_concepts_tutor',
                                          retrieve_metric_array(low_group, 'new_concepts_tutor'),
                                          retrieve_metric_array(high_group, 'new_concepts_tutor'))
    calculate_difference_and_significance('new_concepts_student',
                                          retrieve_metric_array(low_group, 'new_concepts_student'),
                                          retrieve_metric_array(high_group, 'new_concepts_student'))
    print('\n\n')
        
    '''
    # New words (Tutor) --------------------------------------------------------
    print('Metric is:\t%s ----------' % 'new_words_tutor')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                word_set = set()
                
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text':
                        text = record['text'].lower()
                        words = WordPunctTokenizer().tokenize(text)
                        
                        if record['role'] == 'tutor':
                            for word in words:
                                if word not in word_set:
                                    session_metric_array[index] += 1
                        
                        for word in words:
                            word_set.add(word)
                
                group[session_id]['new_words_tutor'] = session_metric_array
        
        calculate_difference_and_significance('new_words_tutor',
                                              retrieve_metric_array(low_group, 'new_words_tutor', i),
                                              retrieve_metric_array(high_group, 'new_words_tutor', i))
    print('\n\n')
    
    # New words (Student) --------------------------------------------------------
    print('Metric is:\t%s ----------' % 'new_words_student')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                word_set = set()
                
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text':
                        text = record['text'].lower()
                        words = WordPunctTokenizer().tokenize(text)
                        
                        if record['role'] == 'student':
                            for word in words:
                                if word not in word_set:
                                    session_metric_array[index] += 1
                        
                        for word in words:
                            word_set.add(word)
                
                group[session_id]['new_words_student'] = session_metric_array
        
        calculate_difference_and_significance('new_words_student',
                                              retrieve_metric_array(low_group, 'new_words_student', i),
                                              retrieve_metric_array(high_group, 'new_words_student', i))
    print('\n\n')
    
    # NER (Overall) ------------------------------------------------------------
    print('Metric is:\t%s ----------' % 'ner_overall')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text':
                        text = record['text']
                        try:
                            tokenized_text = word_tokenize(text)
                            tagged_text = pos_tag(tokenized_text)
                            ne_tree = ne_chunk(tagged_text)
                            iob_tagged = tree2conlltags(ne_tree)
                            for element in iob_tagged:                        
                                ne_type = element[2]
                                if ne_type.startswith("B-"):
                                    session_metric_array[index] += 1
                        except Exception as e:
                            pass
                            
                        
                group[session_id]['ner_overall'] = session_metric_array
        
        calculate_difference_and_significance('ner_overall',
                                              retrieve_metric_array(low_group, 'ner_overall', i),
                                              retrieve_metric_array(high_group, 'ner_overall', i))
    print('\n\n')
       
    # NER (Tutor) --------------------------------------------------------------
    print('Metric is:\t%s ----------' % 'ner_tutor')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text' and record['role'] == 'tutor':
                        text = record['text']
                        try:
                            tokenized_text = word_tokenize(text)
                            tagged_text = pos_tag(tokenized_text)
                            ne_tree = ne_chunk(tagged_text)
                            iob_tagged = tree2conlltags(ne_tree)
                            for element in iob_tagged:                        
                                ne_type = element[2]
                                if ne_type.startswith("B-"):
                                    session_metric_array[index] += 1
                        except Exception as e:
                            pass
                        
                group[session_id]['ner_tutor'] = session_metric_array
        
        calculate_difference_and_significance('ner_tutor',
                                              retrieve_metric_array(low_group, 'ner_tutor', i),
                                              retrieve_metric_array(high_group, 'ner_tutor', i))
    print('\n\n')
    
    # NER (Student) --------------------------------------------------------------
    print('Metric is:\t%s ----------' % 'ner_student')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text' and record['role'] == 'student':
                        text = record['text']
                        try:
                            tokenized_text = word_tokenize(text)
                            tagged_text = pos_tag(tokenized_text)
                            ne_tree = ne_chunk(tagged_text)
                            iob_tagged = tree2conlltags(ne_tree)
                            for element in iob_tagged:                        
                                ne_type = element[2]
                                if ne_type.startswith("B-"):
                                    session_metric_array[index] += 1
                        except Exception as e:
                            pass
                        
                group[session_id]['ner_student'] = session_metric_array
        
        calculate_difference_and_significance('ner_student',
                                              retrieve_metric_array(low_group, 'ner_student', i),
                                              retrieve_metric_array(high_group, 'ner_student', i))
    print('\n\n')
    
    # New NER (Tutor) --------------------------------------------------------------
    print('Metric is:\t%s ----------' % 'new_ner_tutor')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                ner_set = set()
                
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text':
                        text = record['text']
                        try:
                            tokenized_text = word_tokenize(text)
                            tagged_text = pos_tag(tokenized_text)
                            ne_tree = ne_chunk(tagged_text)
                            iob_tagged = tree2conlltags(ne_tree)
                            for element in iob_tagged:                        
                                ne_type = element[2]
                                ner = element[0]
                                if ne_type.startswith("B-"):
                                    if ner not in ner_set:
                                        if record['role'] == 'tutor':
                                            session_metric_array[index] += 1
                                        
                            for element in iob_tagged:                        
                                ne_type = element[2]
                                ner = element[0]
                                if ne_type.startswith("B-"):
                                    ner_set.add(ner)
                                    
                        except Exception as e:
                            pass
                        
                group[session_id]['new_ner_tutor'] = session_metric_array
        
        calculate_difference_and_significance('new_ner_tutor',
                                              retrieve_metric_array(low_group, 'new_ner_tutor', i),
                                              retrieve_metric_array(high_group, 'new_ner_tutor', i))
    print('\n\n')
    
    # New NER (Tutor) --------------------------------------------------------------
    print('Metric is:\t%s ----------' % 'new_ner_student')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                ner_set = set()
                
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text':
                        text = record['text']
                        try:
                            tokenized_text = word_tokenize(text)
                            tagged_text = pos_tag(tokenized_text)
                            ne_tree = ne_chunk(tagged_text)
                            iob_tagged = tree2conlltags(ne_tree)
                            for element in iob_tagged:                        
                                ne_type = element[2]
                                ner = element[0]
                                if ne_type.startswith("B-"):
                                    if ner not in ner_set:
                                        if record['role'] == 'student':
                                            session_metric_array[index] += 1
                                        
                            for element in iob_tagged:                        
                                ne_type = element[2]
                                ner = element[0]
                                if ne_type.startswith("B-"):
                                    ner_set.add(ner)
                                    
                        except Exception as e:
                            pass
                        
                group[session_id]['new_ner_student'] = session_metric_array
        
        calculate_difference_and_significance('new_ner_student',
                                              retrieve_metric_array(low_group, 'new_ner_student', i),
                                              retrieve_metric_array(high_group, 'new_ner_student', i))
    print('\n\n')
    
    # Unique NER (Overall/Tutor/Student) --------------------------------------------------------------
    print('Metric is:\t%s ----------' % 'unique_ner_overall/tutor/student')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array_overall = [0]*i
                session_metric_array_tutor = [0]*i
                session_metric_array_student = [0]*i
                
                overall_ner_set = set()
                tutor_ner_set = set()
                student_ner_set = set()
                
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array_overall):
                        index -= 1
                    
                    if record['content_type'] == 'text':
                        text = record['text']
                        try:
                            tokenized_text = word_tokenize(text)
                            tagged_text = pos_tag(tokenized_text)
                            ne_tree = ne_chunk(tagged_text)
                            iob_tagged = tree2conlltags(ne_tree)
                            for element in iob_tagged:
                                word = element[0]                       
                                ne_type = element[2]
                                if ne_type.startswith("B-"):
                                    overall_ner_set.add(word)
                                    if record['role'] == 'tutor':
                                        tutor_ner_set.add(word)
                                    if record['role'] == 'student':
                                        student_ner_set.add(word)
                        except:
                            pass
                    
                    session_metric_array_overall[index] = len(overall_ner_set)
                    session_metric_array_tutor[index] = len(tutor_ner_set)
                    session_metric_array_student[index] = len(student_ner_set)
                        
                group[session_id]['unique_ner_overall'] = session_metric_array_overall
                group[session_id]['unique_ner_tutor'] = session_metric_array_tutor
                group[session_id]['unique_ner_student'] = session_metric_array_student
                
        
        calculate_difference_and_significance('unique_ner_overall',
                                              retrieve_metric_array(low_group, 'unique_ner_overall', i),
                                              retrieve_metric_array(high_group, 'unique_ner_overall', i))
    
        calculate_difference_and_significance('unique_ner_tutor',
                                              retrieve_metric_array(low_group, 'unique_ner_tutor', i),
                                              retrieve_metric_array(high_group, 'unique_ner_tutor', i))
        
        calculate_difference_and_significance('unique_ner_student',
                                              retrieve_metric_array(low_group, 'unique_ner_student', i),
                                              retrieve_metric_array(high_group, 'unique_ner_student', i))
        
    print('\n\n')
    '''
    
def hypothesis_complexity(session_message_map, low_group, high_group):
    # Complexity (Tutor/Student) -----------------------------------------------
    for group in [low_group, high_group]:
        for session_id in group.keys():
            tutor_metric_array = []
            student_metric_array = []
            
            for record in session_message_map[session_id]['time_role_message_array']:
                if record['content_type'] == 'text':
                    text = record['text']
                    score = textstat.flesch_reading_ease(text)
                    
                    if record['role'] == 'tutor':
                        tutor_metric_array.append(score)
                    if record['role'] == 'student':
                        student_metric_array.append(score)
            
            if len(tutor_metric_array) != 0:
                tutor_metric_array = numpy.mean(tutor_metric_array)
            else:
                tutor_metric_array = 50
                
            if len(student_metric_array) != 0:
                student_metric_array = numpy.mean(student_metric_array)
            else:
                student_metric_array = 50
            
            group[session_id]['complexity_tutor'] = tutor_metric_array
            group[session_id]['complexity_student'] = student_metric_array
    
    calculate_difference_and_significance('complexity_tutor',
                                          retrieve_metric_array(low_group, 'complexity_tutor'),
                                          retrieve_metric_array(high_group, 'complexity_tutor'))
    calculate_difference_and_significance('complexity_student',
                                          retrieve_metric_array(low_group, 'complexity_student'),
                                          retrieve_metric_array(high_group, 'complexity_student'))
    print('\n\n')
    
    '''
    # Complexity (Student) -----------------------------------------------------
    print('Metric is:\t%s ----------' % 'complexity_student')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = []
                for j in range(i):
                    session_metric_array.append([])
                
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['content_type'] == 'text' and record['role'] == 'student':
                        text = record['text']
                        session_metric_array[index].append(textstat.flesch_reading_ease(text))
                        
                for j in range(i):
                    if len(session_metric_array[j]) != 0:
                        session_metric_array[j] = numpy.mean(session_metric_array[j])
                    else:
                        session_metric_array[j] = 0
                
                group[session_id]['complexity_student'] = session_metric_array
        
        calculate_difference_and_significance('complexity_student',
                                              retrieve_metric_array(low_group, 'complexity_student', i),
                                              retrieve_metric_array(high_group, 'complexity_student', i))
    print('\n\n')
    '''

def hypothesis_responsiveness(session_meta_map, session_message_map, low_group, high_group, divide_num):
    # Wait time ----------------------------------------------------------------
    # Always divide_num = 1
    print('Metric is:\t%s ----------' % 'wait_time')
    for group in [low_group, high_group]:
        for session_id in group.keys():
            wait_time = session_meta_map[session_id]['wait_time']
            group[session_id]['wait_time'] = [wait_time]
    
    # Always divide_num=1 for 'wait_time'
    calculate_difference_and_significance('wait_time',
                                          retrieve_metric_array(low_group, 'wait_time', 1),
                                          retrieve_metric_array(high_group, 'wait_time', 1))
    print('\n\n')
    
    # Responsiveness mean (responsiveness_mean) -----------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'response_time_mean & response_time_std')
    for group in [low_group, high_group]:
        for session_id in group.keys():
            time_role_message_array = []
            
            latest_role = None    
            for record in session_message_map[session_id]['time_role_message_array']:
                role = record['role']
                created_at = record['created_at']
                if latest_role is None:
                    latest_role = role
                    time_role_message_array.append({'role':role, 'created_at':created_at})
                elif latest_role != role:
                    latest_role = role
                    time_role_message_array.append({'role':role, 'created_at':created_at})
            
            response_time_array = []
            student_latest_talk_time = None
            for record in time_role_message_array:
                role = record['role']
                created_at = record['created_at']
                
                if role == 'student':
                    student_latest_talk_time = created_at
                
                if student_latest_talk_time is not None and role == 'tutor':
                    response_time = (created_at - student_latest_talk_time).total_seconds()
                    response_time_array.append(response_time)
                
                if len(response_time_array) != 0:
                    group[session_id]['response_time_mean'] = [numpy.mean(response_time_array)]
                    group[session_id]['response_time_std'] = [numpy.std(response_time_array)]
                else:
                    group[session_id]['response_time_mean'] = [0]
                    group[session_id]['response_time_std'] = [0]
       
    # Always divide_num=1 for 'wait_time'
    calculate_difference_and_significance('response_time_mean',
                                          retrieve_metric_array(low_group, 'response_time_mean', 1),
                                          retrieve_metric_array(high_group, 'response_time_mean', 1))
    print('\n\n')
    
    calculate_difference_and_significance('response_time_std',
                                          retrieve_metric_array(low_group, 'response_time_std', 1),
                                          retrieve_metric_array(high_group, 'response_time_std', 1))
    print('\n\n')
    

def hypothesis_experience(session_meta_map, session_message_map, low_group, high_group, divide_num):
    # Tutor/Student platform experience -----------------------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'tutor_platform_experience')
    tutor_experience_map = dict()
    student_experience_map = dict()
    
    for session_id in session_message_map.keys():
        tutor_id = session_meta_map[session_id]['tutor_id']
        student_id = session_meta_map[session_id]['student_id']
        timestamp = session_meta_map[session_id]['timestamp']
        
        if tutor_id not in tutor_experience_map.keys():
            tutor_experience_map[tutor_id] = dict()
        tutor_experience_map[tutor_id][session_id] = timestamp
        
        if student_id not in student_experience_map.keys():
            student_experience_map[student_id] = dict()
        student_experience_map[student_id][session_id] = timestamp
        
    def retrieve_experience_times(current_session_id, current_timestamp, experience_dictionary):
        cnt = 0
        for session_id in experience_dictionary.keys():
            timestamp = datetime.datetime.strptime(record['timestamp'].replace(' UTC', ''), "%Y-%m-%d %H:%M:%S")
            if current_session_id != session_id:
                time_difference = (current_timestamp - timestamp).total_seconds()
                if time_difference > 0:
                    cnt += 1
                    # print('%s\t%s\t%d' % (str(timestamp), str(current_timestamp), time_difference))
        return cnt
    
    for group in [low_group, high_group]:
        for session_id in group.keys():
            tutor_id = session_meta_map[session_id]['tutor_id']
            timestamp = session_meta_map[session_id]['timestamp']
            timestamp = datetime.datetime.strptime(timestamp.replace(' UTC', ''), "%Y-%m-%d %H:%M:%S")
            tutor_platform_experience = retrieve_experience_times(session_id, timestamp, tutor_experience_map[tutor_id])
            student_platform_experience = retrieve_experience_times(session_id, timestamp, student_experience_map[student_id])
            
            group[session_id]['tutor_platform_experience'] = [tutor_platform_experience]
            group[session_id]['student_platform_experience'] = [student_platform_experience]
    
    calculate_difference_and_significance('tutor_platform_experience',
                                          retrieve_metric_array(low_group, 'tutor_platform_experience', 1),
                                          retrieve_metric_array(high_group, 'tutor_platform_experience', 1))
    
    calculate_difference_and_significance('student_platform_experience',
                                          retrieve_metric_array(low_group, 'student_platform_experience', 1),
                                          retrieve_metric_array(high_group, 'student_platform_experience', 1))
    print('\n\n')
    

def hypothesis_questions(session_meta_map, session_message_map, low_group, high_group, divide_num):
    '''
    # Questions (Overall) -----------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'questions_overall')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    text = record['text']
                    while '??' in text:
                        text = text.replace('??', '?')
                        
                    session_metric_array[index] += text.count('?')
                
                group[session_id]['question_overall'] = session_metric_array
        
        calculate_difference_and_significance('question_overall',
                                              retrieve_metric_array(low_group, 'question_overall', i),
                                              retrieve_metric_array(high_group, 'question_overall', i))  
    print('\n\n')
    '''
    
    # Questions (Tutor/Student) ------------------------------------------------
    for group in [low_group, high_group]:
        for session_id in group.keys():
            tutor_cnt = 0
            student_cnt = 0
            for record in session_message_map[session_id]['time_role_message_array']:
                text = record['text']                
                try:
                    sentences = sent_tokenize(text)
                    for sentence in sentences:
                        if sentence.endswith('?'):
                            if record['role'] == 'tutor':  
                                tutor_cnt += 1
                            if record['role'] == 'student':  
                                student_cnt += 1
                except:
                    pass    

            group[session_id]['question_tutor'] = tutor_cnt
            group[session_id]['question_student'] = student_cnt
    
    calculate_difference_and_significance('question_tutor',
                                          retrieve_metric_array(low_group, 'question_tutor'),
                                          retrieve_metric_array(high_group, 'question_tutor'))
    calculate_difference_and_significance('question_student',
                                          retrieve_metric_array(low_group, 'question_student'),
                                          retrieve_metric_array(high_group, 'question_student'))  
    print('\n\n')
    
    '''
    # Simple questions (Overall) -----------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'simple_questions_overall')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    text = record['text']
                    try:
                        sentences = sent_tokenize(text)
                        for sent in sentences:
                            sent = sent.lower()
                            if sent.endswith('?'):
                                words = word_tokenize(sent)
                                if words[0] in ['what', 'when', 'who', 'where']:
                                    session_metric_array[index] += 1
                    except Exception as e:
                        pass
                
                group[session_id]['simple_questions_overall'] = session_metric_array
        
        calculate_difference_and_significance('simple_questions_overall',
                                              retrieve_metric_array(low_group, 'simple_questions_overall', i),
                                              retrieve_metric_array(high_group, 'simple_questions_overall', i))  
    print('\n\n')
    
    # Simple questions (Tutor) -----------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'simple_questions_tutor')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['role'] == 'tutor':
                        text = record['text']
                        try:
                            sentences = sent_tokenize(text)
                            for sent in sentences:
                                sent = sent.lower()
                                if sent.endswith('?'):
                                    words = word_tokenize(sent)
                                    if words[0] in ['what', 'when', 'who', 'where']:
                                        session_metric_array[index] += 1
                        except Exception as e:
                            pass
                
                group[session_id]['simple_questions_tutor'] = session_metric_array
        
        calculate_difference_and_significance('simple_questions_tutor',
                                              retrieve_metric_array(low_group, 'simple_questions_tutor', i),
                                              retrieve_metric_array(high_group, 'simple_questions_tutor', i))  
    print('\n\n')
    
    # Simple questions (Student) -----------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'simple_questions_student')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['role'] == 'student':
                        text = record['text']
                        try:
                            sentences = sent_tokenize(text)
                            for sent in sentences:
                                sent = sent.lower()
                                if sent.endswith('?'):
                                    words = word_tokenize(sent)
                                    if words[0] in ['what', 'when', 'who', 'where']:
                                        session_metric_array[index] += 1
                        except Exception as e:
                            pass
                
                group[session_id]['simple_questions_student'] = session_metric_array
        
        calculate_difference_and_significance('simple_questions_student',
                                              retrieve_metric_array(low_group, 'simple_questions_student', i),
                                              retrieve_metric_array(high_group, 'simple_questions_student', i))  
    print('\n\n')
    
    # Complex questions (Overall) -----------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'complex_questions_overall')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    text = record['text']
                    try:
                        sentences = sent_tokenize(text)
                        for sent in sentences:
                            sent = sent.lower()
                            if sent.endswith('?'):
                                words = word_tokenize(sent)
                                if words[0] in ['why', 'how']:
                                    session_metric_array[index] += 1
                    except Exception as e:
                        pass
                
                group[session_id]['complex_questions_overall'] = session_metric_array
        
        calculate_difference_and_significance('complex_questions_overall',
                                              retrieve_metric_array(low_group, 'complex_questions_overall', i),
                                              retrieve_metric_array(high_group, 'complex_questions_overall', i))  
    print('\n\n')
    
    # Complex questions (Tutor) -----------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'complex_questions_tutor')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['role'] == 'tutor':
                        text = record['text']
                        try:
                            sentences = sent_tokenize(text)
                            for sent in sentences:
                                sent = sent.lower()
                                if sent.endswith('?'):
                                    words = word_tokenize(sent)
                                    if words[0] in ['why', 'how']:
                                        session_metric_array[index] += 1
                        except Exception as e:
                            pass
                
                group[session_id]['complex_questions_tutor'] = session_metric_array
        
        calculate_difference_and_significance('complex_questions_tutor',
                                              retrieve_metric_array(low_group, 'complex_questions_tutor', i),
                                              retrieve_metric_array(high_group, 'complex_questions_tutor', i))  
    print('\n\n')
    
    # Complex questions (Student) -----------------------------------------------------
    print('Metric is:\t%s ------------------------------' % 'complex_questions_student')
    for i in range(0,divide_num):
        i += 1
        print('DivideNum is:\t%d ----------' % i)
        for group in [low_group, high_group]:
            for session_id in group.keys():
                duration = session_message_map[session_id]['duration']
                duration_step = duration / i
                
                session_metric_array = [0]*i
                start_time = None
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    index = int(time_difference / duration_step)
                    if index >= len(session_metric_array):
                        index -= 1
                    
                    if record['role'] == 'student':
                        text = record['text']
                        try:
                            sentences = sent_tokenize(text)
                            for sent in sentences:
                                sent = sent.lower()
                                if sent.endswith('?'):
                                    words = word_tokenize(sent)
                                    if words[0] in ['why', 'how']:
                                        session_metric_array[index] += 1
                        except Exception as e:
                            pass
                
                group[session_id]['complex_questions_student'] = session_metric_array
        
        calculate_difference_and_significance('complex_questions_student',
                                              retrieve_metric_array(low_group, 'complex_questions_student', i),
                                              retrieve_metric_array(high_group, 'complex_questions_student', i))  
    print('\n\n')
    '''

def hypothesis_alignment(session_meta_map, session_message_map, low_group, high_group, path, divide_num):
    '''
    word_frequency_map = json.loads(open(path + 'word_frequency_map', 'r').read())
    print("# original words:\t%d" % len(word_frequency_map.keys()))
    word_frequency_map = {k:v for k,v in word_frequency_map.items() if v >= 5}
    print("# filtered words:\t%d" % len(word_frequency_map.keys()))
    
    considered_words = set(word_frequency_map.keys())

    # Alignment ----------------------------------------------------------------
    for group in [low_group, high_group]:
        for session_id in group.keys():
            tutor_text = ''
            student_text = ''
            for record in session_message_map[session_id]['time_role_message_array']:
                role = record['role']
                text = record['text'].lower()
                if role == 'tutor':
                    tutor_text += text + ' '
                if role == 'student':
                    student_text += text + ' '
            
            tutor_vector = [tutor_text.count(word.encode('utf-8')) for word in considered_words]
            student_vector = [student_text.count(word.encode('utf-8')) for word in considered_words]
            
            if numpy.linalg.norm(tutor_vector) == 0 or numpy.linalg.norm(student_vector) == 0:
                alignment_score = 0
            else:
                alignment_score = 1 - cosine(tutor_vector, student_vector)
            
            group[session_id]['alignment'] = [alignment_score]
    
    
    calculate_difference_and_significance('alignment',
                                          retrieve_metric_array(low_group, 'alignment'),
                                          retrieve_metric_array(high_group, 'alignment'))
    print('\n\n')
    
    # Alignment NO stopwords ---------------------------------------------------
    considered_stopwords = set(stopwords.words('english'))
    considered_words = considered_words - considered_stopwords
    print("# filtered words:\t%d" % len(considered_words))
    
    for group in [low_group, high_group]:
        for session_id in group.keys():
            tutor_text = ''
            student_text = ''
            for record in session_message_map[session_id]['time_role_message_array']:
                role = record['role']
                text = record['text'].lower()
                if role == 'tutor':
                    tutor_text += text + ' '
                if role == 'student':
                    student_text += text + ' '
            
            tutor_vector = [tutor_text.count(word.encode('utf-8')) for word in considered_words]
            student_vector = [student_text.count(word.encode('utf-8')) for word in considered_words]
            
            if numpy.linalg.norm(tutor_vector) == 0 or numpy.linalg.norm(student_vector) == 0:
                alignment_score = 0
            else:
                alignment_score = 1 - cosine(tutor_vector, student_vector)
            
            group[session_id]['alignment_nostopwords'] = [alignment_score]
    
    calculate_difference_and_significance('alignment_nostopwords',
                                          retrieve_metric_array(low_group, 'alignment_nostopwords'),
                                          retrieve_metric_array(high_group, 'alignment_nostopwords'))
    print('\n\n')
    '''
    # Alignment Concepts ---------------------------------------------------
    noun_frequency_map = json.loads(open(path + 'noun_frequency_map', 'r').read())
    noun_frequency_map = {k:v for k,v in noun_frequency_map.items() if v >= 5}
    print("# original nouns:\t%d" % len(noun_frequency_map.keys()))
    
    considered_nouns = set(noun_frequency_map.keys())
    
    for group in [low_group, high_group]:
        for session_id in group.keys():
            tutor_text = ''
            student_text = ''
            for record in session_message_map[session_id]['time_role_message_array']:
                role = record['role']
                text = record['text'].lower()
                if role == 'tutor':
                    tutor_text += text + ' '
                if role == 'student':
                    student_text += text + ' '
            
            tutor_vector = [tutor_text.count(word.encode('utf-8')) for word in considered_nouns]
            student_vector = [student_text.count(word.encode('utf-8')) for word in considered_nouns]
            
            if numpy.linalg.norm(tutor_vector) == 0 or numpy.linalg.norm(student_vector) == 0:
                alignment_score = 0
            else:
                alignment_score = 1 - cosine(tutor_vector, student_vector)
            
            group[session_id]['alignment_concept'] = [alignment_score]
    
    calculate_difference_and_significance('alignment_concept',
                                          retrieve_metric_array(low_group, 'alignment_concept'),
                                          retrieve_metric_array(high_group, 'alignment_concept'))
    print('\n\n')
    
    
# def hypothesis_sentiment(session_meta_map, session_message_map, low_group, high_group, divide_num):
#     print('Metric is:\t%s ------------------------------' % 'sentiment_overall')
#     sid = SentimentIntensityAnalyzer()
#     for i in range(0,divide_num):
#         i += 1
#         print('DivideNum is:\t%d ----------' % i)
#         for group in [low_group, high_group]:
#             for session_id in group.keys():
#                 duration = session_message_map[session_id]['duration']
#                 duration_step = duration / i
#                 
#                 session_metric_array = [0]*i
#                 
#                 start_time = None
#                 for record in session_message_map[session_id]['time_role_message_array']:
#                     created_at = record['created_at']
#                     if start_time is None:
#                         start_time = created_at
#                     time_difference = (created_at - start_time).total_seconds() / 60
#                     
#                     index = int(time_difference / duration_step)
#                     if index >= len(session_metric_array):
#                         index -= 1
#                     session_metric_array[index] += sid.polarity_scores(record['text'])['compound']
#                 
#                 group[session_id]['sentiment_overall'] = session_metric_array
#         
#         calculate_difference_and_significance('sentiment_overall',
#                                               retrieve_metric_array(low_group, 'sentiment_overall', i),
#                                               retrieve_metric_array(high_group, 'sentiment_overall', i))  
#     print('\n\n')

    
def hypothesis_sentiment(session_meta_map, session_message_map, low_group, high_group, divide_num):
    sid = SentimentIntensityAnalyzer()
    
    for group in [low_group, high_group]:
        for session_id in group.keys():
            
            '''
            tutor_array = []
            student_array = []
              
            for record in session_message_map[session_id]['time_role_message_array']:
                score = sid.polarity_scores(record['text'])['compound']
                
                if record['role'] == 'tutor':
                    tutor_array.append(score)
                if record['role'] == 'student':
                    student_array.append(score)
            
            if len(tutor_array) != 0: 
                group[session_id]['sentiment_tutor'] = numpy.mean(tutor_array)
            else:
                group[session_id]['sentiment_tutor'] = 0
            
            if len(student_array) != 0: 
                group[session_id]['sentiment_student'] = numpy.mean(student_array)
            else:
                group[session_id]['sentiment_student'] = 0
            '''
            
            tutor_score = 0
            student_score = 0
              
            for record in session_message_map[session_id]['time_role_message_array']:
                score = sid.polarity_scores(record['text'])['compound']
                
                if record['role'] == 'tutor':
                    tutor_score += score
                if record['role'] == 'student':
                    student_score += score
            
            group[session_id]['sentiment_tutor'] = tutor_score
            group[session_id]['sentiment_student'] = student_score
            
    
    calculate_difference_and_significance('sentiment_tutor',
                                          retrieve_metric_array(low_group, 'sentiment_tutor'),
                                          retrieve_metric_array(high_group, 'sentiment_tutor'))  
    
    calculate_difference_and_significance('sentiment_student',
                                          retrieve_metric_array(low_group, 'sentiment_student'),
                                          retrieve_metric_array(high_group, 'sentiment_student'))  
    
    print('\n\n')
    

 
def main():
    
    path = './data/'
    
    # 1. Compute basic statistics (unfiltered)
    session_meta_map, session_message_map = import_data(path, filter_mark=True, print_mark=False, subject_mark=False, subject_name='Math')
    
    # Retrieve low/high groups
    def retrieve_session_group(session_meta_map, session_message_map, ratings):
        session_group = dict()
        session_id_set = set(session_message_map.keys())
        for session_id in session_id_set:
            student_rating = int(session_meta_map[session_id]['student_rating'])
            if student_rating in ratings:
                session_group[session_id] = dict()
                session_group[session_id]['student_rating'] = student_rating
        print("Group size is:\t%d" % len(session_group.keys()))
        return session_group
    
    low_group = retrieve_session_group(session_meta_map, session_message_map, [1,2])
    high_group = retrieve_session_group(session_meta_map, session_message_map, [4,5])
    
    # hypothesis_efforts(session_meta_map, session_message_map, low_group, high_group, 1)
    
    # hypothesis_informativeness(session_meta_map, session_message_map, low_group, high_group, path, 1)
    
    # hypothesis_responsiveness(session_meta_map, session_message_map, low_group, high_group, 1)
    
    # hypothesis_experience(session_meta_map, session_message_map, low_group, high_group, 1)
    
    # hypothesis_questions(session_meta_map, session_message_map, low_group, high_group, 1)
    
    # hypothesis_alignment(session_meta_map, session_message_map, low_group, high_group, path, 1)
    
    hypothesis_sentiment(session_meta_map, session_message_map, low_group, high_group, 1)
    
    # hypothesis_complexity(session_message_map, low_group, high_group)
    
if __name__ == "__main__":
    main()