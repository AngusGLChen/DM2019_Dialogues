'''
Created on 17 Feb 2019

@author: gche0022
'''


from functions import import_data

import random
import numpy
import textstat
import json
import csv
import datetime
import pandas
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import bigrams, trigrams

from scipy.spatial.distance import cosine
from _collections import defaultdict
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import codecs

from collections import Counter
import os


def split_data(session_meta_map, session_message_map, path):
    session_message_map = {k:v for k,v in session_message_map.items() if session_meta_map[k]['student_rating'] != '3'}
    num_records = len(session_message_map.keys())
    print('# records/sessions is:\t%d' % num_records)
    
    data_splits = {'train':[],
                   'dev':[],
                   'test':[]}
    
    keys = list(session_message_map.keys())
    random.shuffle(keys)
    
    def sample_records(keys,num):
        selected_records = random.sample(keys, num)
        keys = list(set(keys) - set(selected_records))
        return selected_records, keys
    
    data_splits['train'], keys = sample_records(keys, int(num_records*0.80))
    data_splits['dev'], data_splits['test'] = sample_records(keys, int(num_records*0.10))
        
    def calculate_rating_distributions(session_meta_map, name, session_id_set):
        print("%s\t%d" % (name, len(session_id_set)))
        rating_array = [0] * 5
        for session_id in session_id_set:
            index = int(session_meta_map[session_id]['student_rating']) - 1
            rating_array[index] += 1
        for i in range(5):
            print('%d\t%.2f' % (i+1, rating_array[i]/float(len(session_id_set))*100))
        print('')
        
    calculate_rating_distributions(session_meta_map, 'Train', data_splits['train'])
    calculate_rating_distributions(session_meta_map, 'Dev', data_splits['dev'])
    calculate_rating_distributions(session_meta_map, 'Test', data_splits['test'])
    
    outfile = open(path + 'data_splits', 'w')
    outfile.write(json.dumps(data_splits))
    outfile.close()



    

# def generate_data_files(session_meta_map, session_message_map, path, last_mark):
# 
#     data_splits = json.loads(open(path + 'data_splits', 'r').read())
#     
#     # Tutor/Student experience -------------------------------------------------
#     tutor_experience_map = dict()
#     student_experience_map = dict()
#     
#     for session_id in session_message_map.keys():
#         tutor_id = session_meta_map[session_id]['tutor_id']
#         student_id = session_meta_map[session_id]['student_id']
#         timestamp = session_meta_map[session_id]['timestamp']
#         
#         if tutor_id not in tutor_experience_map.keys():
#             tutor_experience_map[tutor_id] = dict()
#         tutor_experience_map[tutor_id][session_id] = timestamp
#         
#         if student_id not in student_experience_map.keys():
#             student_experience_map[student_id] = dict()
#         student_experience_map[student_id][session_id] = timestamp
#         
#     # Frequent words -----------------------------------------------------------
#     word_frequency_map = json.loads(open(path + 'word_frequency_map', 'r').read())
#     print("# original words:\t%d" % len(word_frequency_map.keys()))
#     
#     word_frequency_map = {k:v for k,v in word_frequency_map.items() if v >= 5}
#     considered_words = set(word_frequency_map.keys())
#     print("# filtered words:\t%d" % len(considered_words))
#     
#     # considered_stopwords = set(stopwords.words('english'))
#     # considered_words_no_stopwords = considered_words - considered_stopwords
#     # print("# filtered words:\t%d" % len(considered_words_no_stopwords))
#     
#     noun_frequency_map = json.loads(open(path + 'noun_frequency_map', 'r').read())
#     considered_concepts = set([k for k,v in noun_frequency_map.items() if v >= 5])
#     print("# concepts:\t%d" % len(considered_concepts))
#     
#     ############################################################################
#     
#     sid = SentimentIntensityAnalyzer()
#     
#     ############################################################################
#     for key in ['train']:
#         for percentage in [0.8, 1.0]:
#             if not last_mark:
#                 outfile = open(path + key + "_" + str(percentage), 'w')
#             else:
#                 outfile = open(path + key + "_" + str(percentage) + '_last', 'w')
#             
#             writer = csv.writer(outfile)
#             
#             name_row = ['label', 'duration', 
#                         'utterance_overall', 'utterance_tutor', 'utterance_student',
#                         
#                         'utterance_words_overall', 'utterance_words_tutor', 'utterance_words_student',
#                         'utterance_unique_words_overall', 'utterance_unique_words_tutor', 'utterance_unique_words_student',
#                         'utterance_new_words_tutor', 'utterance_new_words_student',
#                         
#                         'ner_overall', 'ner_tutor', 'ner_student',
#                         'unique_ner_overall', 'unique_ner_tutor', 'unique_ner_student',
#                         'new_ner_tutor', 'new_ner_student',
#                         
#                         'complexity_overall', 'complexity_tutor', 'complexity_student',
#                         
#                         'wait_time',
#                         
#                         'responsiveness_mean', 'responsiveness_std',
#                         
#                         'tutor_experience', 'student_experience',
#                         
#                         'questions_overall', 'questions_tutor', 'questions_student',
#                         'simple_questions_overall', 'simple_questions_tutor', 'simple_questions_student',
#                         'complex_questions_overall', 'complex_questions_tutor', 'complex_questions_student',
#                         
#                         'alignment_all', 'alignment_nostopwords', 'alignment_concept',
#                         'sentiment_overall', 'sentiment_tutor', 'sentiment_student' 
#                         ]
#             
#             writer.writerow(name_row)
#             
#             for session_id in data_splits[key]:
#                 duration = session_message_map[session_id]['duration']
#                 duration_threshold = duration * percentage
#                 
#                 selected_time_role_message_array = []
#                 start_time = None
#                 
#                 for record in session_message_map[session_id]['time_role_message_array']:
#                     created_at = record['created_at']
#                     if start_time is None:
#                         start_time = created_at
#                         
#                     time_difference = (created_at - start_time).total_seconds() / 60
#                     
#                     if not last_mark:
#                         if time_difference < duration_threshold:
#                             selected_time_role_message_array.append(record)
#                         else:
#                             break
#                     else:
#                         if time_difference > duration - duration_threshold:
#                             selected_time_role_message_array.append(record)
#                 
#                 data_row = []
#                 
#                 # Label
#                 if session_meta_map[session_id]['student_rating'] in ['1', '2']:
#                     data_row.append(0)
#                 if session_meta_map[session_id]['student_rating'] in ['4', '5']:
#                     data_row.append(1)
#                     
#                 # Feature engineering ==========================================
#                 
#                 # Duration -----------------------------------------------------
#                 data_row.append(duration_threshold)
#                 
#                 # Utterances Overall/Tutor/Student -----------------------------
#                 data_row.append(len(selected_time_role_message_array))
#                 data_row.append(len([record for record in selected_time_role_message_array if record['role'] == 'tutor']))
#                 data_row.append(len([record for record in selected_time_role_message_array if record['role'] == 'student']))
#                 
#                 # Words Overall/Tutor/Student & Unique words Overall/Tutor/Student
#                 utterance_words_overall = 0
#                 utterance_words_tutor = 0
#                 utterance_words_student = 0
#                 
#                 utterance_word_set_overall = set()
#                 utterance_word_set_tutor = set()
#                 utterance_word_set_student = set()
#                 
#                 for record in selected_time_role_message_array:
#                     text = record['text'].lower()
#                     role = record['role']
#                     
#                     try:
#                         words = word_tokenize(text)
#                         
#                         utterance_words_overall += len(words)
#                         utterance_word_set_overall = utterance_word_set_overall | set(words)
#                         if role == 'tutor':
#                             utterance_words_tutor += len(words)
#                             utterance_word_set_tutor = utterance_word_set_tutor | set(words)
#                         if role == 'student':
#                             utterance_words_student += len(words)
#                             utterance_word_set_student = utterance_word_set_student | set(words)
#                     except:
#                         pass
#                         
#                 data_row.append(utterance_words_overall)
#                 data_row.append(utterance_words_tutor)
#                 data_row.append(utterance_words_student)
#                 
#                 data_row.append(len(utterance_word_set_overall))
#                 data_row.append(len(utterance_word_set_tutor))
#                 data_row.append(len(utterance_word_set_student))
#                 
#                 # New words Tutor/Student --------------------------------------
#                 utterance_new_words_tutor = 0
#                 utterance_new_words_student = 0
#                 
#                 utterance_word_set = set()
#         
#                 for record in selected_time_role_message_array:
#                     text = record['text'].lower()
#                     role = record['role']
#                     
#                     try:
#                         words = set(word_tokenize(text))
#                         if record['role'] == 'tutor':
#                             utterance_new_words_tutor += len(words - utterance_word_set)
#                         if record['role'] == 'student':
#                             utterance_new_words_student += len(words - utterance_word_set)
#                         utterance_word_set = utterance_word_set | words
#                     except:
#                         pass 
#                     
#                 data_row.append(utterance_new_words_tutor)
#                 data_row.append(utterance_new_words_student)
#                     
#                 # NER & Unique NER Overall/Tutor/Student & Unique NER Tutor/Student
#                 ner_overall = 0
#                 ner_tutor = 0
#                 ner_student = 0
#                 
#                 ner_set_overall = set()
#                 ner_set_tutor = set()
#                 ner_set_student = set()
#                 
#                 new_ner_tutor = 0
#                 new_ner_student = 0
#                 
#                 for record in selected_time_role_message_array:
#                     text = record['text'].lower()
#                     role = record['role']
#                     
#                     '''
#                     try:
#                         tokenized_text = word_tokenize(text)
#                         tagged_text = pos_tag(tokenized_text)
#                         ne_tree = ne_chunk(tagged_text)
#                         iob_tagged = tree2conlltags(ne_tree)
#                         for element in iob_tagged:  
#                             word = element[0]         
#                             ne_type = element[2]
#                             if ne_type.startswith("B-"):
#                                 ner_overall += 1
#                                 
#                                 if record['role'] == 'tutor':
#                                     ner_tutor += 1
#                                     ner_set_tutor.add(word)
#                                     if word not in ner_set_overall:
#                                         new_ner_tutor += 1
#                                     
#                                 if record['role'] == 'student':
#                                     ner_student += 1
#                                     ner_set_student.add(word)
#                                     if word not in ner_set_overall:
#                                         new_ner_student += 1
#                                     
#                                 ner_set_overall.add(word)
#                                 
#                     except:
#                         pass
#                     '''
#                     
#                     words = word_tokenize(text)
#                     for word in words:
#                         if word in considered_concepts:
#                             ner_overall += 1
#                                 
#                             if record['role'] == 'tutor':
#                                 ner_tutor += 1
#                                 ner_set_tutor.add(word)
#                                 if word not in ner_set_overall:
#                                     new_ner_tutor += 1
#                                 
#                             if record['role'] == 'student':
#                                 ner_student += 1
#                                 ner_set_student.add(word)
#                                 if word not in ner_set_overall:
#                                     new_ner_student += 1
#                                 
#                             ner_set_overall.add(word)
#                         
#                 data_row.append(ner_overall)
#                 data_row.append(ner_tutor)
#                 data_row.append(ner_student)
#                 data_row.append(len(ner_set_overall))
#                 data_row.append(len(ner_set_tutor))
#                 data_row.append(len(ner_set_student))
#                 data_row.append(new_ner_tutor)
#                 data_row.append(new_ner_student)
#                     
#                 # Complexity Overall/Tutor/Student -----------------------------
#                 complexity_overall_array = []
#                 complexity_tutor_array = []
#                 complexity_student_array = []
#                 
#                 for record in selected_time_role_message_array:
#                     text = record['text']
#                     role = record['role']
#                 
#                     score = textstat.flesch_reading_ease(text)
#                     complexity_overall_array.append(score)
#                     
#                     if role == 'tutor':
#                         complexity_tutor_array.append(score)
#                     
#                     if role == 'student':
#                         complexity_student_array.append(score)
#                         
#                 def compute_array_average(array):
#                     if len(array) != 0:
#                         return numpy.mean(array)
#                     else:
#                         return 0
#                 
#                 data_row.append(compute_array_average(complexity_overall_array))
#                 data_row.append(compute_array_average(complexity_tutor_array))
#                 data_row.append(compute_array_average(complexity_student_array))
#                 
#                 # Wait time ----------------------------------------------------
#                 data_row.append(session_meta_map[session_id]['wait_time'])
#                 
#                 # Responsiveness mean/std --------------------------------------
#                 time_role_message_array = []
#             
#                 latest_role = None    
#                 for record in selected_time_role_message_array:
#                     role = record['role']
#                     created_at = record['created_at']
#                     if latest_role is None:
#                         latest_role = role
#                         time_role_message_array.append({'role':role, 'created_at':created_at})
#                     elif latest_role != role:
#                         latest_role = role
#                         time_role_message_array.append({'role':role, 'created_at':created_at})
#             
#                 response_time_array = []
#                 student_latest_talk_time = None
#                 for record in time_role_message_array:
#                     role = record['role']
#                     created_at = record['created_at']
#                     
#                     if role == 'student':
#                         student_latest_talk_time = created_at
#                     
#                     if student_latest_talk_time is not None and role == 'tutor':
#                         response_time = (created_at - student_latest_talk_time).total_seconds()
#                         response_time_array.append(response_time)
#                     
#                 if len(response_time_array) != 0:
#                     data_row.append(numpy.mean(response_time_array))
#                     data_row.append(numpy.std(response_time_array))
#                 else:
#                     data_row.append(0)
#                     data_row.append(0)
#                     
#                 # Tutor/Student experience -------------------------------------
#                 def retrieve_experience_times(current_session_id, current_timestamp, experience_dictionary):
#                     cnt = 0
#                     for session_id in experience_dictionary.keys():
#                         timestamp = datetime.datetime.strptime(experience_dictionary[session_id].replace(' UTC', ''), "%Y-%m-%d %H:%M:%S")
#                         if current_session_id != session_id:
#                             time_difference = (current_timestamp - timestamp).total_seconds()
#                             if time_difference > 0:
#                                 cnt += 1
#                     return cnt
#     
#                 tutor_id = session_meta_map[session_id]['tutor_id']
#                 student_id = session_meta_map[session_id]['student_id']
#                 
#                 timestamp = session_meta_map[session_id]['timestamp']
#                 timestamp = datetime.datetime.strptime(timestamp.replace(' UTC', ''), "%Y-%m-%d %H:%M:%S")
#                 
#                 tutor_platform_experience = retrieve_experience_times(session_id, timestamp, tutor_experience_map[tutor_id])
#                 student_platform_experience = retrieve_experience_times(session_id, timestamp, student_experience_map[student_id])
#             
#                 data_row.append(tutor_platform_experience)
#                 data_row.append(student_platform_experience)
#                 
#                 # Questions Overall/Tutor/Student & Simple questions/Complex questions Overall/Tutor/Student
#                 questions_overall = 0
#                 questions_tutor = 0
#                 questions_student = 0
#                 
#                 simple_questions_overall = 0
#                 simple_questions_tutor = 0
#                 simple_questions_student = 0
#                 
#                 complex_questions_overall = 0
#                 complex_questions_tutor = 0
#                 complex_questions_student = 0
#                 
#                 for record in selected_time_role_message_array:
#                     text = record['text']
#                     role = record['role']
#                     
#                     try:
#                         sentences = sent_tokenize(text)
#                         cnt = 0
#                         for sentence in sentences:
#                             if sentence.endswith('?'):
#                                 cnt += 1
#                                 
#                                 sent = sent.lower()                            
#                                 words = word_tokenize(sent)
#                                 if words[0] in ['what', 'when', 'who', 'where']:
#                                     simple_questions_overall += 1
#                                     if role == 'tutor':
#                                         simple_questions_tutor += 1
#                                     if role == 'student':
#                                         simple_questions_student += 1
#                                 
#                                 if words[0] in ['why', 'how']:
#                                     complex_questions_overall += 1
#                                     if role == 'tutor':
#                                         complex_questions_tutor += 1
#                                     if role == 'student':
#                                         complex_questions_student += 1
#                             
#                         questions_overall += cnt
#                         if role == 'tutor':
#                             questions_tutor += cnt
#                         if role == 'student':
#                             questions_student += cnt
#                     except:
#                         pass
#                 
#                 data_row.append(questions_overall)
#                 data_row.append(questions_tutor)
#                 data_row.append(questions_student)
#                 
#                 data_row.append(simple_questions_overall)
#                 data_row.append(simple_questions_tutor)
#                 data_row.append(simple_questions_student)
#                 
#                 data_row.append(complex_questions_overall)
#                 data_row.append(complex_questions_tutor)
#                 data_row.append(complex_questions_student)
#                 
#                 # Alignment Overall/Tutor/Student ------------------------------
#                 tutor_text = ''
#                 student_text = ''
#                 for record in selected_time_role_message_array:
#                     role = record['role']
#                     text = record['text'].lower()
#                     if role == 'tutor':
#                         tutor_text += text + ' '
#                     if role == 'student':
#                         student_text += text + ' '
#                 
#                 # All
#                 tutor_vector = [tutor_text.count(word) for word in considered_words]
#                 student_vector = [student_text.count(word) for word in considered_words]
#                 
#                 if numpy.linalg.norm(tutor_vector) == 0 or numpy.linalg.norm(student_vector) == 0:
#                     alignment_score = 0
#                 else:
#                     alignment_score = 1 - cosine(tutor_vector, student_vector)
#                 
#                 data_row.append(alignment_score)
#                 
#                 # No stopwords
#                 tutor_vector = [tutor_text.count(word) for word in considered_words_no_stopwords]
#                 student_vector = [student_text.count(word) for word in considered_words_no_stopwords]
#                 
#                 if numpy.linalg.norm(tutor_vector) == 0 or numpy.linalg.norm(student_vector) == 0:
#                     alignment_score = 0
#                 else:
#                     alignment_score = 1 - cosine(tutor_vector, student_vector)
#                 
#                 data_row.append(alignment_score)
#                 
#                 # Concepts
#                 tutor_vector = [tutor_text.count(word) for word in considered_concepts]
#                 student_vector = [student_text.count(word) for word in considered_concepts]
#                 
#                 if numpy.linalg.norm(tutor_vector) == 0 or numpy.linalg.norm(student_vector) == 0:
#                     alignment_score = 0
#                 else:
#                     alignment_score = 1 - cosine(tutor_vector, student_vector)
#                 
#                 data_row.append(alignment_score)
#                 
#                 # Sentiment ----------------------------------------------------
#                 sentiment_overall = 0
#                 sentiment_tutor = 0
#                 sentiment_student = 0
#                 
#                 for record in selected_time_role_message_array:
#                     role = record['role']
#                     text = record['text'].lower()
#                     
#                     score = sid.polarity_scores(text)['compound']
#                     
#                     sentiment_overall += score
#                     if record['role'] == 'tutor':
#                         sentiment_tutor += score
#                     if record['role'] == 'student':
#                         sentiment_student += score
#                 
#                 data_row.append(sentiment_overall)
#                 data_row.append(sentiment_tutor)
#                 data_row.append(sentiment_student)
#                 
#                 writer.writerow(data_row)
#                 
#             outfile.close()


def merge_data_files(path, last_mark=False):
    for key in ['test', 'dev', 'train']:
        for percentage in [0.2, 0.4, 0.6, 0.8]:
            if not last_mark:
                outfile = open(path + key + "_merge_" + str(percentage), 'w')
                
                data_file = open(path + key + '_' + str(percentage), 'r').readlines()
                linguistic_data_file = open(path + key + "_linguistic_" + str(percentage), 'r').readlines()
                
                for data_line, linguistic_data_line in zip(data_file, linguistic_data_file):
                    outfile.write(data_line.strip() + ',' + linguistic_data_line)
                
                outfile.close()
            
            else:
                
                outfile = open(path + key + "_merge_" + str(percentage) + '_last', 'w')
                
                data_file = open(path + key + '_' + str(percentage) + '_last', 'r').readlines()
                linguistic_data_file = open(path + key + "_linguistic_" + str(percentage) + '_last', 'r').readlines()
                
                for data_line, linguistic_data_line in zip(data_file, linguistic_data_file):
                    outfile.write(data_line.strip() + ',' + linguistic_data_line)
                
                outfile.close()


def generate_linguistic_data_files(session_meta_map, session_message_map, path, last_mark=True):
    data_splits = json.loads(open(path + 'data_splits', 'r').read())
    stopwords_set = set(stopwords.words('english'))
    
    # unigrams &  bigrams & trigrams -------------------------------------------
    unigram_words = defaultdict(int)
    bigram_words = defaultdict(int)
    trigram_words = defaultdict(int)
    
    for session_id in session_message_map.keys():              
        for record in session_message_map[session_id]['time_role_message_array']:
            if record['content_type'] == 'text':
                text = record['text'].lower()
                try:
                    words = word_tokenize(text.decode("utf8"))
                    filtered_words = [w for w in words if (w not in stopwords_set and len(w) > 1)] 
                    
                    for filtered_word in filtered_words:
                        unigram_words[filtered_word] += 1
                    
                    bi_tokens = bigrams(filtered_words)
                    for bi_token in bi_tokens:
                        bigram_words[bi_token] += 1
                    
                    tri_tokens = trigrams(filtered_words)
                    for tri_token in tri_tokens:
                        trigram_words[tri_token] += 1    
                except Exception as e:
                    print('Preprocess:\t' + str(e) + '\t' + str(filtered_words))
    
    unigram_words = [(k,v) for (k,v) in Counter(unigram_words).most_common(100)]
    bigram_words = [(k,v) for (k,v) in Counter(bigram_words).most_common(100)]
    trigram_words = [(k,v) for (k,v) in Counter(trigram_words).most_common(100)]   
    
    print(unigram_words)
    print('')   
    print(bigram_words)
    print('')
    print(trigram_words)          
    print('')
    
    unigram_words = [k for (k,v) in unigram_words]
    bigram_words = [k for (k,v) in bigram_words]
    trigram_words = [k for (k,v) in trigram_words]
        
    ############################################################################
    for key in ['test', 'dev', 'train']:
        for percentage in [0.2, 0.4, 0.6, 0.8]:
            if not last_mark:
                outfile = open(path + key + "_linguistic_" + str(percentage), 'w')
            else:
                outfile = open(path + key + "_linguistic_" + str(percentage) + '_last', 'w')
            
            writer = csv.writer(outfile)
            
            name_row = [w for w in (unigram_words + bigram_words + trigram_words)]
            writer.writerow(name_row)
            
            for session_id in data_splits[key]:
                
                duration = session_message_map[session_id]['duration']
                duration_threshold = duration * percentage
                
                selected_time_role_message_array = []
                start_time = None
                
                data_row = [0]*300
                
                # print(len(session_message_map[session_id]['time_role_message_array']))
                
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                        
                    time_difference = (created_at - start_time).total_seconds() / float(60)
                    
                    if not last_mark:
                        if time_difference < duration_threshold:
                            selected_time_role_message_array.append(record)
                        else:
                            break
                    else:
                        if time_difference > duration - duration_threshold:
                            selected_time_role_message_array.append(record)
                
                #  print(len(selected_time_role_message_array))
                                    
                # Feature engineering ==========================================
                for record in selected_time_role_message_array:
                    text = record['text'].lower()
                    
                    try:
                        words = word_tokenize(text.decode("utf8")) 
                        
                        uni_tokens = [w for w in words if (w not in stopwords_set and len(w) > 1)] 
                        bi_tokens = [item for item in bigrams(uni_tokens)]
                        tri_tokens = [item for item in trigrams(uni_tokens)]
                        
                        # print([item for item in bi_tokens])
                                                                        
                        for token in set(uni_tokens):
                            if token in unigram_words:
                                index = unigram_words.index(token)
                                cnt = uni_tokens.count(token)
                                data_row[index] += cnt
                        
                        for token in set(bi_tokens):
                            if token in bigram_words:
                                index = bigram_words.index(token) + 100
                                cnt = bi_tokens.count(token)
                                data_row[index] += cnt
                                
                        for token in set(tri_tokens):
                            if token in trigram_words:
                                index = trigram_words.index(token) + 200
                                cnt = tri_tokens.count(token)
                                data_row[index] += cnt
                        
                    except Exception as e:
                        print(e)
                
                writer.writerow(data_row)
            
            outfile.close()


def generate_data_files(session_meta_map, session_message_map, path, last_mark):
    data_splits = json.loads(open(path + 'data_splits', 'r').read())

    # Tutor/Student experience -------------------------------------------------
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
        
    # Frequent words -----------------------------------------------------------
    word_frequency_map = json.loads(open(path + 'word_frequency_map', 'r').read())
    print("# original words:\t%d" % len(word_frequency_map.keys()))
    
    word_frequency_map = {k:v for k,v in word_frequency_map.items() if v >= 5}
    considered_words = set(word_frequency_map.keys())
    print("# filtered words:\t%d" % len(considered_words))
    
    noun_frequency_map = json.loads(open(path + 'noun_frequency_map', 'r').read())
    considered_concepts = set([k for k,v in noun_frequency_map.items() if v >= 5])
    print("# concepts:\t%d" % len(considered_concepts))
    
    ############################################################################
    
    sid = SentimentIntensityAnalyzer()
    
    ############################################################################
    for key in ['test', 'dev', 'train']:
        for percentage in [0.2, 0.4, 0.6, 0.8, 1.0]:
            if not last_mark:
                outfile = open(path + key + "_" + str(percentage), 'w')
            else:
                outfile = open(path + key + "_" + str(percentage) + '_last', 'w')
            
            writer = csv.writer(outfile)
            
            name_row = ['label', 'duration', 
                        'utterance_tutor', 'utterance_student',
                        'words_tutor', 'words_student',
                        
                        'unique_words_tutor', 'unique_words_student',
                        'unique_concepts_tutor', 'unique_concepts_student',
                        'new_words_tutor', 'new_words_student',
                        'new_concepts_tutor', 'new_concepts_student',
                        
                        'complexity_tutor', 'complexity_student',
                        
                        'wait_time', 'responsiveness_mean',
                        
                        'questions_tutor', 'questions_student',
                        
                        'alignment_all', 'alignment_concept',
                        
                        'sentiment_tutor', 'sentiment_student' ,
                        
                        'tutor_experience', 'student_experience'
                        ]
            
            writer.writerow(name_row)
            
            for session_id in data_splits[key]:
                duration = session_message_map[session_id]['duration']
                duration_threshold = duration * percentage
                
                selected_time_role_message_array = []
                start_time = None
                
                for record in session_message_map[session_id]['time_role_message_array']:
                    created_at = record['created_at']
                    if start_time is None:
                        start_time = created_at
                        
                    time_difference = (created_at - start_time).total_seconds() / 60
                    
                    if not last_mark:
                        if time_difference < duration_threshold:
                            selected_time_role_message_array.append(record)
                        else:
                            break
                    else:
                        if time_difference > duration - duration_threshold:
                            selected_time_role_message_array.append(record)
                
                data_row = []
                
                # Label
                if session_meta_map[session_id]['student_rating'] in ['1', '2']:
                    data_row.append(0)
                if session_meta_map[session_id]['student_rating'] in ['4', '5']:
                    data_row.append(1)
                    
                # Feature engineering ==========================================
                
                # Duration -----------------------------------------------------
                data_row.append(duration_threshold)
                
                # Utterances Tutor/Student -----------------------------
                data_row.append(len([record for record in selected_time_role_message_array if record['role'] == 'tutor']))
                data_row.append(len([record for record in selected_time_role_message_array if record['role'] == 'student']))
                
                # Words Tutor/Student & Unique words Tutor/Student
                words_tutor = 0
                words_student = 0
                
                word_set_tutor = set()
                word_set_student = set()
                
                concept_set_tutor = set()
                concept_set_student = set()
                
                for record in selected_time_role_message_array:
                    text = record['text'].lower()
                    role = record['role']
                    
                    try:
                        words = set(word_tokenize(text))
                        
                        if role == 'tutor':
                            words_tutor += len(words)
                            word_set_tutor = word_set_tutor | words
                            concept_set_tutor = concept_set_tutor | (set.intersection(words, considered_concepts))
                            
                        if role == 'student':
                            words_student += len(words)
                            word_set_student = word_set_student | words
                            concept_set_student = concept_set_student | (set.intersection(words, considered_concepts))
                    except:
                        pass
                        
                data_row.append(words_tutor)
                data_row.append(words_student)
                
                data_row.append(len(word_set_tutor))
                data_row.append(len(word_set_student))
                
                data_row.append(len(concept_set_tutor))
                data_row.append(len(concept_set_student))
                
                # New words & New concepts Tutor/Student -----------------------
                new_words_tutor = 0
                new_words_student = 0
                
                new_concepts_tutor = 0
                new_concepts_student = 0
                
                word_set = set()
                concept_set = set()
        
                for record in selected_time_role_message_array:
                    text = record['text'].lower()
                    role = record['role']
                    
                    try:
                        words = set(word_tokenize(text))
                        concepts = set.intersection(words, considered_concepts)
                        if record['role'] == 'tutor':
                            new_words_tutor += len(words - word_set)
                            new_concepts_tutor += len(concepts - concept_set)
                        if record['role'] == 'student':
                            new_words_student += len(words - word_set)
                            new_concepts_student += len(concepts - concept_set)
                        word_set = word_set | words
                        concept_set = concept_set | concepts
                    except:
                        pass 
                
                if len(word_set) > 0:
                    data_row.append(new_words_tutor/float(len(word_set))*100)
                    data_row.append(new_words_student/float(len(word_set))*100)
                else:
                    data_row.append(0)
                    data_row.append(0)
                
                if len(concept_set) > 0:
                    data_row.append(new_concepts_tutor/float(len(concept_set))*100)
                    data_row.append(new_concepts_student/float(len(concept_set))*100)
                else:
                    data_row.append(0)
                    data_row.append(0)
                
                # Complexity Tutor/Student -------------------------------------
                complexity_tutor_array = []
                complexity_student_array = []
                
                for record in selected_time_role_message_array:
                    text = record['text']
                    role = record['role']
                
                    score = textstat.flesch_reading_ease(text)
                    
                    if role == 'tutor':
                        complexity_tutor_array.append(score)
                    
                    if role == 'student':
                        complexity_student_array.append(score)
                        
                def compute_array_average(array):
                    if len(array) != 0:
                        return numpy.mean(array)
                    else:
                        return 50
                
                data_row.append(compute_array_average(complexity_tutor_array))
                data_row.append(compute_array_average(complexity_student_array))
                
                # Wait time ----------------------------------------------------
                data_row.append(session_meta_map[session_id]['wait_time'])
                
                # Responsiveness mean --------------------------------------
                time_role_message_array = []
            
                latest_role = None    
                for record in selected_time_role_message_array:
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
                    data_row.append(numpy.mean(response_time_array))
                else:
                    data_row.append(0)
                
                # Questions Tutor/Student --------------------------------------
                questions_tutor = 0
                questions_student = 0
                
                for record in selected_time_role_message_array:
                    text = record['text']
                    role = record['role']
                    
                    try:
                        sentences = sent_tokenize(text)
                        cnt = 0
                        for sentence in sentences:
                            if sentence.endswith('?'):
                                cnt += 1
                        if role == 'tutor':
                            questions_tutor += cnt
                        if role == 'student':
                            questions_student += cnt
                    except:
                        pass
                
                data_row.append(questions_tutor)
                data_row.append(questions_student)
                
                # Alignment ----------------------------------------------------
                tutor_text = ''
                student_text = ''
                for record in selected_time_role_message_array:
                    role = record['role']
                    text = record['text'].lower()
                    if role == 'tutor':
                        tutor_text += text + ' '
                    if role == 'student':
                        student_text += text + ' '
                
                # All
                tutor_vector = [tutor_text.count(word.encode('utf-8')) for word in considered_words]
                student_vector = [student_text.count(word.encode('utf-8')) for word in considered_words]
                
                if numpy.linalg.norm(tutor_vector) == 0 or numpy.linalg.norm(student_vector) == 0:
                    alignment_score = 0
                else:
                    alignment_score = 1 - cosine(tutor_vector, student_vector)
                
                data_row.append(alignment_score)
                
                # Concepts
                tutor_vector = [tutor_text.count(word.encode('utf-8')) for word in considered_concepts]
                student_vector = [student_text.count(word.encode('utf-8')) for word in considered_concepts]
                
                if numpy.linalg.norm(tutor_vector) == 0 or numpy.linalg.norm(student_vector) == 0:
                    alignment_score = 0
                else:
                    alignment_score = 1 - cosine(tutor_vector, student_vector)
                
                data_row.append(alignment_score)
                
                # Sentiment ----------------------------------------------------
                sentiment_tutor = 0
                sentiment_student = 0
                
                for record in selected_time_role_message_array:
                    role = record['role']
                    text = record['text']
                    
                    score = sid.polarity_scores(text)['compound']
                    
                    if record['role'] == 'tutor':
                        sentiment_tutor += score
                    if record['role'] == 'student':
                        sentiment_student += score
                
                data_row.append(sentiment_tutor)
                data_row.append(sentiment_student)
                
                # Tutor/Student experience -------------------------------------
                def retrieve_experience_times(current_session_id, current_timestamp, experience_dictionary):
                    cnt = 0
                    for session_id in experience_dictionary.keys():
                        timestamp = datetime.datetime.strptime(experience_dictionary[session_id].replace(' UTC', ''), "%Y-%m-%d %H:%M:%S")
                        if current_session_id != session_id:
                            time_difference = (current_timestamp - timestamp).total_seconds()
                            if time_difference > 0:
                                cnt += 1
                    return cnt
    
                tutor_id = session_meta_map[session_id]['tutor_id']
                student_id = session_meta_map[session_id]['student_id']
                
                timestamp = session_meta_map[session_id]['timestamp']
                timestamp = datetime.datetime.strptime(timestamp.replace(' UTC', ''), "%Y-%m-%d %H:%M:%S")
                
                tutor_platform_experience = retrieve_experience_times(session_id, timestamp, tutor_experience_map[tutor_id])
                student_platform_experience = retrieve_experience_times(session_id, timestamp, student_experience_map[student_id])
            
                data_row.append(tutor_platform_experience)
                data_row.append(student_platform_experience)
                
                writer.writerow(data_row)
            
            outfile.close()
            

def para_tuning(train, dev, scale_value):
    train_predictors = [x for x in train.columns if x not in ['label']]
    dev_predictors = [x for x in dev.columns if x not in ['label']]
    
    para_values = [x / 10.0 for x in range(1, 10, 1)]
    para_values = [0.05, 0.10, 0.15]
    para_values = [x / 100.0 for x in range(1, 10, 1)]
    
    for value in para_values:
        print("Value is:\t%.5f" % value)
        xgb = XGBClassifier(learning_rate=0.03, n_estimators=43, max_depth=5,
                            min_child_weight=4, gamma=0, subsample=0.7, 
                            colsample_bytree=0.7, reg_alpha=0.0001, objective= 'binary:logistic',
                            nthread=4, scale_pos_weight=scale_value, seed=27)
                
        xgb.fit(train[train_predictors], train['label'], eval_metric='auc')
        
        # Prediction
        dtrain_predprob = xgb.predict_proba(train[train_predictors])[:,1]    
        ddev_predprob = xgb.predict_proba(dev[dev_predictors])[:,1]
            
        print("AUC Score (Train/Test):\t%f\t%f" % (metrics.roc_auc_score(train['label'], dtrain_predprob), metrics.roc_auc_score(dev['label'], ddev_predprob)))


def para_tuning_1(train, dev, scale_value):
    xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, \
                        min_child_weight=1, gamma=0, subsample=0.8, \
                        colsample_bytree=0.8, reg_alpha=0.005, objective= 'binary:logistic', \
                        nthread=4, scale_pos_weight=scale_value, seed=27)
    modelfit(xgb1, train, dev)
    

def modelfit(alg, dtrain, dtest, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    train_predictors = [x for x in dtrain.columns if x not in ['label']]    
    if useTrainCV:        
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[train_predictors].values, label=dtrain['label'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0])
    
    # Fit the algorithm on the data
    alg.fit(dtrain[train_predictors], dtrain['label'], eval_metric='auc')
        
    # Predict training set:
    dtrain_predprob = alg.predict_proba(dtrain[train_predictors])[:,1]
    
    # Test prediction
    test_predictors = [x for x in dtest.columns if x not in ['label']]
    dtest_predprob = alg.predict_proba(dtest[test_predictors])[:,1]
      
    # Print model report:
    # print("AUC Score (Train/Test):\t%f\t%f" % (metrics.roc_auc_score(dtrain['label'], dtrain_predprob), metrics.roc_auc_score(dtest['label'], dtest_predprob)))
    
    print("AUC Score (Train/Test):\t%f\t%f" % (metrics.roc_auc_score(dtrain['label'], dtrain_predprob), metrics.roc_auc_score(dtest['label'], dtest_predprob)))
    print("Accuracy Score (Train/Test):\t%f\t%f" % (metrics.accuracy_score(dtrain['label'], [round(ele) for ele in dtrain_predprob]), metrics.accuracy_score(dtest['label'], [round(ele) for ele in dtest_predprob])))
    print("F1 Score (Train/Test):\t%f\t%f" % (metrics.f1_score(dtrain['label'], [round(ele) for ele in dtrain_predprob]), metrics.f1_score(dtest['label'], [round(ele) for ele in dtest_predprob])))
    print("Kappa Score (Train/Test):\t%f\t%f" % (metrics.cohen_kappa_score(dtrain['label'], [round(ele) for ele in dtrain_predprob]), metrics.cohen_kappa_score(dtest['label'], [round(ele) for ele in dtest_predprob])))
    
    
def para_tuning_2(train, scale_value):
    param_test1 = {'max_depth':range(1,10,1),
                   'min_child_weight':range(1,6,1)
                   }
            
    gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=43, max_depth=5,
                                                      min_child_weight=4, gamma=0, subsample=0.8, 
                                                      colsample_bytree=0.8, reg_alpha=0.005, objective= 'binary:logistic',
                                                      nthread=4, scale_pos_weight=scale_value, seed=27), 
                                                      param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    
    train_predictors = [x for x in train.columns if x not in ['label']]
    gsearch1.fit(train[train_predictors], train['label'])
    
    print(gsearch1.cv_results_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
    

def para_tuning_3(train, scale_value):
    param_test3 = {
     'gamma':[i/10.0 for i in range(0,10)]
    }
    gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=43, max_depth=5,
                                                      min_child_weight=4, gamma=0, subsample=0.8, 
                                                      colsample_bytree=0.8, reg_alpha=0.005, objective= 'binary:logistic',
                                                      nthread=4, scale_pos_weight=scale_value, seed=27), 
                                                      param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    
    train_predictors = [x for x in train.columns if x not in ['label']]
    gsearch3.fit(train[train_predictors], train['label'])
    
    print(gsearch3.cv_results_)
    print(gsearch3.best_params_)
    print(gsearch3.best_score_)
    

def para_tuning_4(train, scale_value):
    param_test4 = {
        'subsample':[i/10.0 for i in range(1,10)],
        'colsample_bytree':[i/10.0 for i in range(1,10)]
    }
    
    param_test4 = {
        'subsample':[i/100.0 for i in range(65,75,5)],
        'colsample_bytree':[i/100.0 for i in range(65,75,5)]
    }
    
    gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=43, max_depth=5,
                                                      min_child_weight=4, gamma=0, subsample=0.7, 
                                                      colsample_bytree=0.7, reg_alpha=0.005, objective= 'binary:logistic',
                                                      nthread=4, scale_pos_weight=scale_value, seed=27), 
                                                      param_grid = param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    
    train_predictors = [x for x in train.columns if x not in ['label']]
    gsearch4.fit(train[train_predictors], train['label'])
    
    print(gsearch4.cv_results_)
    print(gsearch4.best_params_)
    print(gsearch4.best_score_)
    
    
def para_tuning_5(train, scale_value):
    param_test5 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
    param_test5 = {'reg_alpha':[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    
    gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=43, max_depth=5,
                                                      min_child_weight=4, gamma=0, subsample=0.7, 
                                                      colsample_bytree=0.7, reg_alpha=0.0001, objective= 'binary:logistic',
                                                      nthread=4, scale_pos_weight=scale_value, seed=27), 
                                                      param_grid = param_test5, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    
    train_predictors = [x for x in train.columns if x not in ['label']]
    gsearch5.fit(train[train_predictors], train['label'])
    
    print(gsearch5.cv_results_)
    print(gsearch5.best_params_)
    print(gsearch5.best_score_)


def build_prediction_model(path, percentage, para_tuning_mark, last_mark):
    # Read data
    if not last_mark:
        train = pandas.read_csv(path + "train_merge_" + str(percentage))
        dev = pandas.read_csv(path + "dev_merge_" + str(percentage))
        test = pandas.read_csv(path + "test_merge_" + str(percentage))
    else:
        train = pandas.read_csv(path + "train_merge_" + str(percentage) + "_last")
        dev = pandas.read_csv(path + "dev_merge_" + str(percentage) + "_last")
        test = pandas.read_csv(path + "test_merge_" + str(percentage) + "_last")
    
    # Scale
    scale_pos_weight = {0:0, 1:0}
    for index,value in train['label'].iteritems():
        scale_pos_weight[value] += 1
    scale_value = scale_pos_weight[0] / float(scale_pos_weight[1])
    
    # Build prediction model
    train_predictors = [x for x in train.columns if x not in ['label']]
    dev_predictors = [x for x in dev.columns if x not in ['label']]
    test_predictors = [x for x in test.columns if x not in ['label']]
    
    # print(train_predictors)
    
    if para_tuning_mark:
        # Parameter: learning_rate
        para_tuning(train, dev, scale_value)
        # para_tuning_1(train, dev, scale_value)
        # para_tuning_2(train, scale_value)
        # para_tuning_3(train, scale_value)
        # para_tuning_4(train, scale_value)
        # para_tuning_5(train, scale_value)
        
    else:
        '''
        if percentage == 1.00:
            xgb = XGBClassifier(learning_rate=0.02, n_estimators=641, max_depth=4, \
                                min_child_weight=4, gamma=0.7, subsample=0.8, \
                                colsample_bytree=0.8, objective= 'binary:logistic', \
                                nthread=4, scale_pos_weight=1, seed=27)
            xgb = XGBClassifier(learning_rate=0.05, n_estimators=500, max_depth=6, \
                                min_child_weight=1, gamma=0, subsample=0.8, \
                                colsample_bytree=0.8, objective= 'binary:logistic', \
                                nthread=4, scale_pos_weight=1, seed=27)
            
        if percentage == 0.80:
            xgb = XGBClassifier(learning_rate=0.001, n_estimators=162, max_depth=7, \
                                min_child_weight=5, gamma=0.9, subsample=0.6, 
                                colsample_bytree=0.7, reg_alpha=1e-5, objective= 'binary:logistic',
                                nthread=4, scale_pos_weight=1, seed=27)
            xgb = XGBClassifier(learning_rate=0.05, n_estimators=500, max_depth=6, \
                                min_child_weight=1, gamma=0, subsample=0.8, \
                                colsample_bytree=0.8, objective= 'binary:logistic', \
                                nthread=4, scale_pos_weight=1, seed=27)
                                
        
        '''
        
        xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, \
                            min_child_weight=1, gamma=0, subsample=0.8, \
                            colsample_bytree=0.8, objective= 'binary:logistic', \
                            nthread=4, scale_pos_weight=scale_value, seed=27)
        
        xgb = XGBClassifier(learning_rate=0.05, n_estimators=500, max_depth=10, \
                            min_child_weight=3, gamma=0, subsample=0.8, \
                            colsample_bytree=0.8, objective= 'binary:logistic', \
                            nthread=4, scale_pos_weight=scale_value, seed=27)
        
           
        xgb.fit(train[train_predictors], train['label'], eval_metric='auc')
        dtest_predprob = xgb.predict_proba(test[test_predictors])[:,1]
        
        # print("There is a total of %d testing records." % test['label'].size)
        
        print("AUC/F1 Score/Kappa (Test):\t%f\t%f\t%f\t" % (metrics.roc_auc_score(test['label'], dtest_predprob),
                                                            metrics.f1_score(test['label'], dtest_predprob.round()),
                                                            metrics.cohen_kappa_score(test['label'], dtest_predprob.round())))
    

def alation_test(path):
    # Read data
    train = pandas.read_csv(path + "train_merge_1.0")
    dev = pandas.read_csv(path + "dev_merge_1.0")
    test = pandas.read_csv(path + "test_merge_1.0")
    
    # Scale
    scale_pos_weight = {0:0, 1:0}
    for index,value in train['label'].iteritems():
        scale_pos_weight[value] += 1
    scale_value = scale_pos_weight[0] / float(scale_pos_weight[1])
    
    # Build prediction model
    non_linguistic_features = ['duration', 
                'utterance_tutor', 'utterance_student',
                'words_tutor', 'words_student',
                
                'unique_words_tutor', 'unique_words_student',
                'unique_concepts_tutor', 'unique_concepts_student',
                'new_words_tutor', 'new_words_student',
                'new_concepts_tutor', 'new_concepts_student',
                
                'complexity_tutor', 'complexity_student',
                
                'wait_time', 'responsiveness_mean',
                
                'questions_tutor', 'questions_student',
                
                'alignment_all', 'alignment_concept',
                
                'sentiment_tutor', 'sentiment_student' ,
                
                'tutor_experience', 'student_experience'
                ]
    
    features_groups = [['duration', 
                        'utterance_tutor', 'utterance_student',
                        'words_tutor', 'words_student'],
                        
                        ['unique_words_tutor', 'unique_words_student',
                        'unique_concepts_tutor', 'unique_concepts_student',
                        'new_words_tutor', 'new_words_student',
                        'new_concepts_tutor', 'new_concepts_student'],
                        
                        ['complexity_tutor', 'complexity_student'],
                        
                        ['wait_time', 'responsiveness_mean'],
                        
                        ['questions_tutor', 'questions_student'],
                        
                        ['alignment_all', 'alignment_concept'],
                        
                        ['sentiment_tutor', 'sentiment_student'],
                        
                        ['tutor_experience', 'student_experience']
                        ]
    
    # Each feature in turns
    '''
    for name in name_row[1:]:
        train_predictors = [x for x in train.columns if x not in ['label', name]]
        test_predictors = [x for x in test.columns if x not in ['label', name]]
    
        xgb = XGBClassifier(learning_rate=0.05, n_estimators=500, max_depth=5, \
                            min_child_weight=1, gamma=0, subsample=0.8, \
                            colsample_bytree=0.8, objective= 'binary:logistic', \
                            nthread=4, scale_pos_weight=scale_value, seed=27)
        
        xgb.fit(train[train_predictors], train['label'], eval_metric='auc')
        dtest_predprob = xgb.predict_proba(test[test_predictors])[:,1]
        
        # print("There is a total of %d testing records." % test['label'].size)
        print("AUC/F1 Score/Kappa (Test):\t%f\t%f\t%f\t" % (metrics.roc_auc_score(test['label'], dtest_predprob),
                                                            metrics.f1_score(test['label'], dtest_predprob.round()),
                                                            metrics.cohen_kappa_score(test['label'], dtest_predprob.round())))
    '''
    # Feature groups
    k = 0
    for i in range(len(features_groups) + 3):
        
        # if i < len(features_groups):
        #     continue
        
        if i < len(features_groups):
            features_group = features_groups[i]
            features_group.append('label')
        else:
            features_group = [x for x in train.columns if x not in non_linguistic_features][100*k+1:100*(k+1)+1]
            features_group.append('label')
            k += 1
        
        train_predictors = [x for x in train.columns if x not in features_group]
        test_predictors = [x for x in test.columns if x not in features_group]
    
        xgb = XGBClassifier(learning_rate=0.05, n_estimators=500, max_depth=10, \
                            min_child_weight=3, gamma=0, subsample=0.8, \
                            colsample_bytree=0.8, objective= 'binary:logistic', \
                            nthread=4, scale_pos_weight=scale_value, seed=27)
        
        xgb.fit(train[train_predictors], train['label'], eval_metric='auc')
        dtest_predprob = xgb.predict_proba(test[test_predictors])[:,1]
        
        # print("There is a total of %d testing records." % test['label'].size)
        
        print("AUC/F1 Score/Kappa (Test):\t%f\t%f\t%f\t" % (metrics.roc_auc_score(test['label'], dtest_predprob),
                                                            metrics.f1_score(test['label'], dtest_predprob.round()),
                                                            metrics.cohen_kappa_score(test['label'], dtest_predprob.round())))
        print('')

def build_baseline_model_random(path):
    # Read data
    train = pandas.read_csv(path + "train_1.0")
    test = pandas.read_csv(path + "test_1.0")
    
    # Scale
    distribution = {0:0, 1:0}
    for index,value in train['label'].iteritems():
        distribution[value] += 1
    # scale_value = scale_pos_weight[0] / float(scale_pos_weight[1])
    
    predictions = []
    for i in range(test['label'].size):
        predictions.append(numpy.random.choice([0, 1], replace=False,
                                               p=[distribution[0] / float(distribution[0] + distribution[1]), \
                                                  distribution[1] / float(distribution[0] + distribution[1])]))
    
    # print("There is a total of %d testing records." % test['label'].size)
    print("RANDOM AUC/F1 Score/Kappa (Test):\t%f\t%f\t%f" % (metrics.roc_auc_score(test['label'], predictions),
                                                            metrics.f1_score(test['label'],  [round(ele) for ele in predictions]),
                                                            metrics.cohen_kappa_score(test['label'], [round(ele) for ele in predictions]),))

    
def build_baseline_model_random_forest(path):
    # Read data
    train = pandas.read_csv(path + "train_merge_1.0")
    dev = pandas.read_csv(path + "dev_merge_1.0")
    test = pandas.read_csv(path + "test_merge_1.0")
    
    # Scale
    # distribution = {0:0, 1:0}
    # for index,value in train['label'].iteritems():
    #     distribution[value] += 1
    
    train_predictors = [x for x in train.columns if x not in ['label']]
    dev_predictors = [x for x in dev.columns if x not in ['label']]
    test_predictors = [x for x in test.columns if x not in ['label']]
    
    clf = RandomForestClassifier(n_estimators=500, max_depth=15, max_features='auto', class_weight='balanced')
    clf.fit(train[train_predictors], train['label'])
    
    # Train
    predictions = clf.predict_proba(train[train_predictors])
    predictions = [ele[1] for ele in predictions]
    print("Random Forest AUC/F1 Score/Kappa (Train):\t%f\t%f\t%f" % (metrics.roc_auc_score(train['label'], predictions),
                                                                    metrics.f1_score(train['label'],  [round(ele) for ele in predictions]),
                                                                    metrics.cohen_kappa_score(train['label'], [round(ele) for ele in predictions])))
    # Dev
    predictions = clf.predict_proba(dev[dev_predictors])
    predictions = [ele[1] for ele in predictions]
    print("Random Forest AUC/F1 Score/Kappa (Dev):\t%f\t%f\t%f" % (metrics.roc_auc_score(dev['label'], predictions),
                                                                    metrics.f1_score(dev['label'],  [round(ele) for ele in predictions]),
                                                                    metrics.cohen_kappa_score(dev['label'], [round(ele) for ele in predictions])))
    
    # Test
    predictions = clf.predict_proba(test[test_predictors])
    predictions = [ele[1] for ele in predictions]
    print("Random Forest AUC/F1 Score/Kappa (Test):\t%f\t%f\t%f" % (metrics.roc_auc_score(test['label'], predictions),
                                                                    metrics.f1_score(test['label'],  [round(ele) for ele in predictions]),
                                                                    metrics.cohen_kappa_score(test['label'], [round(ele) for ele in predictions])))


def build_baseline_model_logistic_regression(path):
    # Read data
    train = pandas.read_csv(path + "train_1.0")
    test = pandas.read_csv(path + "test_1.0")
    
    # Scale
    distribution = {0:0, 1:0}
    for index,value in train['label'].iteritems():
        distribution[value] += 1
    
    train_predictors = [x for x in train.columns if x not in ['label']]
    test_predictors = [x for x in test.columns if x not in ['label']]
    
    clf = LogisticRegression()
    clf.fit(train[train_predictors], train['label'])
    
    predictions = clf.predict_proba(test[test_predictors])
    predictions = [ele[1] for ele in predictions]
    
    # print("There is a total of %d testing records." % test['label'].size)
    print("Logistic regression AUC/Accuracy/F1 Score/Kappa (Test): %f\t%f\t%f\t%f" % (metrics.roc_auc_score(test['label'], predictions),
                                                                                metrics.accuracy_score(test['label'], [round(ele) for ele in predictions]),
                                                                                metrics.f1_score(test['label'],  [round(ele) for ele in predictions]),
                                                                                metrics.cohen_kappa_score(test['label'], [round(ele) for ele in predictions]),))


def retrieve_frequent_ngrams(session_meta_map, session_message_map, path, percentage, last_mark=True):
    stopwords_set = set(stopwords.words('english'))
    
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
    
    # unigrams &  bigrams & trigrams -------------------------------------------
    
    if not os.path.exists(path + 'ngram_words_map'):
    
        unigram_words = defaultdict(int)
        bigram_words = defaultdict(int)
        trigram_words = defaultdict(int)
    
        for session_id in session_message_map.keys():              
            for record in session_message_map[session_id]['time_role_message_array']:
                if record['content_type'] == 'text':
                    text = record['text'].lower()
                    try:
                        words = word_tokenize(text.decode("utf8"))
                        filtered_words = [w for w in words if (w not in stopwords_set and len(w) > 1)] 
                        
                        for filtered_word in filtered_words:
                            unigram_words[filtered_word] += 1
                        
                        bi_tokens = bigrams(filtered_words)
                        for bi_token in bi_tokens:
                            bigram_words[bi_token] += 1
                        
                        tri_tokens = trigrams(filtered_words)
                        for tri_token in tri_tokens:
                            trigram_words[tri_token] += 1    
                    except Exception as e:
                        print('Preprocess:\t' + str(e) + '\t' + str(filtered_words))
        
        unigram_words = [(k,v) for (k,v) in Counter(unigram_words).most_common(100)]
        bigram_words = [(k,v) for (k,v) in Counter(bigram_words).most_common(100)]
        trigram_words = [(k,v) for (k,v) in Counter(trigram_words).most_common(100)]   
        
        print(unigram_words)
        print('')   
        print(bigram_words)
        print('')
        print(trigram_words)          
        print('')
        
        unigram_words = [k for (k,v) in unigram_words]
        bigram_words = [k for (k,v) in bigram_words]
        trigram_words = [k for (k,v) in trigram_words]
       
        ngram_words_map = {'unigrams':unigram_words,
                           'bigrams':bigram_words,
                           'trigrams':trigram_words}
        
        outfile = open(path + 'ngram_words_map', 'w')
        outfile.write(json.dumps(ngram_words_map))
        outfile.close()
        
    else:
        
        ngram_words_map = json.loads(open(path + 'ngram_words_map', 'r').read())
        unigram_words = [item for item in ngram_words_map['unigrams']]
        bigram_words = [tuple(item) for item in ngram_words_map['bigrams']]
        trigram_words = [tuple(item) for item in ngram_words_map['trigrams']]
        
    ############################################################################
    for group in [low_group, high_group]:
        
        ngrams_map = {'unigrams':defaultdict(int),
                      'bigrams':defaultdict(int),
                      'trigrams':defaultdict(int)}
        
        for session_id in group.keys():
            duration = session_message_map[session_id]['duration']
            duration_threshold = duration * percentage
            
            selected_time_role_message_array = []
            start_time = None
            
            for record in session_message_map[session_id]['time_role_message_array']:
                created_at = record['created_at']
                if start_time is None:
                    start_time = created_at
                    
                time_difference = (created_at - start_time).total_seconds() / float(60)
                
                if not last_mark:
                    if time_difference < duration_threshold:
                        selected_time_role_message_array.append(record)
                    else:
                        break
                else:
                    if time_difference > duration - duration_threshold:
                        selected_time_role_message_array.append(record)
                
                
                for record in selected_time_role_message_array:
                    text = record['text'].lower()
                    
                    try:
                        words = word_tokenize(text.decode("utf8")) 
                        
                        uni_tokens = [w for w in words if (w not in stopwords_set and len(w) > 1)] 
                        bi_tokens = [item for item in bigrams(uni_tokens)]
                        tri_tokens = [item for item in trigrams(uni_tokens)]
                                                                   
                        for token in set(uni_tokens):
                            if token in unigram_words:
                                cnt = uni_tokens.count(token)
                                ngrams_map['unigrams'][token] += cnt
                        
                        for token in set(bi_tokens):
                            if token in bigram_words:
                                cnt = bi_tokens.count(token)
                                ngrams_map['bigrams'][token] += cnt
                                
                                
                        for token in set(tri_tokens):
                            if token in trigram_words:
                                cnt = tri_tokens.count(token)
                                ngrams_map['trigrams'][token] += cnt
                        
                    except Exception as e:
                        print(e)
                        
                    # print(type(bi_tokens[0]))
                    # print(bi_tokens[0])
                
        for key in ['unigrams', 'bigrams', 'trigrams']:
            print(key)
            topk_frequent_ngrams = [k for (k,v) in Counter(ngrams_map[key]).most_common(20)]
            print(topk_frequent_ngrams)
            print('')
    



def main():
    
    path = './data/'
    
    session_meta_map, session_message_map = import_data(path, filter_mark=True, print_mark=False)
    
    # Step 1: split train/dev/test
    split_data(session_meta_map, session_message_map, path)
    
    # Step 2: generate data files
    # get_word_frequency_map(session_message_map, path)
    # generate_concept_frequency_map(session_message_map, path)
    # generate_noun_frequency_map(session_message_map, path)
    # generate_data_files(session_meta_map, session_message_map, path, last_mark=True)
    
    # generate_linguistic_data_files(session_meta_map, session_message_map, path, last_mark=True)
    
    # merge_data_files(path, last_mark=True)
    
    
    # Step 3: xgboost
    # [0.2, 0.4, 0.6, 0.8, 1.0]
    '''
    for percentage in [0.2, 0.4, 0.6, 0.8, 1.0]:
        print('Percengage is:\t%s' % str(percentage))
        build_prediction_model(path, percentage, para_tuning_mark=False, last_mark=True)
        print('')
    '''
    
    
    # Step 4: alation test
    # alation_test(path)
    
    # Step 5: baselines
    # build_baseline_model_random(path)
    # build_baseline_model_random_forest(path)
    # build_baseline_model_logistic_regression(path)
    
    # percentage = 0.2
    # retrieve_frequent_ngrams(session_meta_map, session_message_map, path, percentage, last_mark=True)


if __name__ == '__main__':
    main()