'''
Created on 13 Feb 2019

@author: gche0022
'''


import json
import numpy
import datetime

from textstat.textstat import textstat
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import cosine
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from functions import import_data, retrieve_metric_array


def calculate_difference_and_significance(metric, metric_array_1, metric_array_2):
    mean_1 = numpy.mean(metric_array_1)
    mean_2 = numpy.mean(metric_array_2)
    print("%s\t%.2f\t%.2f" % (metric, mean_1, mean_2))
    print(mannwhitneyu(metric_array_1, metric_array_2))
    print('')


def test_hypothesis_efforts(session_message_map, failure_group, success_group):
    print('Metric is:\t%s ----------' % 'Session duration (mins)')
    for group in [failure_group, success_group]:
        for session_id in group.keys():
            duration = session_message_map[session_id]['duration']
            group[session_id]['duration'] = [duration]
    calculate_difference_and_significance('duration',
                                          retrieve_metric_array(failure_group, 'duration'),
                                          retrieve_metric_array(success_group, 'duration'))
    print('\n')
    
    # Utterances (Tutor) -------------------------------------------------------
    print('Metric is:\t%s ----------' % '# Utterances (T/S)')
    
    for group in [failure_group, success_group]:
        for session_id in group.keys():
            tutor_cnt = 0
            student_cnt = 0
            for record in session_message_map[session_id]['time_role_message_array']:
                role = record['role']
                if role == 'tutor':
                    tutor_cnt += 1
                if role == 'student':
                    student_cnt += 1
            group[session_id]['# Utterances (T)'] = tutor_cnt
            group[session_id]['# Utterances (S)'] = student_cnt
    
    calculate_difference_and_significance('# Utterances (T)',
                                          retrieve_metric_array(failure_group, '# Utterances (T)'),
                                          retrieve_metric_array(success_group, '# Utterances (T)'))
    calculate_difference_and_significance('# Utterances (S)',
                                          retrieve_metric_array(failure_group, '# Utterances (S)'),
                                          retrieve_metric_array(success_group, '# Utterances (S)'))  
    print('\n')
    
    # Words (Tutor/Student) ----------------------------------------------------
    print('Metric is:\t%s ----------' % '# Words (T/S)')
    for group in [failure_group, success_group]:
        for session_id in group.keys():
            tutor_cnt = 0
            student_cnt = 0
            for record in session_message_map[session_id]['time_role_message_array']:
                if record['content_type'] == 'text':
                    text = record['text'].lower()
                    role = record['role']
                    try:
                        words = word_tokenize(text)
                        if role == 'tutor':
                            tutor_cnt += len(words)
                        if role == 'student':
                            student_cnt += len(words)
                    except:
                        pass
            
            group[session_id]['# Words (T)'] = tutor_cnt
            group[session_id]['# Words (S)'] = student_cnt
    
    calculate_difference_and_significance('# Words (T)',
                                          retrieve_metric_array(failure_group, '# Words (T)'),
                                          retrieve_metric_array(success_group, '# Words (T)'))
    calculate_difference_and_significance('# Words (S)',
                                          retrieve_metric_array(failure_group, '# Words (S)'),
                                          retrieve_metric_array(success_group, '# Words (S)')) 
     
    print('\n')
    
       
def test_hypothesis_informativeness(session_message_map, failure_group, success_group, path):
    # Unique words (Tutor/Student) ---------------------------------------------
    print('Metric is:\t%s ----------' % '# Unique words (T/S)')
    for group in [failure_group, success_group]:
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
            group[session_id]['# Unique words (T)'] = len(tutor_word_set)
            group[session_id]['# Unique words (S)'] = len(student_word_set)
            
    calculate_difference_and_significance('# Unique words (T)',
                                          retrieve_metric_array(failure_group, '# Unique words (T)'),
                                          retrieve_metric_array(success_group, '# Unique words (T)'))
    
    calculate_difference_and_significance('# Unique words (S)',
                                          retrieve_metric_array(failure_group, '# Unique words (S)'),
                                          retrieve_metric_array(success_group, '# Unique words (S)'))
    print('\n')
    
    
    # Unique concepts (Tutor/Student) ------------------------------------------
    print('Metric is:\t%s ----------' % '# Unique concepts (T/S)')
    
    concept_frequency_map = json.loads(open(path + 'noun_frequency_map', 'r').read())
    considered_concepts = set([k for k,v in concept_frequency_map.items() if v >= 1])
    print('# considered nouns is:\t%d' % len(considered_concepts))
    
    for group in [failure_group, success_group]:
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
            group[session_id]['# Unique concepts (T)'] = len(tutor_concept_set)
            group[session_id]['# Unique concepts (S)'] = len(student_concept_set)
            
    calculate_difference_and_significance('# Unique concepts (T)',
                                          retrieve_metric_array(failure_group, '# Unique concepts (T)'),
                                          retrieve_metric_array(success_group, '# Unique concepts (T)'))
    
    calculate_difference_and_significance('# Unique concepts (S)',
                                          retrieve_metric_array(failure_group, '# Unique concepts (S)'),
                                          retrieve_metric_array(success_group, '# Unique concepts (S)'))
    
    print('\n')
    
    # New words (Tutor/Student) ------------------------------------------------
    print('Metric is:\t%s ----------' % '# New words (T/S)')
    for group in [failure_group, success_group]:
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
            
            group[session_id]['# New words (T)'] = tutor_cnt / float(len(word_set)) * 100
            group[session_id]['# New words (S)'] = student_cnt  / float(len(word_set)) * 100
    
    calculate_difference_and_significance('# New words (T)',
                                          retrieve_metric_array(failure_group, '# New words (T)'),
                                          retrieve_metric_array(success_group, '# New words (T)'))
    calculate_difference_and_significance('# New words (S)',
                                          retrieve_metric_array(failure_group, '# New words (S)'),
                                          retrieve_metric_array(success_group, '# New words (S)'))
    print('\n')
    
    # New concepts (Tutor/Student) ---------------------------------------------
    print('Metric is:\t%s ----------' % '# New concepts (T/S)')
    
    noun_frequency_map = json.loads(open(path + 'noun_frequency_map', 'r').read())
    considered_nouns = set([k for k,v in noun_frequency_map.items() if v>= 1])
    # print('# considered nouns is:\t%d' % len(considered_nouns))
    
    for group in [failure_group, success_group]:
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
            group[session_id]['# New concepts (T)'] = tutor_cnt / float(len(concept_set)) * 100
            group[session_id]['# New concepts (S)'] = student_cnt / float(len(concept_set)) * 100
    
    calculate_difference_and_significance('# New concepts (T)',
                                          retrieve_metric_array(failure_group, '# New concepts (T)'),
                                          retrieve_metric_array(success_group, '# New concepts (T)'))
    calculate_difference_and_significance('# New concepts (S)',
                                          retrieve_metric_array(failure_group, '# New concepts (S)'),
                                          retrieve_metric_array(success_group, '# New concepts (S)'))
    print('\n')
    

def test_hypothesis_responsiveness(session_meta_map, session_message_map, failure_group, success_group):
    # Wait time ----------------------------------------------------------------
    print('Metric is:\t%s ----------' % 'wait_time')
    for group in [failure_group, success_group]:
        for session_id in group.keys():
            wait_time = session_meta_map[session_id]['wait_time']
            group[session_id]['wait_time'] = [wait_time]
    calculate_difference_and_significance('wait_time',
                                          retrieve_metric_array(failure_group, 'wait_time'),
                                          retrieve_metric_array(success_group, 'wait_time'))
    print('\n')
    
    # Responsiveness mean (responsiveness_mean) --------------------------------
    print('Metric is:\t%s ----------' % 'response_time_mean')
    for group in [failure_group, success_group]:
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
                else:
                    group[session_id]['response_time_mean'] = [0]
    
    calculate_difference_and_significance('response_time_mean',
                                          retrieve_metric_array(failure_group, 'response_time_mean'),
                                          retrieve_metric_array(success_group, 'response_time_mean'))
    print('\n')


def test_hypothesis_entrainment(session_message_map, failure_group, success_group, path):
    # Entrainment (All) --------------------------------------------------------
    print('Metric is:\t%s ----------' % 'Entrainment (all)')
    word_frequency_map = json.loads(open(path + 'word_frequency_map', 'r').read())
    print("# original words:\t%d" % len(word_frequency_map.keys()))
    word_frequency_map = {k:v for k,v in word_frequency_map.items() if v >= 5}
    print("# filtered words:\t%d" % len(word_frequency_map.keys()))
    
    considered_words = set(word_frequency_map.keys())

    for group in [failure_group, success_group]:
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
            
            group[session_id]['Entrainment (all)'] = [alignment_score]
    
    
    calculate_difference_and_significance('Entrainment (all)',
                                          retrieve_metric_array(failure_group, 'Entrainment (all)'),
                                          retrieve_metric_array(success_group, 'Entrainment (all)'))
    print('\n')
    
    # Entrainment Concepts -----------------------------------------------------
    print('Metric is:\t%s ----------' % 'Entrainment (concepts)')
    noun_frequency_map = json.loads(open(path + 'noun_frequency_map', 'r').read())
    print("# original nouns:\t%d" % len(word_frequency_map.keys()))
    noun_frequency_map = {k:v for k,v in noun_frequency_map.items() if v >= 5}
    print("# original nouns:\t%d" % len(noun_frequency_map.keys()))
    
    considered_nouns = set(noun_frequency_map.keys())
    
    for group in [failure_group, success_group]:
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
            
            group[session_id]['Entrainment (concepts)'] = [alignment_score]
    
    calculate_difference_and_significance('Entrainment (concepts)',
                                          retrieve_metric_array(failure_group, 'Entrainment (concepts)'),
                                          retrieve_metric_array(success_group, 'Entrainment (concepts)'))
    print('\n')
    

def test_hypothesis_complexity(session_message_map, failure_group, success_group):
    # Complexity (Tutor/Student) -----------------------------------------------
    print('Metric is:\t%s ----------' % 'Complexity (T/S)')
    for group in [failure_group, success_group]:
        for session_id in group.keys():
            tutor_array = []
            student_array = []
            
            for record in session_message_map[session_id]['time_role_message_array']:
                if record['content_type'] == 'text':
                    text = record['text']
                    score = textstat.flesch_reading_ease(text)
                    
                    if record['role'] == 'tutor':
                        tutor_array.append(score)
                    if record['role'] == 'student':
                        student_array.append(score)
            
            if len(tutor_array) != 0:
                tutor_array = numpy.mean(tutor_array)
            else:
                tutor_array = 50
                
            if len(student_array) != 0:
                student_array = numpy.mean(student_array)
            else:
                student_array = 50
            
            group[session_id]['Complexity (T)'] = tutor_array
            group[session_id]['Complexity (S)'] = student_array
    
    calculate_difference_and_significance('Complexity (T)',
                                          retrieve_metric_array(failure_group, 'Complexity (T)'),
                                          retrieve_metric_array(success_group, 'Complexity (T)'))
    calculate_difference_and_significance('Complexity (S)',
                                          retrieve_metric_array(failure_group, 'Complexity (S)'),
                                          retrieve_metric_array(success_group, 'Complexity (S)'))
    print('\n')
    

def test_hypothesis_questions(session_message_map, failure_group, success_group):
    # Questions (Tutor/Student) ------------------------------------------------
    print('Metric is:\t%s ----------' % 'Questions (T/S)')
    for group in [failure_group, success_group]:
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

            group[session_id]['Questions (T)'] = tutor_cnt
            group[session_id]['Questions (S)'] = student_cnt
    
    calculate_difference_and_significance('Questions (T)',
                                          retrieve_metric_array(failure_group, 'Questions (T)'),
                                          retrieve_metric_array(success_group, 'Questions (T)'))
    calculate_difference_and_significance('Questions (S)',
                                          retrieve_metric_array(failure_group, 'Questions (S)'),
                                          retrieve_metric_array(success_group, 'Questions (S)'))  
    print('\n')


def test_hypothesis_sentiment(session_message_map, failure_group, success_group):
    print('Metric is:\t%s ----------' % 'Sentiment (T/S)')
    sid = SentimentIntensityAnalyzer()
    for group in [failure_group, success_group]:
        for session_id in group.keys():
            tutor_score = 0
            student_score = 0
            for record in session_message_map[session_id]['time_role_message_array']:
                score = sid.polarity_scores(record['text'])['compound']
                if record['role'] == 'tutor':
                    tutor_score += score
                if record['role'] == 'student':
                    student_score += score
            group[session_id]['Sentiment (T)'] = tutor_score
            group[session_id]['Sentiment (S)'] = student_score
            
    calculate_difference_and_significance('Sentiment (T)',
                                          retrieve_metric_array(failure_group, 'Sentiment (T)'),
                                          retrieve_metric_array(success_group, 'Sentiment (T)'))  
    calculate_difference_and_significance('Sentiment (S)',
                                          retrieve_metric_array(failure_group, 'Sentiment (S)'),
                                          retrieve_metric_array(success_group, 'Sentiment (S)'))  
    
    print('\n')
    

def test_hypothesis_platform_experience(session_meta_map, session_message_map, failure_group, success_group):
    print('Metric is:\t%s ----------' % 'Platform experiences (T/S)')
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
            timestamp = datetime.datetime.strptime(experience_dictionary[session_id].replace(' UTC', ''), "%Y-%m-%d %H:%M:%S")
            if current_session_id != session_id:
                time_difference = (current_timestamp - timestamp).total_seconds()
                if time_difference > 0:
                    cnt += 1
        return cnt
    
    for group in [failure_group, success_group]:
        for session_id in group.keys():
            tutor_id = session_meta_map[session_id]['tutor_id']
            student_id = session_meta_map[session_id]['student_id']
            
            timestamp = session_meta_map[session_id]['timestamp']
            timestamp = datetime.datetime.strptime(timestamp.replace(' UTC', ''), "%Y-%m-%d %H:%M:%S")
            
            tutor_platform_experience = retrieve_experience_times(session_id, timestamp, tutor_experience_map[tutor_id])
            student_platform_experience = retrieve_experience_times(session_id, timestamp, student_experience_map[student_id])
            
            group[session_id]['Platform experiences (T)'] = tutor_platform_experience
            group[session_id]['Platform experiences (S)'] = student_platform_experience
    
    calculate_difference_and_significance('Platform experiences (T)',
                                          retrieve_metric_array(failure_group, 'Platform experiences (T)'),
                                          retrieve_metric_array(success_group, 'Platform experiences (T)'))
    
    calculate_difference_and_significance('Platform experiences (S)',
                                          retrieve_metric_array(failure_group, 'Platform experiences (S)'),
                                          retrieve_metric_array(success_group, 'Platform experiences (S)'))
    print('\n')
   
 
def main():
    
    path = './data/'
    
    # Read data
    session_meta_map, session_message_map = import_data(path, filter_mark=True, print_mark=False)
    
    # Retrieve failure/success groups
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
    
    failure_group = retrieve_session_group(session_meta_map, session_message_map, [1,2])
    success_group = retrieve_session_group(session_meta_map, session_message_map, [4,5])
    
    # Testing hypotheses
    '''
    test_hypothesis_efforts(session_message_map, failure_group, success_group)
    test_hypothesis_informativeness(session_message_map, failure_group, success_group, path)
    test_hypothesis_responsiveness(session_meta_map, session_message_map, failure_group, success_group)
    test_hypothesis_entrainment(session_message_map, failure_group, success_group, path)
    test_hypothesis_complexity(session_message_map, failure_group, success_group)
    test_hypothesis_questions(session_message_map, failure_group, success_group)
    test_hypothesis_sentiment(session_message_map, failure_group, success_group)
    '''
    test_hypothesis_platform_experience(session_meta_map, session_message_map, failure_group, success_group)
    
    
if __name__ == "__main__":
    main()