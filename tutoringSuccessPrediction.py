'''
Created on 17 Feb 2019

@author: gche0022
'''

import os
import csv
import json
import numpy
import random
import pandas
import codecs
import datetime
import textstat

from functions import import_data
from nltk.corpus import stopwords
from nltk import bigrams, trigrams
from _collections import defaultdict
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


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
        
    def analyze_rating_distributions(session_meta_map, name, session_id_set):
        print("%s\t%d" % (name, len(session_id_set)))
        rating_array = [0] * 5
        for session_id in session_id_set:
            index = int(session_meta_map[session_id]['student_rating']) - 1
            rating_array[index] += 1
        for i in range(5):
            print('%d\t%.2f' % (i+1, rating_array[i]/float(len(session_id_set))*100))
        print('')
        
    analyze_rating_distributions(session_meta_map, 'Train', data_splits['train'])
    analyze_rating_distributions(session_meta_map, 'Dev', data_splits['dev'])
    analyze_rating_distributions(session_meta_map, 'Test', data_splits['test'])
    
    outfile = open(path + 'data_splits', 'w')
    outfile.write(json.dumps(data_splits))
    outfile.close()


def generate_frequent_ngrams(session_message_map, path):
    # Stopwords
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
    
    unigram_words = [k for (k,v) in unigram_words]
    bigram_words = [k for (k,v) in bigram_words]
    trigram_words = [k for (k,v) in trigram_words]
    
    ngrams_map = {'unigrams':unigram_words,
                  'bigrams':bigram_words,
                  'trigrams':trigram_words}
    
    outfile = open(path + 'ngrams_map', 'w')
    outfile.write(json.dumps(ngrams_map, ensure_ascii=False))
    outfile.close()
    

def generate_data_files(session_meta_map, session_message_map, path, last_mark):
    # Read data splits ---------------------------------------------------------
    data_splits = json.loads(open(path + 'data_splits', 'r').read())
    stopwords_set = set(stopwords.words('english'))
    
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
    
    ngrams_map = json.loads(open(path + 'ngrams_map', 'r').read())
    unigram_words = ngrams_map['unigrams']
    bigram_words = ngrams_map['bigrams']
    trigram_words = ngrams_map['trigrams']
    
    bigram_words = [tuple(item) for item in bigram_words]
    trigram_words = [tuple(item) for item in trigram_words]
    
    ############################################################################
    for key in ['train']:
        for percentage in [0.6, 0.8, 1.0]:
            if last_mark and percentage == 1.0:
                continue
            
            if not last_mark:
                outfile = open(path + key + "_" + str(percentage), 'w')
            else:
                outfile = open(path + key + "_" + str(percentage) + '_last', 'w')
            
            writer = csv.writer(outfile)
            
            name_row = ['label',
                        
                        'duration',
                        'utterance_tutor', 'utterance_student', 
                        'words_tutor', 'words_student',
                        
                        'unique_words_tutor', 'unique_words_student',
                        'unique_concepts_tutor', 'unique_concepts_student',
                        'new_words_tutor', 'new_words_student',
                        'new_concepts_tutor', 'new_concepts_student',
                        
                        'wait_time', 'responsiveness_mean',
                        
                        'entrainment_all', 'entrainment_concept',
                        
                        'complexity_tutor', 'complexity_student',
                        
                        'questions_tutor', 'questions_student',
                        
                        'sentiment_tutor', 'sentiment_student' ,
                        
                        'tutor_experience', 'student_experience'
                        ]
            
            num_non_linguistic_features = len(name_row)
            
            # N-grams features
            for ngram in ['unigrams', 'bigrams', 'trigrams']:
                for i in range(100):
                    name_row.append(ngram + '_' + str(i))
            
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
                    
                    if percentage == 1.0:
                        selected_time_role_message_array.append(record)
                    else:
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
                    
                # Feature engineering (Non-linguistic features) ================
                
                # Duration -----------------------------------------------------
                data_row.append(duration_threshold)
                
                # Utterances Tutor/Student -----------------------------
                data_row.append(len([record for record in selected_time_role_message_array if record['role'] == 'tutor']))
                data_row.append(len([record for record in selected_time_role_message_array if record['role'] == 'student']))
                
                # Words Tutor/Student & Unique words/concepts Tutor/Student
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
                
                # Wait time ----------------------------------------------------
                data_row.append(session_meta_map[session_id]['wait_time'])
                
                # Responsiveness mean ------------------------------------------
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
                    
                # Entrainment --------------------------------------------------
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
                
                # Feature engineering (Linguistic features) ====================
                for i in range(300):
                    data_row.append(0)
                    
                for record in selected_time_role_message_array:
                    text = record['text'].lower()
                    
                    try:
                        words = word_tokenize(text.decode("utf8")) 
                        
                        uni_tokens = [w for w in words if (w not in stopwords_set and len(w) > 1)] 
                        bi_tokens = [item for item in bigrams(uni_tokens)]
                        tri_tokens = [item for item in trigrams(uni_tokens)]
                                                                        
                        for token in set(uni_tokens):
                            if token in unigram_words:
                                index = unigram_words.index(token) + num_non_linguistic_features
                                cnt = uni_tokens.count(token)
                                data_row[index] += cnt
                        
                        for token in set(bi_tokens):
                            if token in bigram_words:
                                index = bigram_words.index(token) + num_non_linguistic_features + 100
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
            
            
def build_prediction_model(path, percentage, para_tuning_mark, last_mark):
    # Read data
    if not last_mark:
        train = pandas.read_csv(path + "train_" + str(percentage))
        dev = pandas.read_csv(path + "dev_" + str(percentage))
        test = pandas.read_csv(path + "test_" + str(percentage))
    else:
        if percentage == 1.0:
            return
        train = pandas.read_csv(path + "train_" + str(percentage) + "_last")
        dev = pandas.read_csv(path + "dev_" + str(percentage) + "_last")
        test = pandas.read_csv(path + "test_" + str(percentage) + "_last")
        
    # Check whether there are any columns with all zeros
    nonzero_colums = train.loc[:, (train != 0).any(axis=0)].columns
    
    # Scale
    scale_pos_weight = {0:0, 1:0}
    for index,value in train['label'].iteritems():
        scale_pos_weight[value] += 1
    scale_value = scale_pos_weight[0] / float(scale_pos_weight[1])
    
    # Build prediction model
    predictors = [x for x in nonzero_colums if x not in ['label']]
    
    if para_tuning_mark:
        # Parameter turning guide: 
        # https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
        # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
        
        # Parameter: learning_rate
        para_tuning_0(train, dev, test, scale_value)
        # para_tuning_1(train, dev, test, scale_value)
        # para_tuning_2(train, dev, test, scale_value)
        # para_tuning_3(train, dev, test, scale_value)
        # para_tuning_4(train, dev, test, scale_value)
        
    else:
        
        xgb = XGBClassifier(learning_rate=0.015, n_estimators=686, max_depth=9,
                            min_child_weight=5, gamma=0.0, subsample=0.8, 
                            colsample_bytree=0.8, reg_alpha=0.01, objective='binary:logistic',
                            nthread=4, scale_pos_weight=scale_value, seed=27)
        
        xgb.fit(train[predictors], train['label'], eval_metric='auc')
        dtest_predprob = xgb.predict_proba(test[predictors])[:,1]
        
        print("AUC/F1 Score/Kappa (Test):\t%f\t%f\t%f\t" % (metrics.roc_auc_score(test['label'], dtest_predprob),
                                                            metrics.f1_score(test['label'], dtest_predprob.round()),
                                                            metrics.cohen_kappa_score(test['label'], dtest_predprob.round())))            

   
def modelfit(alg, dtrain, ddev, dtest, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    dtrain_predictors = [x for x in dtrain.columns if x not in ['label']]    
    
    if useTrainCV:       
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[dtrain_predictors].values, label=dtrain['label'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=True)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0])
    
    # Fit the algorithm on the data
    alg.fit(dtrain[dtrain_predictors], dtrain['label'], eval_metric='auc')
        
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[dtrain_predictors])
    dtrain_predprob = alg.predict_proba(dtrain[dtrain_predictors])[:,1]
    
    # Dev prediction
    dev_predictors = [x for x in ddev.columns if x not in ['label']]
    ddev_predictions = alg.predict(ddev[dev_predictors])
    ddev_predprob = alg.predict_proba(ddev[dev_predictors])[:,1]
    
    # Test prediction
    test_predictors = [x for x in dtest.columns if x not in ['label']]
    dtest_predictions = alg.predict(dtest[test_predictors])
    dtest_predprob = alg.predict_proba(dtest[test_predictors])[:,1]
      
    # Print model report:
    print "\nModel Report"
    print("AUC    (Train/Test):\t%f\t%f\t%f" % (metrics.roc_auc_score(dtrain['label'], dtrain_predprob), metrics.roc_auc_score(ddev['label'], ddev_predprob), metrics.roc_auc_score(dtest['label'], dtest_predprob)))
    print("F1     (Train/Test):\t%f\t%f\t%f" % (metrics.f1_score(dtrain['label'], dtrain_predictions), metrics.f1_score(ddev['label'], ddev_predictions), metrics.f1_score(dtest['label'], dtest_predictions)))
    print("Kappa  (Train/Test):\t%f\t%f\t%f" % (metrics.cohen_kappa_score(dtrain['label'], dtrain_predictions), metrics.cohen_kappa_score(ddev['label'], ddev_predictions), metrics.cohen_kappa_score(dtest['label'], dtest_predictions)))
    
    # feat_imp = pandas.Series(alg.get_booster().get_fscore()).nlargest(50)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    # plt.show()
    

def para_tuning_0(train, dev, test, scale_value):
    xgb1 = XGBClassifier(learning_rate=0.010, n_estimators=5000, max_depth=6,
                         min_child_weight=2, gamma=0, subsample=0.8,
                         colsample_bytree=0.6, reg_alpha=1e-5, objective='binary:logistic',
                         nthread=4, scale_pos_weight=scale_value, seed=27)
    
    modelfit(xgb1, train, dev, test)
    
    # n_estimators=116
    
     
def para_tuning_1(dtrain, ddev, dtest, scale_value):
    param_test1 = {'max_depth':range(3,10,2),
                   'min_child_weight':range(1,9,2)
                   }
    
    param_test1 = {'max_depth':range(6,8,1),
                   'min_child_weight':range(2,4,1)
                   }

    
    kappa_scorer = metrics.make_scorer(metrics.cohen_kappa_score)    
    gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.15, n_estimators=116, max_depth=6,
                                                      min_child_weight=2, gamma=0, subsample=0.6,
                                                      colsample_bytree=0.6, objective='binary:logistic',
                                                      nthread=4, scale_pos_weight=scale_value, seed=27), 
                                                      param_grid=param_test1, scoring=kappa_scorer, n_jobs=4, iid=False, cv=5, verbose=10)
    
    dtrain_predictors = [x for x in dtrain.columns if x not in ['label']]
    gsearch1.fit(dtrain[dtrain_predictors], dtrain['label'])
    
    # print(gsearch1.cv_results_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
    print('')
    
    alg = gsearch1.best_estimator_
    
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[dtrain_predictors])
    dtrain_predprob = alg.predict_proba(dtrain[dtrain_predictors])[:,1]
    
    # Dev prediction
    dev_predictors = [x for x in ddev.columns if x not in ['label']]
    ddev_predictions = alg.predict(ddev[dev_predictors])
    ddev_predprob = alg.predict_proba(ddev[dev_predictors])[:,1]
    
    # Test prediction
    test_predictors = [x for x in dtest.columns if x not in ['label']]
    dtest_predictions = alg.predict(dtest[test_predictors])
    dtest_predprob = alg.predict_proba(dtest[test_predictors])[:,1]
      
    # Print model report:
    print "\nModel Report"
    print("AUC    (Train/Test):\t%f\t%f\t%f" % (metrics.roc_auc_score(dtrain['label'], dtrain_predprob), metrics.roc_auc_score(ddev['label'], ddev_predprob), metrics.roc_auc_score(dtest['label'], dtest_predprob)))
    print("F1     (Train/Test):\t%f\t%f\t%f" % (metrics.f1_score(dtrain['label'], dtrain_predictions), metrics.f1_score(ddev['label'], ddev_predictions), metrics.f1_score(dtest['label'], dtest_predictions)))
    print("Kappa  (Train/Test):\t%f\t%f\t%f" % (metrics.cohen_kappa_score(dtrain['label'], dtrain_predictions), metrics.cohen_kappa_score(ddev['label'], ddev_predictions), metrics.cohen_kappa_score(dtest['label'], dtest_predictions)))
  
    # {'max_depth': 7, 'min_child_weight': 3}
    # {'max_depth': 6, 'min_child_weight': 2}
  

def para_tuning_2(dtrain, ddev, dtest, scale_value):
    param_test2 = {
     'gamma':[i/10.0 for i in range(0,10)]
    }
    
    kappa_scorer = metrics.make_scorer(metrics.cohen_kappa_score)
    gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.15, n_estimators=116, max_depth=6,
                                                      min_child_weight=2, gamma=0, subsample=0.6,
                                                      colsample_bytree=0.6, objective='binary:logistic',
                                                      nthread=4, scale_pos_weight=scale_value, seed=27), 
                                                      param_grid=param_test2, scoring=kappa_scorer, n_jobs=4, iid=False, cv=5, verbose=10)
    
    dtrain_predictors = [x for x in dtrain.columns if x not in ['label']]
    gsearch2.fit(dtrain[dtrain_predictors], dtrain['label'])
    
    # print(gsearch2.cv_results_)
    print(gsearch2.best_params_)
    print(gsearch2.best_score_)
    print('')
    
    alg = gsearch2.best_estimator_
    
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[dtrain_predictors])
    dtrain_predprob = alg.predict_proba(dtrain[dtrain_predictors])[:,1]
    
    # Dev prediction
    dev_predictors = [x for x in ddev.columns if x not in ['label']]
    ddev_predictions = alg.predict(ddev[dev_predictors])
    ddev_predprob = alg.predict_proba(ddev[dev_predictors])[:,1]
    
    # Test prediction
    test_predictors = [x for x in dtest.columns if x not in ['label']]
    dtest_predictions = alg.predict(dtest[test_predictors])
    dtest_predprob = alg.predict_proba(dtest[test_predictors])[:,1]
      
    # Print model report:
    print "\nModel Report"
    print("AUC    (Train/Test):\t%f\t%f\t%f" % (metrics.roc_auc_score(dtrain['label'], dtrain_predprob), metrics.roc_auc_score(ddev['label'], ddev_predprob), metrics.roc_auc_score(dtest['label'], dtest_predprob)))
    print("F1     (Train/Test):\t%f\t%f\t%f" % (metrics.f1_score(dtrain['label'], dtrain_predictions), metrics.f1_score(ddev['label'], ddev_predictions), metrics.f1_score(dtest['label'], dtest_predictions)))
    print("Kappa  (Train/Test):\t%f\t%f\t%f" % (metrics.cohen_kappa_score(dtrain['label'], dtrain_predictions), metrics.cohen_kappa_score(ddev['label'], ddev_predictions), metrics.cohen_kappa_score(dtest['label'], dtest_predictions)))
  

def para_tuning_3(dtrain, ddev, dtest, scale_value):
    param_test3 = {
        'subsample':[i/10.0 for i in range(5,10)],
        'colsample_bytree':[i/10.0 for i in range(5,10)]
    }
    
    param_test3 = {
        'subsample':[i/100.0 for i in range(75,85,5)],
        'colsample_bytree':[i/100.0 for i in range(55,65,5)]
    }
    
    kappa_scorer = metrics.make_scorer(metrics.cohen_kappa_score)
    gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.15, n_estimators=116, max_depth=6,
                                                      min_child_weight=2, gamma=0, subsample=0.8,
                                                      colsample_bytree=0.6, objective='binary:logistic',
                                                      nthread=4, scale_pos_weight=scale_value, seed=27), 
                                                      param_grid=param_test3, scoring=kappa_scorer, n_jobs=4, iid=False, cv=5, verbose=10)
    
    
    dtrain_predictors = [x for x in dtrain.columns if x not in ['label']]
    gsearch3.fit(dtrain[dtrain_predictors], dtrain['label'])
    
    # print(gsearch3.cv_results_)
    print(gsearch3.best_params_)
    print(gsearch3.best_score_)
    print('')
    
    alg = gsearch3.best_estimator_
    
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[dtrain_predictors])
    dtrain_predprob = alg.predict_proba(dtrain[dtrain_predictors])[:,1]
    
    # Dev prediction
    dev_predictors = [x for x in ddev.columns if x not in ['label']]
    ddev_predictions = alg.predict(ddev[dev_predictors])
    ddev_predprob = alg.predict_proba(ddev[dev_predictors])[:,1]
    
    # Test prediction
    test_predictors = [x for x in dtest.columns if x not in ['label']]
    dtest_predictions = alg.predict(dtest[test_predictors])
    dtest_predprob = alg.predict_proba(dtest[test_predictors])[:,1]
      
    # Print model report:
    print "\nModel Report"
    print("AUC    (Train/Test):\t%f\t%f\t%f" % (metrics.roc_auc_score(dtrain['label'], dtrain_predprob), metrics.roc_auc_score(ddev['label'], ddev_predprob), metrics.roc_auc_score(dtest['label'], dtest_predprob)))
    print("F1     (Train/Test):\t%f\t%f\t%f" % (metrics.f1_score(dtrain['label'], dtrain_predictions), metrics.f1_score(ddev['label'], ddev_predictions), metrics.f1_score(dtest['label'], dtest_predictions)))
    print("Kappa  (Train/Test):\t%f\t%f\t%f" % (metrics.cohen_kappa_score(dtrain['label'], dtrain_predictions), metrics.cohen_kappa_score(ddev['label'], ddev_predictions), metrics.cohen_kappa_score(dtest['label'], dtest_predictions)))
  
  
def para_tuning_4(dtrain, ddev, dtest, scale_value):
    param_test4 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
    
    kappa_scorer = metrics.make_scorer(metrics.cohen_kappa_score)
    gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.15, n_estimators=116, max_depth=6,
                                                      min_child_weight=2, gamma=0, subsample=0.8,
                                                      colsample_bytree=0.6, objective='binary:logistic',
                                                      nthread=4, scale_pos_weight=scale_value, seed=27), 
                                                      param_grid=param_test4, scoring=kappa_scorer, n_jobs=4, iid=False, cv=5, verbose=10)
    
    dtrain_predictors = [x for x in dtrain.columns if x not in ['label']]
    gsearch4.fit(dtrain[dtrain_predictors], dtrain['label'])
    
    # print(gsearch4.cv_results_)
    print(gsearch4.best_params_)
    print(gsearch4.best_score_)
    print('')
    
    alg = gsearch4.best_estimator_
    
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[dtrain_predictors])
    dtrain_predprob = alg.predict_proba(dtrain[dtrain_predictors])[:,1]
    
    # Dev prediction
    dev_predictors = [x for x in ddev.columns if x not in ['label']]
    ddev_predictions = alg.predict(ddev[dev_predictors])
    ddev_predprob = alg.predict_proba(ddev[dev_predictors])[:,1]
    
    # Test prediction
    test_predictors = [x for x in dtest.columns if x not in ['label']]
    dtest_predictions = alg.predict(dtest[test_predictors])
    dtest_predprob = alg.predict_proba(dtest[test_predictors])[:,1]
      
    # Print model report:
    print "\nModel Report"
    print("AUC    (Train/Test):\t%f\t%f\t%f" % (metrics.roc_auc_score(dtrain['label'], dtrain_predprob), metrics.roc_auc_score(ddev['label'], ddev_predprob), metrics.roc_auc_score(dtest['label'], dtest_predprob)))
    print("F1     (Train/Test):\t%f\t%f\t%f" % (metrics.f1_score(dtrain['label'], dtrain_predictions), metrics.f1_score(ddev['label'], ddev_predictions), metrics.f1_score(dtest['label'], dtest_predictions)))
    print("Kappa  (Train/Test):\t%f\t%f\t%f" % (metrics.cohen_kappa_score(dtrain['label'], dtrain_predictions), metrics.cohen_kappa_score(ddev['label'], ddev_predictions), metrics.cohen_kappa_score(dtest['label'], dtest_predictions)))
  

def alation_test(path):
    # Read data
    train = pandas.read_csv(path + "train_1.0")
    test = pandas.read_csv(path + "test_1.0")
    
    # Check whether there are any columns with all zeros
    nonzero_colums = train.loc[:, (train != 0).any(axis=0)].columns
    
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
                               
                               'wait_time', 'responsiveness_mean',
                               
                               'alignment_all', 'alignment_concept',
                               
                               'complexity_tutor', 'complexity_student',
                            
                               'questions_tutor', 'questions_student',
                            
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
                        
                        ['wait_time', 'responsiveness_mean'],
                        
                        ['alignment_all', 'alignment_concept'],
                        
                        ['complexity_tutor', 'complexity_student'],
                        
                        ['questions_tutor', 'questions_student'],
                        
                        ['sentiment_tutor', 'sentiment_student'],
                        
                        ['tutor_experience', 'student_experience']
                    ]
    
    # Feature groups
    k = 0
    for i in range(len(features_groups) + 3):
        
        # if i < len(features_groups):
        #     continue
        
        if i in range(len(features_groups)):
            print(features_groups[i])
        
        if i < len(features_groups):
            features_group = features_groups[i]
            features_group.append('label')
        else:
            
            if i < len(features_groups) + 2:
                features_group = [x for x in train.columns if x not in non_linguistic_features][100*k+1:100*(k+1)+1]
            else:
                # Trigrams
                features_group = [x for x in train.columns if x not in non_linguistic_features][100*k+1:]
            
            features_group.append('label')
            k += 1
        
        train_predictors = [x for x in nonzero_colums if x not in features_group]
        test_predictors = [x for x in nonzero_colums if x not in features_group]
    
        xgb = XGBClassifier(learning_rate=0.015, n_estimators=686, max_depth=9,
                            min_child_weight=5, gamma=0.0, subsample=0.8, 
                            colsample_bytree=0.8, reg_alpha=0.01, objective='binary:logistic',
                            nthread=4, scale_pos_weight=scale_value, seed=27)
        
        xgb.fit(train[train_predictors], train['label'], eval_metric='auc')
        dtest_predprob = xgb.predict_proba(test[test_predictors])[:,1]
        
        print("AUC/F1 Score/Kappa (Test):\t%f\t%f\t%f\t" % (metrics.roc_auc_score(test['label'], dtest_predprob),
                                                            metrics.f1_score(test['label'], dtest_predprob.round()),
                                                            metrics.cohen_kappa_score(test['label'], dtest_predprob.round())))
        print('')


def build_baseline_model_random_forest(path):
    # Read data
    train = pandas.read_csv(path + "train_1.0")
    dev = pandas.read_csv(path + "dev_1.0")
    test = pandas.read_csv(path + "test_1.0")
    
    train_predictors = [x for x in train.columns if x not in ['label']]
    dev_predictors = [x for x in dev.columns if x not in ['label']]
    test_predictors = [x for x in test.columns if x not in ['label']]
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=9, max_features=None, class_weight='balanced')
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


def parameter_tuning_random_forest(path):
    # Read data
    dtrain = pandas.read_csv(path + "train_1.0")
    ddev = pandas.read_csv(path + "dev_1.0")
    dtest = pandas.read_csv(path + "test_1.0")
    
    dtrain_predictors = [x for x in dtrain.columns if x not in ['label']]
    ddev_predictors = [x for x in ddev.columns if x not in ['label']]
    dtest_predictors = [x for x in dtest.columns if x not in ['label']]
    
    scoring = {'AUC': metrics.make_scorer(metrics.roc_auc_score),
               'Cohen\'s Kappa': metrics.make_scorer(metrics.cohen_kappa_score),
               'F1':metrics.make_scorer(metrics.f1_score)}
    
    clf=RandomForestClassifier(random_state=27, class_weight='balanced')
    param_grid = {'n_estimators': range(25,201,25),
                  'max_features': ['auto', 'sqrt', 'log2', None],
                  'max_depth': range(3,10,1)
                  }
    
    gsearch = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5,
                           scoring=scoring,
                           refit='Cohen\'s Kappa', return_train_score=True, verbose=10)
    gsearch.fit(dtrain[dtrain_predictors], dtrain['label'])
    
    print(gsearch.best_params_)
    print('')
    
    alg = gsearch.best_estimator_
    
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[dtrain_predictors])
    dtrain_predprob = alg.predict_proba(dtrain[dtrain_predictors])[:,1]
    
    # Dev prediction
    dev_predictors = [x for x in ddev.columns if x not in ['label']]
    ddev_predictions = alg.predict(ddev[dev_predictors])
    ddev_predprob = alg.predict_proba(ddev[dev_predictors])[:,1]
    
    # Test prediction
    test_predictors = [x for x in dtest.columns if x not in ['label']]
    dtest_predictions = alg.predict(dtest[test_predictors])
    dtest_predprob = alg.predict_proba(dtest[test_predictors])[:,1]
      
    # Print model report:
    print "\nModel Report"
    print("AUC    (Train/Test):\t%f\t%f\t%f" % (metrics.roc_auc_score(dtrain['label'], dtrain_predprob), metrics.roc_auc_score(ddev['label'], ddev_predprob), metrics.roc_auc_score(dtest['label'], dtest_predprob)))
    print("F1     (Train/Test):\t%f\t%f\t%f" % (metrics.f1_score(dtrain['label'], dtrain_predictions), metrics.f1_score(ddev['label'], ddev_predictions), metrics.f1_score(dtest['label'], dtest_predictions)))
    print("Kappa  (Train/Test):\t%f\t%f\t%f" % (metrics.cohen_kappa_score(dtrain['label'], dtrain_predictions), metrics.cohen_kappa_score(ddev['label'], ddev_predictions), metrics.cohen_kappa_score(dtest['label'], dtest_predictions)))


def main():
    
    path = './data/'
    
    # session_meta_map, session_message_map = import_data(path, filter_mark=True, print_mark=False)
    
    # Step 1: split train/dev/test
    # split_data(session_meta_map, session_message_map, path)
    
    # Step 2: generate data files
    # generate_frequent_ngrams(session_message_map, path)
    # generate_data_files(session_meta_map, session_message_map, path, last_mark=False)
    
    # Step 3: Prediction - Gradient Tree Boosting
    # [0.2, 0.4, 0.6, 0.8, 1.0]
    '''
    for percentage in [0.2, 0.4, 0.6, 0.8, 1.0]:
        print('Percengage is:\t%s' % str(percentage))
        build_prediction_model(path, percentage, para_tuning_mark=False, last_mark=True)
        print('')
    '''
    
    # Step 4: Alation test
    # alation_test(path)
    
    # Step 5: Random forest
    # parameter_tuning_random_forest(path)
    build_baseline_model_random_forest(path)
    


if __name__ == '__main__':
    main()