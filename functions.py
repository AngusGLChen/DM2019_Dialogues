'''
Created on 11 Mar 2019

@author: gche0022
'''

import pandas as pd
import datetime


def retrieve_metric_array(group, metri_key):
    metric_array = []
    for session_id in group.keys():
        metric_array.append(group[session_id][metri_key])
    return metric_array


def import_data(path, filter_mark, print_mark):
    '''
    (1) read data;
    (2) get basic statistics
    '''
    
    # Import session data
    session_data = pd.read_csv(path + 'datadump-20150801-20171219 (2).csv', low_memory=False, keep_default_na=False)
    session_data = session_data.drop_duplicates()
    if print_mark:
        print(session_data.head())
        print("")
        
    # Remove records with student_rating of None or 0
    session_data = session_data[session_data.student_rating != '']
    session_data = session_data[session_data.student_rating != '0']
    
    # Gather session metadata
    session_meta_map = dict()
    for index, row in session_data.iterrows():
        session_id = row['session_id']
        session_meta_map[session_id] = row
    
    # Import dialogue data
    dialogue_data = pd.read_csv(path + 'dialogue_mesages.csv', skiprows=range(1,28), low_memory=False)
    if print_mark:
        print(dialogue_data.head())
        print("")
    
    # Remove duplicated records and messages sent from system/bot
    dialogue_data = dialogue_data.drop_duplicates()
    for record_value in ['system info', 'system alert', 'system warn', 'bot']:
        dialogue_data = dialogue_data[dialogue_data.sent_from != record_value]
    for record_value in ['bot']:
        dialogue_data = dialogue_data[dialogue_data.sent_to != record_value]
        
    # Remove records with session_id = 0
    dialogue_data = dialogue_data[dialogue_data.session_id != 0]
    
    # Clean dialogue_data
    session_id_set = set(session_meta_map.keys())
    
    # Retrieve # utterances for each session -----------------------------------
    session_message_map = dict()
    considered_session_id_set = set()
    for index, row in dialogue_data.iterrows():
        session_id = row['session_id']
        role = row['sent_from']
        created_at = row['created_at']
        content_type = row['content_type']
        text = str(row['text'])
        
        created_at = datetime.datetime.strptime(created_at.replace(' UTC', ''), "%Y-%m-%d %H:%M:%S")
        
        if session_id in session_id_set:
            if session_id not in considered_session_id_set:
                considered_session_id_set.add(session_id)
                session_message_map[session_id] = {'total_num':0,
                                                   'tutor':0,
                                                   'student':0,
                                                   'time_role_message_array':[],
                                                   'duration':0}
                    
            session_message_map[session_id]['total_num'] += 1
            session_message_map[session_id][role] += 1
            session_message_map[session_id]['time_role_message_array'].append({'created_at':created_at,
                                                                               'content_type':content_type,
                                                                               'role':role,
                                                                               'text':text})
            
        # Testing
        # if index > 50000:
        #     break
        
    # Calculate session duration -----------------------------------------------
    for session_id in session_message_map.keys():    
        # Sorting time_role_message_array
        session_message_map[session_id]['time_role_message_array'] = sorted(session_message_map[session_id]['time_role_message_array'], key=lambda d: d['created_at'])
        
        # Calculate duration
        if len(session_message_map[session_id]['time_role_message_array']) > 1:
            session_message_map[session_id]['duration'] = (session_message_map[session_id]['time_role_message_array'][-1]['created_at'] - \
                                                           session_message_map[session_id]['time_role_message_array'][0]['created_at']).total_seconds() / float(60)
        else:
            session_message_map[session_id]['duration'] = 0
    
    # Remove sessions of length less than 10 turns or 1 min
    print('Before removing, # sessions:\t%d' % len(session_message_map))
    if filter_mark:
        session_message_map = {key:val for key, val in session_message_map.items() if val['total_num'] >= 10}
        session_message_map = {key:val for key, val in session_message_map.items() if val['duration'] >= 1}
    print('After removing, # sessions:\t%d' % len(session_message_map))
    
    # Print out basic statistics -----------------------------------------------
    num_sessions = len(session_message_map.keys())
    print('# total sessions:\t%d' % num_sessions)
    
    num_messages = 0
    for session_id in session_message_map.keys():
        num_messages += session_message_map[session_id]['total_num']
    print('# total utterances:\t%d' % num_messages)
    
    tutor_set = set()
    for session_id in session_message_map.keys():
        tutor_set.add(session_meta_map[session_id]['tutor_id'])
    print('# tutors:\t%d' % len(tutor_set))
    
    student_set = set()
    for session_id in session_message_map.keys():
        student_set.add(session_meta_map[session_id]['student_id'])
    print('# students:\t%d' % len(student_set))
    print('\n')
    
    return session_meta_map, session_message_map
