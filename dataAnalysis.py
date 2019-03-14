'''
Created on 1 Feb 2019

@author: gche0022
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import csv
import nltk
import json
import numpy
import matplotlib.pyplot as plt
from functions import import_data
from nltk.tokenize import word_tokenize
from collections import defaultdict

import matplotlib as mpl
mpl.rcParams['font.size'] = 15.0


def analyze_student_rating_distribution(session_meta_map, session_message_map):
    num_sessions = len(session_message_map.keys())
    value_array = [float(session_meta_map[session_id]['student_rating']) for session_id in session_message_map.keys()]
    print('# Avg. student rating:\t%.2f' % numpy.mean(value_array))
    
    # Pie chart
    start = 1
    end = 5
    step = 1
    labels = range(start, end+1, step)
    
    pie_nums = [0] * (len(labels))
    
    for record_value in value_array:
        index = int(record_value) / step - 1
        pie_nums[index] += 1
        
    sizes = [record_value/float(num_sessions)*100 for record_value in pie_nums] 
    colors = ['#ff9999', '#99ff99', '#ffcc99', '#ff6666', '#66b3ff']
    explode = (0.05,0.05,0.05,0.05,0.05)
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode)
    
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    
    title = 'Student ratings.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/'+title, bbox_inches='tight', pad_inches=0)
    
    plt.show()   


def analyze_dialogue_duration_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    value_array = [session_message_map[session_id]['duration'] for session_id in session_message_map.keys()]
    print('Avg. session duration (mins):\t%.2f' % numpy.mean(value_array))
    
    # Bar chart
    start = 0
    end = 60
    step = 5
    x_labels = range(start, end+1, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in value_array:
        index = int(record_value / step)
        if index >= len(bar_nums):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    x_pos = numpy.arange(len(x_labels)) + 1
    
    plt.figure(figsize=(12, 7.5))
    plt.bar(x_pos, bar_nums, width=0.8, linewidth=0.4, edgecolor='black', align='center', color='#5974A4')
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]-0.35, bar_nums[i]+0.30, ('%.1f' % bar_nums[i]) + '%')
    
    # x-axis labels
    x_labels.append("Max")
    updated_x_lables = []
    for i in range(1,len(x_labels)):
        updated_x_lables.append("(" + str(x_labels[i-1]) + "," + str(x_labels[i]) + "]")
    plt.xticks(x_pos, updated_x_lables, rotation=40)

    # y-axix percentage
    points = 0
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('Session duration (mins)')
    plt.ylabel('% Sessions')
    
    title = 'Session length (mins).png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()


def analyze_dialogue_num_utterances_distribution(session_message_map):
    value_array = [session_message_map[session_id]['total_num'] for session_id in session_message_map.keys()]
    print('Avg. # utterances / session:\t%.2f' % numpy.mean(value_array))
    
    
def analyze_dialogue_word_distribution(session_message_map):
    value_array = []
    for session_id in session_message_map.keys():
        word_cnt = 0
        for record in session_message_map[session_id]['time_role_message_array']:
            content_type = record['content_type']
            if content_type == 'text':
                text = record['text'].lower()
                try:
                    words = word_tokenize(text)
                    word_cnt += len(words)
                except:
                    pass
        value_array.append(word_cnt)
    print('Avg. # words / session:\t%.2f' % numpy.mean(value_array))
    
    
def analyze_dialogue_unique_word_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    value_array = []
    for session_id in session_message_map.keys():
        word_set = set()
        for record in session_message_map[session_id]['time_role_message_array']:
            content_type = record['content_type']
            if content_type == 'text':
                text = record['text'].lower()
                try:
                    words = word_tokenize(text)
                    word_set = word_set | set(words)
                except:
                    pass
        value_array.append(len(word_set))
    print('Avg. # unique words / session:\t%.2f' % numpy.mean(value_array))
    
    # Bar chart
    start = 0
    end = 300
    step = 25
    x_labels = range(start, end+1, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in value_array:
        index = int(record_value / step)
        if index >= len(bar_nums):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    x_pos = numpy.arange(len(x_labels)) + 1
    
    plt.figure(figsize=(12, 7.5))
    plt.bar(x_pos, bar_nums, width=0.8, linewidth=0.4, edgecolor='black', align='center', color='#5974A4')
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]-0.45, bar_nums[i]+0.25, ('%.1f' % bar_nums[i]) + '%')
    
    # x-axis labels
    x_labels.append("Max")
    updated_x_lables = []
    for i in range(1,len(x_labels)):
        updated_x_lables.append("(" + str(x_labels[i-1]) + "," + str(x_labels[i]) + "]")
    plt.xticks(x_pos, updated_x_lables, rotation=40)

    # y-axix percentage
    points = 0   
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('# Unique words')    
    plt.ylabel('% Sessions')
    
    title = 'Number of unique words.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    
    
def analyze_tutor_utterances_fraction_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    value_array = []
    for session_id in session_message_map.keys():
        fraction = session_message_map[session_id]['tutor'] / float(session_message_map[session_id]['total_num']) * 100
        value_array.append(fraction)
    print('Avg. fraction utterances sent by tutors:\t%.2f' % numpy.mean(value_array))
    
    # Bar plot
    start = 0
    end = 100
    step = 10
    x_labels = range(start, end, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in value_array:
        index = int(record_value) / step
        if index >= len(x_labels):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    
    # Transform bar_nums by aggregation
    updated_bar_nums = [0] * 5
    for i in range(4):
        updated_bar_nums[0] += bar_nums[i]
    for i in range(4,7):
        updated_bar_nums[i-3] = bar_nums[i]
    for i in range(7, len(bar_nums)):
        updated_bar_nums[-1] += bar_nums[i]
    
    x_pos = numpy.arange(len(updated_bar_nums))
    
    plt.figure(figsize=(12, 7.5))
    plt.bar(x_pos, updated_bar_nums, width=0.8, linewidth=0.4, edgecolor='black', align='center', color='#5974A4')
    for i in range(len(updated_bar_nums)):
        plt.text(x_pos[i]-0.15, updated_bar_nums[i]+0.30, ('%.1f' % updated_bar_nums[i]) + '%')
    
    # x-axis labels
    x_labels = ["(0,40%]", "(40%,50%]", "(50%,60%]", "(60%,70%]", "(70%,100%]"]
    plt.xticks(x_pos, x_labels)

    # y-axix percentage
    points = 0
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('% Utterances sent by tutors')
    plt.ylabel('% Sessions')
    
    title = 'Fraction of utterances sent by tutors.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()


def analyze_tutor_words_fraction_distribution(session_message_map):
    value_array = []
    for session_id in session_message_map.keys():
        word_cnt = 0
        tutor_word_cnt = 0
        for record in session_message_map[session_id]['time_role_message_array']:
            content_type = record['content_type']
            if content_type == 'text':
                text = record['text'].lower()
                try:
                    words = word_tokenize(text)
                    word_cnt += len(words)
                    if record['role'] == 'tutor':
                        tutor_word_cnt += len(words)
                except:
                    pass
        if word_cnt != 0:
            value_array.append(tutor_word_cnt / float(word_cnt) * 100)
        else:
            value_array.append(0)
    print('Avg. fraction words sent by tutors:\t%.2f' % numpy.mean(value_array))
    

def analyze_tutor_new_words_fraction_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    value_array = []
    for session_id in session_message_map.keys():
        word_set = set()
        tutor_word_set = set()
        for record in session_message_map[session_id]['time_role_message_array']:
            content_type = record['content_type']
            if content_type == 'text':
                text = record['text'].lower()
                try:
                    words = word_tokenize(text)
                    if record['role'] == 'tutor':
                        tutor_word_set = tutor_word_set | (set(words) - word_set)
                    word_set = word_set | set(words)
                except:
                    pass
                
        if len(word_set) != 0:
            value_array.append(len(tutor_word_set) / float(len(word_set)) * 100)
        else:
            value_array.append(0)
    print('Avg. fraction new words sent by tutors:\t%.2f' % numpy.mean(value_array))
    
    # Bar plot
    start = 0
    end = 100
    step = 10
    x_labels = range(start, end, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in value_array:
        index = int(record_value) / step
        if index >= len(x_labels):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    
    #  Transform bar_nums by aggregation
    updated_bar_nums = [0] * 5
    for i in range(6):
        updated_bar_nums[0] += bar_nums[i]
    for i in range(6,len(bar_nums)):
        updated_bar_nums[i-5] = bar_nums[i]
    
    x_pos = numpy.arange(len(updated_bar_nums))
    
    plt.figure(figsize=(12, 7.5))
    plt.bar(x_pos, updated_bar_nums, width=0.8, linewidth=0.4, edgecolor='black', align='center', color='#5974A4')
    
    for i in range(len(updated_bar_nums)):
        plt.text(x_pos[i]-0.15, updated_bar_nums[i]+0.30, ('%.1f' % updated_bar_nums[i]) + '%')
    
    # x-axis labels
    x_labels = ["(0,60%]", "(60%,70%]", "(70%,80%]", "(80%,90%]", "(90%,100%]"]
    plt.xticks(x_pos, x_labels)
    
    # y-axix percentage
    points = 0
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('Avg. % new words sent by tutors')
    plt.ylabel('% Sessions')
    
    title = 'Fraction new words sent by tutors.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    
     
def analyze_tutor_platform_exeperience_distribution(session_meta_map, session_message_map, role_type):
    role_session_map = dict()
    role_set = set()

    for session_id in session_message_map.keys():
        if role_type == 'tutor':
            role = session_meta_map[session_id]['tutor_id']
        if role_type == 'student':
            role = session_meta_map[session_id]['student_id']
        
        if role not in role_set:
            role_set.add(role)
            role_session_map[role] = set()
        role_session_map[role].add(session_id)
    
    value_array = []
    for role in role_session_map.keys():
        value_array.append(len(role_session_map[role]))
    
    print('Avg. # sessions that a tutor/student had:\t%.2f' % numpy.mean(value_array))

 
def retrieve_dialogue_examples(session_meta_map, session_message_map, num_examples):
    cnt = 0
    for session_id in session_message_map.keys():
        if session_meta_map[session_id]['student_rating'] in ['1', '2']:
            for record in session_message_map[session_id]['time_role_message_array']:
                role = record['role']
                text = record['text']
                print('%s\t%s' % (role, text))
            print('**************************************************')
            cnt += 1
            if cnt > num_examples:
                break
            
            
def generate_noun_frequency_map(session_message_map, path):
    noun_frequency_map = defaultdict(int)
    for session_id in session_message_map.keys():
        for record in session_message_map[session_id]['time_role_message_array']:
            if record['content_type'] == 'text':
                text = record['text']
                try:
                    is_noun = lambda pos: pos[:2] == 'NN'
                    tokenized_text = word_tokenize(text)
                    nouns = [word for (word, pos) in nltk.pos_tag(tokenized_text) if is_noun(pos)]
                    for noun in nouns:
                        noun_frequency_map[noun.lower().encode("utf-8")] += 1
                except Exception as e:
                    # print(e)
                    pass
                    
    outfile = open(path + 'noun_frequency_map', 'w')
    outfile.write(json.dumps(noun_frequency_map, ensure_ascii=False))
    outfile.close()


def generate_word_frequency_map(session_message_map, path):
    word_frequency_map = defaultdict(int)
    for session_id in session_message_map.keys():
        for record in session_message_map[session_id]['time_role_message_array']:
            if record['content_type'] == 'text':
                text = record['text'].lower()
                try:
                    words = word_tokenize(text)
                    for word in words:
                        word_frequency_map[word.encode("utf-8")] += 1
                except:
                    pass
    outfile = open(path + 'word_frequency_map', 'w')
    outfile.write(json.dumps(word_frequency_map, ensure_ascii=False))
    outfile.close()


def main():
    
    path = './data/'
    
    # 1. Read data
    session_meta_map, session_message_map = import_data(path, filter_mark=True, print_mark=False)
    
    # 2.1 Statistics: Ratings
    # analyze_student_rating_distribution(session_meta_map, session_message_map)
    
    # 2.2 Statistics: Dialogue length
    # analyze_dialogue_duration_distribution(session_message_map)
    # analyze_dialogue_num_utterances_distribution(session_message_map)
    # analyze_dialogue_word_distribution(session_message_map)
    # analyze_dialogue_unique_word_distribution(session_message_map)
    
    # 2.3 Statistics: Activeness
    # analyze_tutor_utterances_fraction_distribution(session_message_map)
    # analyze_tutor_words_fraction_distribution(session_message_map)
    # analyze_tutor_new_words_fraction_distribution(session_message_map)
    
    # 2.4 Statistics: Platform experience
    # analyze_tutor_platform_exeperience_distribution(session_meta_map, session_message_map, 'tutor')
    # analyze_tutor_platform_exeperience_distribution(session_meta_map, session_message_map, 'student')
    
    # 3. Check dialogue examples
    # retrieve_dialogue_examples(session_meta_map, session_message_map)
    
    # 4. Generate frequent words/nouns
    # generate_word_frequency_map(session_message_map, path)
    generate_noun_frequency_map(session_message_map, path)
    
    
if __name__ == "__main__":
    main()