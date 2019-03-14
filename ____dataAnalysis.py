'''
Created on 1 Feb 2019

@author: gche0022
'''


from functions import import_data




import numpy
import datetime
import time
import nltk
from nltk.tokenize import WordPunctTokenizer

import csv
import pandas as pd
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt


import scipy.stats
import matplotlib as mpl
mpl.rcParams['font.size'] = 15.0






def measure_total_num_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    
    total_num_array = []
    for session_id in session_message_map.keys():
        total_num_array.append(session_message_map[session_id]['total_num'])
    
    # Get basic statistics
    print('Avg. # utterances / session:\t%.2f+/-%.2f' % (numpy.mean(total_num_array), numpy.std(total_num_array)))
    step = 25
    start = 0
    end = 200
    x_labels = range(start, end+1, step)
    bar_nums = [0] * (len(x_labels))
    
    # print(x_labels)
    # print(bar_nums)
    
    for record_value in total_num_array:
        index = record_value / step
        if index >= len(bar_nums):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    # print(bar_nums)
    
    x_pos = numpy.arange(len(x_labels)) + 1
    # print(x_pos)
    
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, bar_nums, width=1, linewidth=0.4, edgecolor='black', align='edge', alpha=0.5)
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]+0.35, bar_nums[i] + 0.45, ('%.0f' % bar_nums[i]) + '%')
    
    points = 0    
    
    '''
    updated_x_lables = ["[1," + str(x_labels[1]) + ")"]
    for i in range(1,len(x_labels)-1):
        updated_x_lables.append("[" + str(x_labels[i]) + "," + str(x_labels[i+1]) + ")")
    updated_x_lables.append("[" + str(x_labels[-1]) + ",+" + float(inf) + ")")
    '''
    
    x_labels.append("Max")
    x_pos = numpy.append(x_pos, x_pos[-1]+1)
    plt.xticks(x_pos, x_labels)

    # plt.xlim([0-0.25, x_pos.size-0.75])
    
    # Y axix percentage
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('# Utterances')
    plt.ylabel('% Sessions')
    
    title = 'Number of utterances in sessions.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()

    
def measure_tutor_message_ratio_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    
    tutor_ratio_array = []
    
    # Calculate ration
    for session_id in session_message_map.keys():
        session_message_map[session_id]['tutor_ratio'] = session_message_map[session_id]['tutor'] / \
                                                         float(session_message_map[session_id]['total_num']) * 100
        tutor_ratio_array.append(session_message_map[session_id]['tutor_ratio'])
    
    # Get basic statistics
    print('Avg. fraction utterances sent by tutors:\t%.2f+/-%.2f' % (numpy.mean(tutor_ratio_array), numpy.std(tutor_ratio_array)))
    
    step = 10
    start = 0
    end = 100
    x_labels = range(start, end, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in tutor_ratio_array:
        index = int(record_value) / step
        if index >= len(x_labels):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    
    updated_bar_nums = [0] * 5
    for i in range(4):
        updated_bar_nums[0] += bar_nums[i]
    for i in range(4,7):
        updated_bar_nums[i-3] = bar_nums[i]
    for i in range(7, len(bar_nums)):
        updated_bar_nums[-1] += bar_nums[i]
    
    x_pos = numpy.arange(len(updated_bar_nums))
    
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, updated_bar_nums, width=0.8, linewidth=0.4, edgecolor='black', align='center', color='#5974A4')
    
    for i in range(len(updated_bar_nums)):
        plt.text(x_pos[i]-0.15, updated_bar_nums[i]+0.30, ('%.1f' % updated_bar_nums[i]) + '%')
    
    x_labels = ["(0,40%]", "(40%,50%]", "(50%,60%]", "(60%,70%]", "(70%,100%]"]
    plt.xticks(x_pos, x_labels)

    # Y axix percentage
    points = 0
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('% Utterances sent by tutors')
    plt.ylabel('% Sessions')
    
    title = 'Fraction of utterances sent by tutors.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    
    
    '''
    # Pie chart
    step = 10
    start = 0
    end = 100
    labels = range(start, end, step)
    pie_nums = [0] * (len(labels))
    
    for record_value in tutor_ratio_array:
        index = int(record_value) / step
        if index >= len(labels):
            pie_nums[-1] += 1
        else:
            pie_nums[index] += 1
        
    # Normalization
    sizes = [record_value/float(num_sessions)*100 for record_value in pie_nums]
    
    updated_sizes = [0] * 5
    for i in range(4):
        updated_sizes[0] += sizes[i]
    for i in range(4,7):
        updated_sizes[i-3] = sizes[i]
    for i in range(7, len(sizes)):
        updated_sizes[-1] += sizes[i]
    
    print(updated_sizes)
    
    #colors
    colors = ['#ff9999', '#ff6666', '#ffcc99', '#99ff99', '#66b3ff']
    
    fig1, ax1 = plt.subplots()
    
    #explsion
    explode = (0.05,0.05,0.05,0.05,0.05)
    
    labels = ["(0,40%]", "(40%,50%]", "(50%,60%]", "(60%,70%]", "(70%,100%]"]
    
    ax1.pie(updated_sizes[::-1], colors=colors, labels=labels[::-1], autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
    
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    
    title = 'Fraction of utterances sent by tutors.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    '''
    
    
def measure_duration_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    
    duration_array = []
    
    for session_id in session_message_map.keys():
        duration_array.append(session_message_map[session_id]['duration'])

    # Get basic statistics
    # print('Min/Max value are:\t%d\t%d' % (numpy.min(duration_array), numpy.max(duration_array)))
    print('Avg. session length (mins):\t%.2f+/-%.2f' % (numpy.mean(duration_array), numpy.std(duration_array)))
    
    
    step = 5
    start = 0
    end = 60
    x_labels = range(start, end+1, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in duration_array:
        index = int(record_value / step)
        if index >= len(bar_nums):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    
    x_pos = numpy.arange(len(x_labels)) + 1
    
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, bar_nums, width=0.8, linewidth=0.4, edgecolor='black', align='center', color='#5974A4')
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]-0.35, bar_nums[i]+0.30, ('%.1f' % bar_nums[i]) + '%')
    
    points = 0    
    
    x_labels.append("Max")
    
    updated_x_lables = []
    for i in range(1,len(x_labels)):
        updated_x_lables.append("(" + str(x_labels[i-1]) + "," + str(x_labels[i]) + "]")
    
    plt.xticks(x_pos, updated_x_lables, rotation=40)

    # Y axix percentage
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('Session duration (mins)')
    plt.ylabel('% Sessions')
    
    title = 'Session length (mins).png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    

def measure_word_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    
    word_array = []
    
    for session_id in session_message_map.keys():
        word_cnt = 0
        for record in session_message_map[session_id]['time_role_message_array']:
            content_type = record['content_type']
            if content_type == 'text':
                text = record['text'].lower()
                words = WordPunctTokenizer().tokenize(text)
                word_cnt += len(words)
        word_array.append(word_cnt)
    
    # Get basic statistics
    # print('Min/Max word value are:\t%d\t%d' % (numpy.min(word_array), numpy.max(word_array)))
    print('Avg. # words / session:\t%.2f+/-%.2f' % (numpy.mean(word_array), numpy.std(word_array)))
    
    step = 100
    start = 0
    end = 1000
    x_labels = range(start, end+1, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in word_array:
        index = int(record_value / step)
        if index >= len(bar_nums):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    
    x_pos = numpy.arange(len(x_labels)) + 1
    
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, bar_nums, width=1, linewidth=0.4, edgecolor='black', align='edge', alpha=0.5)
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]+0.20, bar_nums[i] + 0.35, ('%.2f' % bar_nums[i]) + '%')
    
    points = 0    
    
    x_labels.append("Max")
    x_pos = numpy.append(x_pos, x_pos[-1]+1)
    plt.xticks(x_pos, x_labels)

    # Y axix percentage
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('# Words')    
    plt.ylabel('% Sessions')
    
    title = 'Number of words.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()


def measure_unique_word_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    
    word_array = []
    
    for session_id in session_message_map.keys():
        word_cnt = 0
        word_set = set()
        for record in session_message_map[session_id]['time_role_message_array']:
            content_type = record['content_type']
            if content_type == 'text':
                text = record['text'].lower()
                words = WordPunctTokenizer().tokenize(text)
                for word in words:
                    word_set.add(word)
        word_array.append(len(word_set))
    
    # Get basic statistics
    # print('Min/Max word value are:\t%d\t%d' % (numpy.min(word_array), numpy.max(word_array)))
    print('Avg. # unique words / session:\t%.2f+/-%.2f' % (numpy.mean(word_array), numpy.std(word_array)))
    
    step = 25
    start = 0
    end = 300
    x_labels = range(start, end+1, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in word_array:
        index = int(record_value / step)
        if index >= len(bar_nums):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    
    x_pos = numpy.arange(len(x_labels)) + 1
    
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, bar_nums, width=0.8, linewidth=0.4, edgecolor='black', align='center', color='#5974A4')
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]-0.45, bar_nums[i]+0.25, ('%.1f' % bar_nums[i]) + '%')
    
    points = 0   
    
    x_labels.append("Max")
    updated_x_lables = []
    for i in range(1,len(x_labels)):
        updated_x_lables.append("(" + str(x_labels[i-1]) + "," + str(x_labels[i]) + "]")
    plt.xticks(x_pos, updated_x_lables, rotation=40)

    # Y axix percentage
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('# Unique words')    
    plt.ylabel('% Sessions')
    
    title = 'Number of unique words.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()


def measure_tutor_word_ratio_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    
    tutor_ratio_array = []
    
    # Calculate ration
    for session_id in session_message_map.keys():
        word_cnt = 0
        tutor_word_cnt = 0
        
        for record in session_message_map[session_id]['time_role_message_array']:
            content_type = record['content_type']
            if content_type == 'text':
                text = record['text'].lower()
                words = WordPunctTokenizer().tokenize(text)
                word_cnt += len(words)
                
                if record['role'] == 'tutor':
                    tutor_word_cnt += len(words)
        
        if word_cnt != 0:
            tutor_ratio_array.append(tutor_word_cnt / float(word_cnt) * 100)
        else:
            tutor_ratio_array.append(0)
    
    # Get basic statistics
    print('Avg. fraction words sent by tutors:\t%.2f+/-%.2f' % (numpy.mean(tutor_ratio_array), numpy.std(tutor_ratio_array)))
    
    step = 10
    start = 0
    end = 100
    x_labels = range(start, end, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in tutor_ratio_array:
        index = int(record_value) / step
        if index >= len(x_labels):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    x_pos = numpy.arange(len(x_labels))
   
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, bar_nums, width=1, linewidth=0.4, edgecolor='black', align='edge', alpha=0.5)
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]+0.35, bar_nums[i] + 0.5, ('%.0f' % bar_nums[i]) + '%')
    
    x_labels.append(str(end))
    x_pos = numpy.append(x_pos, x_pos[-1]+1)
    plt.xticks(x_pos, x_labels)
    
    # Y axix percentage
    points = 0
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    plt.gca().set_xticklabels([('{:.' + str(points) + 'f}%').format(x*10) for x in plt.gca().get_xticks()])
    
    plt.xlabel('% Words sent by tutors')
    plt.ylabel('% Sessions')
    
    title = 'Fraction words sent by tutors.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()


# def measure_tutor_unique_word_ratio_distribution(session_message_map):
#     num_sessions = len(session_message_map.keys())
#     
#     tutor_ratio_array = []
#     
#     # Calculate ration
#     for session_id in session_message_map.keys():
#         word_set = set()
#         tutor_word_set = set()
#         
#         for record in session_message_map[session_id]['time_role_message_array']:
#             content_type = record['content_type']
#             if content_type == 'text':
#                 text = record['text'].lower()
#                 words = WordPunctTokenizer().tokenize(text)
#                 for word in words:
#                     word_set.add(word)
#                 
#                 if record['role'] == 'tutor':
#                     for word in words:
#                         tutor_word_set.add(word)
#         
#         if len(word_set) != 0:
#             tutor_ratio_array.append(len(tutor_word_set) / float(len(word_set)) * 100)
#         else:
#             tutor_ratio_array.append(0)
#     
#     # Get basic statistics
#     print('Avg. fraction unique words sent by tutors:\t%.2f+/-%.2f' % (numpy.mean(tutor_ratio_array), numpy.std(tutor_ratio_array)))
#     
#     step = 10
#     start = 0
#     end = 100
#     x_labels = range(start, end, step)
#     bar_nums = [0] * (len(x_labels))
#     
#     for record_value in tutor_ratio_array:
#         index = int(record_value) / step
#         if index >= len(x_labels):
#             bar_nums[-1] += 1
#         else:
#             bar_nums[index] += 1
#     
#     # Normalization
#     bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
#     x_pos = numpy.arange(len(x_labels))
#    
#     fig = plt.figure(figsize=(12, 7.5))
#     barlist = plt.bar(x_pos, bar_nums, width=1, linewidth=0.4, edgecolor='black', align='edge', alpha=0.5)
#     
#     for i in range(len(bar_nums)):
#         plt.text(x_pos[i]+0.35, bar_nums[i] + 0.5, ('%.0f' % bar_nums[i]) + '%')
#     
#     x_labels.append(str(end))
#     x_pos = numpy.append(x_pos, x_pos[-1]+1)
#     plt.xticks(x_pos, x_labels)
#     
#     # Y axix percentage
#     points = 0
#     plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
#     
#     plt.xlabel('% Unique words sent by tutors')
#     plt.ylabel('% Sessions')
#     
#     title = 'Fraction unique words sent by tutors.png'
#     title = title.replace(" ", "_")
#     plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
#     
#     plt.show()


# def measure_tutor_score_distribution(session_meta_map, session_message_map):
#     num_sessions = len(session_message_map.keys())
#     
#     score_array = []
#     num_missing = 0
#     
#     threshold = 0.8
#     num_threshold = 0
#     
#     # Calculate ration
#     for session_id in session_message_map.keys():
#         if session_meta_map[session_id]['feedback_score'] == '':
#             num_missing += 1
#         else:
#             feedback_score = float(session_meta_map[session_id]['feedback_score'])
#             score_array.append(feedback_score)
#             if feedback_score >= threshold:
#                 num_threshold += 1
#     
#     # Get basic statistics
#     print("Missing records:\t%d\t%.2f" % (num_missing, float(num_missing)/num_sessions*100))
#     print("Tutor scores >= %.1f:\t%d\t%.2f" % (threshold, num_threshold, float(num_threshold)/(num_sessions-num_missing)*100))
#     print('# Avg. tutor score:\t%.2f+/-%.2f' % (numpy.mean(score_array), numpy.std(score_array)))
#     
#     step = 0.1
#     start = 0
#     end = 1
#     x_labels = [v*step for v in range(start,int(end/step)+1)]
#     bar_nums = [0] * (len(x_labels))
#     
#     for record_value in score_array:
#         index = int(record_value / step)
#         bar_nums[index] += 1
#     
#     # Normalization
#     bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
#     x_pos = numpy.arange(len(x_labels))
#    
#     fig = plt.figure(figsize=(12, 7.5))
#     barlist = plt.bar(x_pos, bar_nums, width=1, linewidth=0.4, edgecolor='black', align='center', alpha=0.5)
#     
#     for i in range(len(bar_nums)):
#         plt.text(x_pos[i]-0.05, bar_nums[i] + 0.5, ('%.0f' % bar_nums[i]) + '%')
#     
#     plt.xticks(x_pos, x_labels)
#     
#     # Y axix percentage
#     points = 0
#     plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
#     
#     plt.xlabel('Tutor score')
#     plt.ylabel('% Sessions')
#     
#     title = 'Tutor ratings.png'
#     title = title.replace(" ", "_")
#     plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
#     
#     plt.show()
#     


def measure_student_score_distribution(session_meta_map, session_message_map):
    num_sessions = len(session_message_map.keys())
    
    score_array = []
    num_missing = 0
    
    threshold = 4
    num_threshold = 0
    
    # Calculate ratio
    for session_id in session_message_map.keys():
        if session_meta_map[session_id]['student_rating'] == '':
            num_missing += 1
        else:
            student_rating = float(session_meta_map[session_id]['student_rating'])
            score_array.append(student_rating)
            if student_rating >= threshold:
                num_threshold += 1
    
    # Get basic statistics
    print("Missing records:\t%d\t%.2f" % (num_missing, float(num_missing)/num_sessions*100))
    print("Tutor scores >= %d:\t%d\t%.2f" % (threshold, num_threshold, float(num_threshold)/(num_sessions-num_missing)*100))
    print('# Avg. student score:\t%.2f+/-%.2f' % (numpy.mean(score_array), numpy.std(score_array)))
    
    '''
    # Bar chart
    step = 1
    start = 0
    end = 5
    x_labels = range(start, end, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in score_array:
        index = int(record_value) / step - 1
        bar_nums[index] += 1
        
    for i in range(len(bar_nums)):
        print("%d\t%d\t%.2f" % (i+1, bar_nums[i], bar_nums[i]/float(num_sessions)*100))
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    x_pos = numpy.arange(len(x_labels))
   
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, bar_nums, width=1, linewidth=0.4, edgecolor='black', align='center', alpha=0.5)
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]-0.05, bar_nums[i] + 0.7, ('%.0f' % bar_nums[i]) + '%')
    
    plt.xticks(x_pos, x_labels)
    
    # Y axix percentage
    points = 0
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    plt.gca().set_xticklabels([('{:.' + str(points) + 'f}').format(x+1) for x in plt.gca().get_xticks()])
    
    plt.xlabel('Student ratings')
    plt.ylabel('% Sessions')
    
    title = 'Student ratings.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    '''
    
    # Pie chart
    start = 1
    end = 5
    step = 1
    labels = range(start, end+1, step)
    
    pie_nums = [0] * (len(labels))
    
    for record_value in score_array:
        index = int(record_value) / step - 1
        pie_nums[index] += 1
        
    sizes = [record_value/float(num_sessions)*100 for record_value in pie_nums]
    
    #colors
    colors = ['#ff9999', '#99ff99', '#ffcc99', '#ff6666', '#66b3ff']
    
    fig1, ax1 = plt.subplots()
    
    #explsion
    explode = (0.05,0.05,0.05,0.05,0.05)
     
    ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
    
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    
    title = 'Student ratings.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    
    

def measure_correlation_between_tutor_student_scores(session_meta_map, session_message_map):
    tutor_score_array = []
    student_score_array = []
    
    for session_id in session_message_map.keys():
        tutor_score = float(session_meta_map[session_id]['feedback_score'])
        student_score = float(session_meta_map[session_id]['student_rating'])
        tutor_score_array.append(tutor_score)
        student_score_array.append(student_score)
    
    print(scipy.stats.pearsonr(tutor_score_array, student_score_array))
    print(scipy.stats.spearmanr(tutor_score_array, student_score_array))
    

def measure_tutor_new_word_ratio_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    
    tutor_ratio_array = []
    
    # Calculate ration
    for session_id in session_message_map.keys():
        word_set = set()
        tutor_word_set = set()
        
        for record in session_message_map[session_id]['time_role_message_array']:
            content_type = record['content_type']
            if content_type == 'text':
                text = record['text'].lower()
                words = WordPunctTokenizer().tokenize(text)
                
                if record['role'] == 'tutor':
                    for word in words:
                        if word not in word_set:
                            tutor_word_set.add(word)
                
                for word in words:
                    word_set.add(word)
                
        if len(word_set) != 0:
            tutor_ratio_array.append(len(tutor_word_set) / float(len(word_set)) * 100)
        else:
            tutor_ratio_array.append(0)
    
    # Get basic statistics
    print('Avg. fraction new words sent by tutors:\t%.2f+/-%.2f' % (numpy.mean(tutor_ratio_array), numpy.std(tutor_ratio_array)))
    

    step = 10
    start = 0
    end = 100
    x_labels = range(start, end, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in tutor_ratio_array:
        index = int(record_value) / step
        if index >= len(x_labels):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    
    updated_bar_nums = [0] * 5
    for i in range(6):
        updated_bar_nums[0] += bar_nums[i]
    for i in range(6,len(bar_nums)):
        updated_bar_nums[i-5] = bar_nums[i]
    
    x_pos = numpy.arange(len(updated_bar_nums))
    
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, updated_bar_nums, width=0.8, linewidth=0.4, edgecolor='black', align='center', color='#5974A4')
    
    for i in range(len(updated_bar_nums)):
        plt.text(x_pos[i]-0.15, updated_bar_nums[i]+0.30, ('%.1f' % updated_bar_nums[i]) + '%')
    
    x_labels = ["(0,60%]", "(60%,70%]", "(70%,80%]", "(80%,90%]", "(90%,100%]"]
    plt.xticks(x_pos, x_labels)
    
    # Y axix percentage
    points = 0
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('Avg. % new words sent by tutors')
    plt.ylabel('% Sessions')
    
    title = 'Fraction new words sent by tutors.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    
    
    '''
    # Pie chart
    step = 10
    start = 0
    end = 100
    labels = range(start, end, step)
    pie_nums = [0] * (len(labels))
    
    for record_value in tutor_ratio_array:
        index = int(record_value) / step
        if index >= len(labels):
            pie_nums[-1] += 1
        else:
            pie_nums[index] += 1
        
    # Normalization
    sizes = [record_value/float(num_sessions)*100 for record_value in pie_nums]
    
    updated_sizes = [0] * 5
    for i in range(6):
        updated_sizes[0] += sizes[i]
    for i in range(6,len(sizes)):
        updated_sizes[i-5] = sizes[i]
    
    print(updated_sizes)
    
    #colors
    colors = ['#ff9999', '#ff6666', '#ffcc99', '#99ff99', '#66b3ff']
    
    fig1, ax1 = plt.subplots()
    
    #explsion
    explode = (0.05,0.05,0.05,0.05,0.05)
    
    labels = ["(0,40%]", "(40%,50%]", "(50%,60%]", "(60%,70%]", "(70%,100%]"]
    
    ax1.pie(updated_sizes[::-1], colors=colors, labels=labels[::-1], autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
    
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    
    title = 'Fraction new words sent by tutors.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    '''
    
    
def measure_student_new_word_ratio_distribution(session_message_map):
    num_sessions = len(session_message_map.keys())
    
    student_ratio_array = []
    
    # Calculate ration
    for session_id in session_message_map.keys():
        word_set = set()
        student_word_set = set()
        
        for record in session_message_map[session_id]['time_role_message_array']:
            content_type = record['content_type']
            if content_type == 'text':
                text = record['text'].lower()
                words = WordPunctTokenizer().tokenize(text)
                
                if record['role'] == 'student':
                    for word in words:
                        if word not in word_set:
                            student_word_set.add(word)
                
                for word in words:
                    word_set.add(word)
                
        if len(word_set) != 0:
            student_ratio_array.append(len(student_word_set) / float(len(word_set)) * 100)
        else:
            student_ratio_array.append(0)
    
    # Get basic statistics
    print('Avg. fraction new words sent by tutors:\t%.2f+/-%.2f' % (numpy.mean(student_ratio_array), numpy.std(student_ratio_array)))
    
    step = 10
    start = 0
    end = 100
    x_labels = range(start, end, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in student_ratio_array:
        index = int(record_value) / step
        if index >= len(x_labels):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(num_sessions)*100 for record_value in bar_nums]
    x_pos = numpy.arange(len(x_labels))
   
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, bar_nums, width=1, linewidth=0.4, edgecolor='black', align='edge', alpha=0.5)
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]+0.35, bar_nums[i] + 0.5, ('%.0f' % bar_nums[i]) + '%')
    
    x_labels.append(str(end))
    x_pos = numpy.append(x_pos, x_pos[-1]+1)
    plt.xticks(x_pos, x_labels)
    
    # Y axix percentage
    points = 0
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('Avg. % new words sent by students')
    plt.ylabel('% Sessions')
    
    title = 'Fraction new words sent by students.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()



def measure_tutor_familiarity_distribution(session_meta_map, session_message_map):

    tutor_session_map = dict()
    tutor_set = set()

    for session_id in session_message_map.keys():
        tutor = session_meta_map[session_id]['tutor_id']
        if tutor not in tutor_set:
            tutor_set.add(tutor)
            tutor_session_map[tutor] = set()
        tutor_session_map[tutor].add(session_id)
        
    familarity_array = []
    for tutor in tutor_session_map.keys():
        familarity_array.append(len(tutor_session_map[tutor])) 
    
    # Get basic statistics
    # print('Min/Max value are:\t%d\t%d' % (numpy.min(familarity_array), numpy.max(familarity_array)))
    print('Avg. # sessions guide by tutors:\t%.2f+/-%.2f' % (numpy.mean(familarity_array), numpy.std(familarity_array)))
    
    step = 50
    start = 0
    end = 500
    x_labels = range(start, end+1, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in familarity_array:
        index = int(record_value / step)
        if index >= len(bar_nums):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(len(tutor_set))*100 for record_value in bar_nums]
    
    x_pos = numpy.arange(len(x_labels)) + 1
    
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, bar_nums, width=1, linewidth=0.4, edgecolor='black', align='edge', alpha=0.5)
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]+0.35, bar_nums[i] + 0.5, ('%.0f' % bar_nums[i]) + '%')
    
    points = 0    
    
    x_labels.append("Max")
    x_pos = numpy.append(x_pos, x_pos[-1]+1)
    plt.xticks(x_pos, x_labels)

    # Y axix percentage
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('# Sessions guided by tutors')    
    plt.ylabel('% Tutors')
    
    title = 'Number of sessions guided by tutors.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()


def measure_student_familiarity_distribution(session_meta_map, session_message_map):
    student_session_map = dict()
    student_set = set()

    for session_id in session_message_map.keys():
        student = session_meta_map[session_id]['student_id']
        if student not in student_set:
            student_set.add(student)
            student_session_map[student] = set()
        student_session_map[student].add(session_id)
        
    familarity_array = []
    for student in student_session_map.keys():
        familarity_array.append(len(student_session_map[student])) 
    
    # Get basic statistics
    print('Min/Max value are:\t%d\t%d' % (numpy.min(familarity_array), numpy.max(familarity_array)))
    print('Avg. # sessions owned by students:\t%.2f+/-%.2f' % (numpy.mean(familarity_array), numpy.std(familarity_array)))
    
    step = 1
    start = 1
    end = 10
    x_labels = range(start, end+1, step)
    bar_nums = [0] * (len(x_labels))
    
    for record_value in familarity_array:
        index = int(record_value / step) - 1
        if index >= len(bar_nums):
            bar_nums[-1] += 1
        else:
            bar_nums[index] += 1
    
    # Normalization
    bar_nums = [record_value/float(len(student_set))*100 for record_value in bar_nums]
    
    x_pos = numpy.arange(len(x_labels)) + 1
    
    fig = plt.figure(figsize=(12, 7.5))
    barlist = plt.bar(x_pos, bar_nums, width=1, linewidth=0.4, edgecolor='black', align='center', alpha=0.5)
    
    for i in range(len(bar_nums)):
        plt.text(x_pos[i]-0.10, bar_nums[i] + 0.5, ('%.0f' % bar_nums[i]) + '%')
    
    points = 0    
    
    x_labels[-1] = ">=" + str(x_labels[-1])
    plt.xticks(x_pos, x_labels)

    # Y axix percentage
    plt.gca().set_yticklabels([('{:.' + str(points) + 'f}%').format(x) for x in plt.gca().get_yticks()])
    
    plt.xlabel('Number of sessions owned by students')
    plt.ylabel('% Students')
    
    title = 'Number of sessions owned by students.png'
    title = title.replace(" ", "_")
    plt.savefig('./data/' + title, bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    
    
def check_dialogues(session_meta_map, session_message_map):
    cnt = 0
    for session_id in session_message_map.keys():
        if session_meta_map[session_id]['student_rating'] in ['1', '2']:
            for record in session_message_map[session_id]['time_role_message_array']:
                role = record['role']
                text = record['text']
                
                print('%s\t%s' % (role, text))
                
            print('-------------\n\n')
            
            cnt += 1
            if cnt > 100:
                break
            
            

    

def main():
    
    path = './data/'
    
    # 1. Read data
    session_meta_map, session_message_map = import_data(path, filter_mark=True, print_mark=False)
    
    # General statistics distribution
    # measure_duration_distribution(session_message_map)
    # measure_total_num_distribution(session_message_map)
    # measure_word_distribution(session_message_map)
    # measure_unique_word_distribution(session_message_map)
    
    # measure_tutor_message_ratio_distribution(session_message_map)
    # measure_tutor_word_ratio_distribution(session_message_map)
    # measure_tutor_new_word_ratio_distribution(session_message_map)
    
    # Student score distribution
    # measure_student_score_distribution(session_meta_map, session_message_map)
    
    # Familarity
    # measure_tutor_familiarity_distribution(session_meta_map, session_message_map)
    # measure_student_familiarity_distribution(session_meta_map, session_message_map)
    
    # check_dialogues(session_meta_map, session_message_map)
    
    
if __name__ == "__main__":
    main()