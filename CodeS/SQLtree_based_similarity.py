from sqlglot import parse_one, exp , diff
from sqlglot.diff import Keep
from sqlglot.errors import ErrorLevel
import random
import json
import math

def jaccard_similarity(set_1 , set_2):
    
    intersection = len(set_1.intersection(set_2))
    union = len(set_1.union(set_2))

    if union==0:
        return 1
    
    return intersection / union

def generator_to_set(myGenerator):
    #It extracts the attribute .this for each element in the generator gotten from parse_one(query1).find_all(exp.Table/Column)
    mySet = set()
    myList = list(myGenerator)
    for i in myList:
        mySet.add(i.this.this)
    return mySet

def combine_similarities(column_similarity, table_similarity , tree_similarity):
    return (column_similarity+table_similarity+tree_similarity)/3

def get_most_similar_query(query_list):
    valid_query_list = []
    valid_q_parsed_list = []
    for i , query in enumerate(query_list):
        query = query.replace('`' , '"')
        try:
            print('query: ', query)
            q_parsed = parse_one(query , error_level=ErrorLevel.IGNORE)
            bfs_obj = q_parsed.bfs()
            dfs_obj = q_parsed.bfs()
            print('bfs:')
            for node in bfs_obj:
                print(node)
            print('dfs:')
            for node in dfs_obj:
                print(node)
            print('\n')
            valid_q_parsed_list.append(q_parsed)
            valid_query_list.append(query)
        except:
            print('One of the queries are invalid: index = ' , i )
    if len(valid_query_list)==0:
        index = random.randint(0, len(query_list)-1)
        return query_list[index] , index
        
    elif len(valid_query_list)==1:
        return valid_query_list[0] , 0
        
    else:
        sim_sum_list = []
        for i in range(len(valid_q_parsed_list)):
            comparing_q_list = valid_q_parsed_list.copy()
            comparing_query = comparing_q_list.pop(i)
            sim_sum = 0
            for q in comparing_q_list:
                sim_sum += query_similarity( comparing_query , q )
            sim_sum_list.append(sim_sum)
        max_sim = max(sim_sum_list)
        max_sim_index = sim_sum_list.index(max_sim)
        return valid_query_list[max_sim_index] , max_sim_index
    
def get_n_most_similar_query(query_list , n):
    output_query_list = []
    output_index_list = []
    for i in range(n):
        query , index = get_most_similar_query(query_list)
        output_query_list.append(query)
        output_index_list.append(index)
        query_list[index] = ''
    return output_query_list, output_index_list


def query_similarity(q1_parsed , q2_parsed):
    #calculates the similarity between two queries.
    
    query1_columns = q1_parsed.find_all(exp.Column)
    query1_tables = q1_parsed.find_all(exp.Table)
    
    query2_columns = q2_parsed.find_all(exp.Column)
    query2_tables = q2_parsed.find_all(exp.Table)
    
    #putting the extracted columns in two sets
    q1_column_set = generator_to_set(query1_columns)
    q2_column_set = generator_to_set(query2_columns)
    column_similarity = jaccard_similarity( q1_column_set , q2_column_set )
        
    q1_table_set = generator_to_set(query1_tables)
    q2_table_set = generator_to_set(query2_tables)
    table_similarity = jaccard_similarity( q1_table_set , q2_table_set )

    diff_list = diff(q1_parsed , q2_parsed)
    number_of_keep = 0
    for i in diff_list:
        number_of_keep += int(isinstance( i , Keep ))
    tree_similarity = number_of_keep/len(diff_list)
    
    return combine_similarities( column_similarity , table_similarity , tree_similarity )

def unparsed_query_similarity(q1 , q2):
    q1 = q1.replace('`' , '"')
    q2 = q2.replace('`' , '"')
    try:
        q1_parsed = parse_one(q1 , error_level=ErrorLevel.IGNORE)
        nodes = q1_parsed.bfs()
    
        q2_parsed = parse_one(q2 , error_level=ErrorLevel.IGNORE)
        
        return math.log( query_similarity( q1_parsed , q2_parsed ) )

    except:
        return -1000000