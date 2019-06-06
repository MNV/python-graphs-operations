import gnureadline
import os
import re
import string
from typing import Tuple, List, Dict, Callable
from collections import defaultdict
from graphviz import Digraph, Graph
import pandas
from itertools import product

from docs import Docs
from functions import flatten, get_mdnf


MIN_NUMBER = 3
MAX_NUMBER = 20
TERMINATOR = '!'

expression = ''

os.system('clear')
docs = Docs()
print(docs.get_welcome_message())


def check_number_input(user_number: str) -> bool:
    """
    Checks user's input for allowed symbols.
    :param user_number:
    :return:
    """
    return user_number.isdigit() and MIN_NUMBER <= int(user_number) <= MAX_NUMBER


def get_number() -> int:
    """
    Prompts the user to input number of vertices.
    :return:
    """
    user_input = input('Please, enter the number of vertices: ')
    if not check_number_input(user_input):
        raise Exception('Invalid number. Please, try again.')
    return int(user_input)


def get_letters(count: int) -> Tuple:
    """
    Return the letters range for given count.
    :param count:
    :return:
    """
    return tuple(string.ascii_lowercase)[:count]


def get_vertex_direction(name: str, vertices_names: Tuple) -> str:
    """
    Prompts the user to input directed vertex name.
    :param name:
    :param vertices_names:
    :return:
    """
    user_input = input('{}: '.format(name))
    if user_input not in vertices_names and user_input != TERMINATOR:
        raise Exception('Invalid vertex name. Please, try again.')
    return user_input


def get_adjacency_matrix(vertices_names: Tuple) -> Dict:
    """
    Builds an adjacency matrix by given vertices.
    :param vertices_names:
    :return:
    """
    print('Choose the edge directions: ')
    matrix = defaultdict(dict)
    for vertex_source in vertices_names:
        while True:
            try:
                vertex_destination = get_vertex_direction(vertex_source, vertices_names)
                if vertex_destination == '!':
                    break

                if matrix.get(vertex_source, {}).get(vertex_destination, {}):
                    raise Exception('This vertex has already been bound. Choose another one.')

                matrix[vertex_source][vertex_destination] = True
            except Exception as exception:
                message = getattr(exception, 'message', repr(exception))
                print(message)
    return matrix


def get_floors(source_matrix: Dict, vertices_names: Tuple, source_floor=()) -> Tuple:
    """
    Calculates the graph floors (levels) to order it.
    :param source_matrix:
    :param vertices_names:
    :param source_floor:
    :return:
    """
    #print('source_floor: ')
    #print(source_floor)

    floor = []
    split_count = 0
    for col_vertex in vertices_names:
        counter = 0

        for row_vertex in vertices_names:
            if not source_matrix.get(row_vertex, {}).get(col_vertex, {}):
                counter += 1

        if counter == len(vertices_names):
            #print('col_vertex: ')
            #print(col_vertex)
            floor.append(col_vertex)
            split_count += 1

    if floor:
        #print('FLOOR: ')
        #print(floor)
        for vertex in floor:
            if vertex in source_matrix:
                del source_matrix[vertex]

    vertices_names_reduced = tuple((set(tuple(vertices_names)) - set(tuple(floor))))
    #print('vertices_names_reduced: ')
    #print(vertices_names_reduced)

    next_floor = list(source_floor)
    if len(floor) == 1:
        next_floor.append(floor[0])
    else:
        next_floor.append(tuple(floor))
    floor = tuple(next_floor)
    #print('floor appended: ')
    #print(floor)

    if split_count == 0:
        return floor + vertices_names_reduced

    return get_floors(source_matrix, vertices_names_reduced, floor)


def get_complement_matrix(source_matrix: Dict, vertices_names: Tuple) -> Dict:
    """
    Build a complement adjacency_matrix for given matrix.
    :param source_matrix:
    :param vertices_names:
    :return:
    """
    complement_matrix = defaultdict(dict)
    for col_vertex in vertices_names:
        for row_vertex in vertices_names:
            if col_vertex == row_vertex:
                break
            if not source_matrix.get(row_vertex, {}).get(col_vertex, {}):
                complement_matrix[row_vertex][col_vertex] = True

    return complement_matrix


def get_cnf_by_adjacency_matrix(source_matrix: Dict) -> str:
    """
    Build a CNF expression by adjacency matrix
    :param source_matrix:
    :return:
    """
    expression = ''
    var_batch = []
    common_batch = []
    for row_vertex, vertices in source_matrix.items():
        for col_vertex in vertices.keys():
            common_batch.append([row_vertex, col_vertex])

    #print(common_batch)

    result_batch = []
    for batch in common_batch:
        result_batch.append('(' + ' or '.join(batch) + ')')

    return ' and '.join(result_batch)


def truth_table(func: Callable) -> pandas.DataFrame:
    """
    Build the truth table using given function
    :param func:
    :return:
    """
    values = [list(args) + [int(func(*args))] for args in product([0, 1], repeat=func.__code__.co_argcount)]
    return pandas.DataFrame(
        values,
        columns=(list(func.__code__.co_varnames) + [expression]))


def find_clique_graphs() -> List:
    """
    Finds clique graphs
    :return:
    """
    cnf_expression = get_cnf_by_adjacency_matrix(complement_adjacency_matrix)
    #print('cnf_expression: ')
    #print(cnf_expression)

    globals()['expression'] = cnf_expression

    expr_variables = re.findall(r'([a-z])\sor\s([a-z])', expression)
    expr_variables = sorted(set(flatten(expr_variables)))
    #print('expr_variables: ')
    #print(expr_variables)

    exec(f"""def expression_function({','.join(expr_variables)}): return eval(expression)""", globals())

    data = truth_table(expression_function)
    #print(data)

    mdnf_string, mdnf_list = get_mdnf(data, expression, expr_variables)

    print('\r\nMDNF: ' + mdnf_string)
    #
    # print('\r\nVertices sets: ')
    # for number, vertices_set in enumerate(mdnf_list, start=1):
    #     print('{number}. {{{vertices}}}'.format(number=number, vertices=', '.join(vertices_set)))

    #max_length = len(max(mdnf_list, key=len))
    #max_graph = [item for item in mdnf_list if len(item) == max_length]

    return mdnf_list


try:
    vertices = get_letters(get_number())
    adjacency_matrix = get_adjacency_matrix(vertices)

    #print(dict(adjacency_matrix))

    """
    vertices = tuple(string.ascii_lowercase)[:8]
    adjacency_matrix = {
        'a': {'c': True},
        'c': {'b': True},
        'd': {'e': True, 'h': True},
        'e': {'a': True, 'g': True},
        'f': {'c': True},
        'g': {'c': True},
        'h': {'a': True, 'f': True},
    }
    """

    """
    vertices = tuple(string.ascii_lowercase)[:8]
    adjacency_matrix = {
        'a': {'b': True, 'f': True, 'g': True},
        'c': {'d': True, 'e': True},
        'd': {'a': True},
        'e': {'g': True},
        'h': {'b': True, 'e': True},
    }
    """

    """
    vertices = tuple(string.ascii_lowercase)[:8]
    adjacency_matrix = {
        'a': {'c': True, 'h': True},
        'b': {'g': True, 'f': True},
        'd': {'c': True},
        'e': {'d': True, 'f': True, 'c': True, 'a': True, 'h': True},
        'f': {'c': True},
        'g': {'f': True},
    }
    """

    """
    # 8
    vertices = tuple(string.ascii_lowercase)[:9]
    adjacency_matrix = {
        'a': {'b': True, 'e': True},
        'b': {'c': True, 'e': True, 'f': True},
        'd': {'b': True, 'e': True, 'h': True},
        'e': {'c': True, 'f': True, 'h': True},
        'g': {'h': True},
        'h': {'i': True},
    }
    """

    """
    # 15
    vertices = tuple(string.ascii_lowercase)[:8]
    adjacency_matrix = {
        'a': {'b': True, 'f': True, 'g': True},
        'b': {'c': True},
        'c': {'h': True},
        'd': {'b': True, 'f': True},
        'e': {'b': True, 'd': True, 'g': True},
        'g': {'f': True},
    }
    """

    """
    # Clique example
    vertices = tuple(string.ascii_lowercase)[:6]
    adjacency_matrix = {
        'a': {'b': True, 'c': True},
        'b': {'d': True, 'e': True, 'f': True},
        'c': {'e': True, 'f': True},
        'd': {'e': True, 'f': True},
        'e': {'f': True},
    }
    """

    floors = get_floors(adjacency_matrix.copy(), vertices)

    #print('adjacency_matrix: ')
    #print(adjacency_matrix)

    #print('floors: ')
    #print(floors)

    print('\r\nGraph levels: ')
    for number, floor in enumerate(floors, start=1):
        if floor:
            print('{}. {}'.format(number, ', '.join(floor)))

    resulted_matrix = {}
    for floor in floors:
        if isinstance(floor, str):
            if floor in adjacency_matrix:
                resulted_matrix[floor] = adjacency_matrix[floor]
        else:
            for sub_floor in floor:
                if sub_floor in adjacency_matrix:
                    resulted_matrix[sub_floor] = adjacency_matrix[sub_floor]

    print('\r\nAdjacency matrix: ')
    for key, value in resulted_matrix.items():
        for vertex in value.keys():
            print('{} -> {}'.format(key, vertex))

    """
    g = Digraph('G', filename='output/ordered_graph.gv', engine='dot')
    for key, value in resulted_matrix.items():
        for sub_key in value.keys():
            g.edge(key, sub_key)
    g.view()
    """

    dot = Digraph()
    graph = Graph(engine='sfdp')
    for floor in floors:
        edges = []
        if isinstance(floor, tuple):
            for vertex in floor:
                if vertex in resulted_matrix:
                    for direction_vertex, value in resulted_matrix[vertex].items():
                        edges.append(vertex + direction_vertex)
                    dot.edges(edges)
                    graph.edges(edges)
                    edges = []
        else:
            if floor in resulted_matrix:
                for direction_vertex, value in resulted_matrix[floor].items():
                    edges.append(floor + direction_vertex)
        if edges:
            dot.edges(edges)
            graph.edges(edges)

    dot.render('output/ordered_graph.gv')
    graph.render('output/unordered_graph.gv')


    complement_adjacency_matrix = get_complement_matrix(adjacency_matrix.copy(), vertices)
    #print('complement_adjacency_matrix: ')
    #print(dict(complement_adjacency_matrix))

    if not complement_adjacency_matrix:
        print('The complement adjacency matrix is empty. Cannot build complement graph.')
    else:
        clique_graphs = find_clique_graphs()
        print('\r\nClique graph: ')
        for num, vertices_set in enumerate(clique_graphs, start=1):
            print('{number}. {{{vertices}}}'.format(number=num, vertices=', '.join(vertices_set)))


except Exception as exception:
    message = getattr(exception, 'message', repr(exception))
    print(message)
