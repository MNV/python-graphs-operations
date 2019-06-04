import gnureadline
import os
import string
from typing import List, Tuple
from collections import defaultdict
from graphviz import Digraph

from docs import Docs


MIN_NUMBER = 3
MAX_NUMBER = 20
TERMINATOR = '!'


os.system('clear')
docs = Docs()
print(docs.get_welcome_message())


def check_number_input(user_number) -> bool:
    return user_number.isdigit() and MIN_NUMBER <= int(user_number) <= MAX_NUMBER


def get_number() -> int:
    user_input = input('Please, enter the number of vertices: ')
    if not check_number_input(user_input):
        raise Exception('Invalid number. Please, try again.')
    return int(user_input)


def get_letters(count) -> Tuple:
    return tuple(string.ascii_lowercase)[:count]


def get_vertex_direction(name: str, vertices_names: Tuple):
    user_input = input('{}: '.format(name))
    if user_input not in vertices_names and user_input != TERMINATOR:
        raise Exception('Invalid vertex name. Please, try again.')
    return user_input


def get_adjacency_matrix(vertices_names: Tuple):
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


def get_floors(source_matrix, vertices_names: Tuple, source_floor=()):
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
    for floor in floors:
        edges = []
        if isinstance(floor, tuple):
            for vertex in floor:
                if vertex in resulted_matrix:
                    for direction_vertex, value in resulted_matrix[vertex].items():
                        edges.append(vertex + direction_vertex)
                    dot.edges(edges)
                    edges = []
        else:
            if floor in resulted_matrix:
                for direction_vertex, value in resulted_matrix[floor].items():
                    edges.append(floor + direction_vertex)
        if edges:
            dot.edges(edges)

    dot.render('output/ordered_graph.gv')


except Exception as exception:
    message = getattr(exception, 'message', repr(exception))
    print(message)
