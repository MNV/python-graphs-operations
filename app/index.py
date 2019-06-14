import gnureadline
import os
import sys
import re
import string
from typing import Tuple, List, Dict, Callable
from collections import defaultdict
from graphviz import Digraph, Graph
import pandas
from itertools import product

from docs import Docs
from functions import flatten, get_mdnf, extract_variables


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


def get_edge_weight() -> int:
    """
    Prompts the user to input edge weight.
    :param name:
    :param vertices_names:
    :return:
    """
    while True:
        try:
            user_input = input('Weight: ')
            if not user_input.isdigit() or \
                    int(user_input) <= 0 or int(user_input) > sys.maxsize:
                raise Exception('Invalid edge weight. It should be a positive integer. Please, try again.')
            break
        except Exception as exception:
            message = getattr(exception, 'message', repr(exception))
            print(message)

    return int(user_input)


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

                matrix[vertex_source][vertex_destination] = get_edge_weight()
            except Exception as exception:
                message = getattr(exception, 'message', repr(exception))
                print(message)
    return matrix


def find_cycle(source_graph):
    """Return True if the directed graph has a cycle.
    The graph must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices.
    https://codereview.stackexchange.com/a/86067
    """
    visited = set()
    path = [object()]
    path_set = set(path)
    stack = [iter(source_graph)]
    while stack:
        for v in stack[-1]:
            if v in path_set:
                return True
            elif v not in visited:
                visited.add(v)
                path.append(v)
                path_set.add(v)
                stack.append(iter(source_graph.get(v, ())))
                break
        else:
            path_set.remove(path.pop())
            stack.pop()
    return False


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


def get_cnf_by_adjacency_matrix(source_matrix: Dict, by_cols=False) -> str:
    """
    Build a CNF expression by adjacency matrix
    :param source_matrix:
    :return:
    """
    expression = ''
    var_batch = []
    common_batch = []
    for row_vertex, vertices in source_matrix.items():
        if by_cols:
            common_batch.append(vertices.keys())
        else:
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


def find_internally_stable_set(source_matrix, source_vertices):
    """
    Finds the graph's internally stable set.
    :return:
    """
    cnf_expression = get_cnf_by_adjacency_matrix(source_matrix)
    #print('cnf_expression: ')
    #print(cnf_expression)

    globals()['expression'] = cnf_expression

    expr_variables = extract_variables(expression)
    #print('expr_variables: ')
    #print(expr_variables)

    exec(f"""def expression_function({','.join(expr_variables)}): return eval(expression)""", globals())

    truth_table_data = truth_table(expression_function)
    #print(truth_table_data)

    mdnf_string, mdnf_list = get_mdnf(truth_table_data, expression, expr_variables)

    #print('\r\nMDNF: ' + mdnf_string)
    #
    # print('\r\nVertices sets: ')
    # for number, vertices_set in enumerate(mdnf_list, start=1):
    #     print('{number}. {{{vertices}}}'.format(number=number, vertices=', '.join(vertices_set)))

    #max_length = len(max(mdnf_list, key=len))
    #max_graph = [item for item in mdnf_list if len(item) == max_length]

    vertices_sets = set()
    for term in mdnf_list:
        vertices_sets.add(tuple(sorted(set(source_vertices) - set(list(term.lower())))))

    #return mdnf_list
    return vertices_sets


def find_externally_stable_set(source_matrix, source_vertices):
    """
    Finds the graph's externally stable set.
    :return:
    """
    diagonalized_matrix = defaultdict(dict)
    for col_vertex in source_vertices:
        for row_vertex in source_vertices:
            if col_vertex in source_matrix:
                diagonalized_matrix[col_vertex] = source_matrix[col_vertex].copy()
            if col_vertex == row_vertex:
                diagonalized_matrix[row_vertex][col_vertex] = True

    cnf_expression = get_cnf_by_adjacency_matrix(diagonalized_matrix, by_cols=True)
    #print('cnf_expression: ')
    #print(cnf_expression)

    globals()['expression'] = cnf_expression

    expr_variables = extract_variables(expression)
    #print('expr_variables: ')
    #print(expr_variables)

    exec(f"""def expression_function({','.join(expr_variables)}): return eval(expression)""", globals())

    truth_table_data = truth_table(expression_function)
    #print(data)

    mdnf_string, mdnf_list = get_mdnf(truth_table_data, expression, expr_variables)

    #print('\r\nMDNF: ' + mdnf_string)

    vertices_sets = set()
    for term in mdnf_list:
        vertices_sets.add(tuple((term.lower())))

    return vertices_sets


#vertices_history = []
#vertices_paths = []


def __vertices_weights(start_vertex, source_matrix, vertices_weights, terminal_vertex, history):
    """
    Recursively traverse a graph and calculate the lowest weights for vertices.
    :param start_vertex:
    :param source_matrix:
    :param vertices_weights:
    :param terminal_vertex:
    :param history:
    :return:
    """
    if start_vertex in source_matrix:
        for sub_vertex, weight in source_matrix[start_vertex].items():

            if (sub_vertex not in vertices_weights) or \
                    ((vertices_weights[start_vertex][0] + weight) <= vertices_weights[sub_vertex][0]):
                vertices_weights.update(
                    {
                        sub_vertex: [weight + vertices_weights[start_vertex][0], start_vertex]
                    }
                )

            if sub_vertex == terminal_vertex:
                return vertices_weights
            else:
                if (start_vertex + sub_vertex) not in history:
                    history.append(start_vertex + sub_vertex)
                    __vertices_weights(sub_vertex, source_matrix, vertices_weights, terminal_vertex, history)
    else:
        #print(vertices_weights)
        #print('vertices_history: ')
        #print(vertices_history)
        #globals()['vertices_history'] = []
        #vertices_paths.append(vertices_weights.copy())
        return vertices_weights


def find_shortest_paths(source_matrix, vertices_list, start_vertex=None, end_vertex=None):
    """
    Find the shortest path using Dijkstra's algorithm.
    :param source_matrix:
    :param vertices_list:
    :param start_vertex:
    :param end_vertex:
    :return:
    """
    if not start_vertex:
        start_vertex = tuple(source_matrix.keys())[0]

    vertices_weights = {start_vertex: [0, start_vertex]}
    __vertices_weights(start_vertex, source_matrix.copy(), vertices_weights, start_vertex, [])

    #shortest_vertices_path = min(vertices_paths, key=len)
    #print(vertices_paths)

    if not end_vertex:
        end_vertex = vertices_list[-1]

    if end_vertex not in vertices_weights:
        return []

    path_stack = []
    while end_vertex:
        if end_vertex in vertices_weights:
            previous_vertex_list = vertices_weights[end_vertex]
            if previous_vertex_list[0]:
                path_stack.append({previous_vertex_list[1] + end_vertex: previous_vertex_list[0]})
            end_vertex = previous_vertex_list[1] if previous_vertex_list[0] != 0 else None

    return tuple(reversed(path_stack))


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
    # Kernel example
    vertices = tuple(string.ascii_lowercase)[:5]
    adjacency_matrix = {
        'a': {'b': True, 'd': True},
        'b': {'e': True},
        'c': {'d': True},
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

    """
    # Clique example 2
    vertices = tuple(string.ascii_lowercase)[:7]
    adjacency_matrix = {
        'a': {'c': True, 'd': True, 'g': True},
        'b': {'c': True, 'd': True, 'g': True},
        'c': {'d': True, 'f': True, 'g': True},
        'd': {'e': True, 'f': True, 'g': True},
        'f': {'g': True},
    }
    """

    """
    # Dijkstra's algorithm
    vertices = tuple(string.ascii_lowercase)[:6]
    adjacency_matrix = {
        'a': {'b': 9, 'd': 6, 'e': 11},
        'b': {'c': 8},
        'c': {'e': 6, 'f': 9},
        'd': {'b': 5, 'c': 7, 'e': 6},
        'e': {'b': 6, 'f': 4},
    }
    """

    """
    # Dijkstra's algorithm 2
    vertices = tuple(string.ascii_lowercase)[:5]
    adjacency_matrix = {
        'a': {'b': 3, 'c': 6},
        'b': {'e': 6},
        'c': {'d': 2},
        'd': {'e': 2},
    }
    """

    # I. The tier-parallel form
    print('\r\nI. The tier-parallel form: ')
    if find_cycle(adjacency_matrix):
        print('The graph has a cycle. Cannot build a tier-parallel form.')
    else:
        floors = get_floors(adjacency_matrix.copy(), vertices)

        #print('adjacency_matrix: ')
        #print(adjacency_matrix)

        #print('floors: ')
        #print(floors)

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
            print('{} -> {{{}}}'.format(key, ', '.join(value.keys())))

    # II. Graph kernel
    internally_stable_set = find_internally_stable_set(adjacency_matrix.copy(), vertices)
    externally_stable_set = find_externally_stable_set(adjacency_matrix.copy(), vertices)

    print('\r\nII. Graph kernels: ')

    #print('internally_stable_set: ')
    #print(internally_stable_set)

    #print('externally_stable_set: ')
    #print(externally_stable_set)

    graph_kernels = tuple(set(internally_stable_set).intersection(externally_stable_set))

    if not graph_kernels:
        print('The graph kernel not found.')

    for number, kernel in enumerate(graph_kernels, start=1):
        print('{number}. {{{kernel}}}'.format(number=number, kernel=', '.join(kernel)))

    # III. Graph clique
    print('\r\nIII. Clique graphs: ')
    complement_adjacency_matrix = get_complement_matrix(adjacency_matrix.copy(), vertices)
    # print('complement_adjacency_matrix: ')
    # print(dict(complement_adjacency_matrix))
    clique_graphs = set()
    if not complement_adjacency_matrix:
        print('The complement adjacency matrix is empty. Cannot build complement graph.')
    else:
        clique_graphs = find_internally_stable_set(complement_adjacency_matrix.copy(), vertices)
        for num, vertices_set in enumerate(clique_graphs, start=1):
            print('{number}. {{{vertices}}}'.format(number=num, vertices=', '.join(vertices_set)))

    # IV. Dijkstra's algorithm
    print('\r\nIV. The shortest path between first and last vertices (Dijkstra\'s algorithm): ')
    shortest_paths = find_shortest_paths(adjacency_matrix.copy(), vertices)
    #print(shortest_paths)

    if not shortest_paths:
        print('The path not found.')

    for number, path in enumerate(shortest_paths, start=1):
        for vertices_path, weight in path.items():
            print('{number}. {path} -> {weight}'.format(number=number, path=vertices_path, weight=weight))

    ordered_graph = Digraph('G', filename='output/ordered_graph.gv', engine='dot')
    unordered_graph = Graph('G', filename='output/unordered_graph.gv', engine='sfdp')

    for key, value in adjacency_matrix.items():
        for sub_key, weight in value.items():
            edge_weight = str(int(weight))

            if list(filter(lambda dict_item: tuple(dict_item.keys())[0] == key+sub_key, shortest_paths)):
                ordered_graph.edge(
                    key, sub_key, weight=edge_weight, label=edge_weight,
                    colorscheme='paired12', color='7', penwidth='3'
                )
            else:
                ordered_graph.edge(key, sub_key, weight=edge_weight, label=edge_weight)

            unordered_graph.edge(key, sub_key, weight=edge_weight, label=edge_weight)

    ordered_graph.render()
    unordered_graph.render()

except Exception as exception:
    message = getattr(exception, 'message', repr(exception))
    print(message)
