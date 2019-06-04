class Docs:
    def __init__(self):
        """Class constructor"""
        self.doc = """
***************************************************
***  Welcome to Graphs operations calculator!  ***
***************************************************

Please, read the instruction carefully before using the program.
1. Enter the number of vertices of the graph (min: 3, max: 20).
2. Choose the edge direction for each vertex.
3. To stop choosing edge directions for the specific vertex use the terminator symbol - !.

Allowed symbols to use: 
1. 0-9
2. a-z
3. !

Example:
        1. Please, enter the number of vertices: 5
        2. Choose the edge directions:
            a: c
            a: !
            b: !
            c: b
            c: !
            d: e
            d: h
            e: a
            e: g
            e: !
        """

    def get_welcome_message(self) -> str:
        """Returns documentation string"""
        return self.doc
