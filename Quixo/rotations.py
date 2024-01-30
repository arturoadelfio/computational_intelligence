import numpy as np


def rotate_board_90_clockwise(matrix):
    if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        raise ValueError("Input matrix must be a 5x5 matrix")
 
    # Transpose the matrix
    transposed_matrix = [list(row) for row in zip(*matrix)]
 
    # Reverse each row in the transposed matrix to get the 90-degree counterclockwise rotation
    rotated_matrix = [row[::-1] for row in transposed_matrix]
 
    return np.array(rotated_matrix)

def rotate_board_90_anticlockwise(matrix):
    if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        raise ValueError("Input matrix must be a 5x5 matrix")
 
    rotated_matrix = [row[::-1] for row in matrix]
    
    transposed_matrix = [list(row) for row in zip(*rotated_matrix)]
    
    return np.array(transposed_matrix)


def rotate_90_clockwise(points):
    if any(len(point) != 2 for point in points):
        raise ValueError("Each point must be a tuple of length 2")

    # Convert tuples to matrix for rotation
    matrix = [[0] * 5 for _ in range(5)]
    for x, y in points:
        matrix[x][y] = 1

    # Transpose the matrix
    transposed_matrix = [list(row) for row in zip(*matrix)]

    rotated_matrix = [row[::-1] for row in transposed_matrix]

    # Extract the rotated points from the rotated matrix
    rotated_points = [(i, j) for i, row in enumerate(rotated_matrix) for j, value in enumerate(row) if value == 1]

    return rotated_points

def rotate_90_anticlockwise(points):
    if any(len(point) != 2 for point in points):
        raise ValueError("Each point must be a tuple of length 2")

    # Convert tuples to matrix for rotation
    matrix = [[0] * 5 for _ in range(5)]
    for x, y in points:
        matrix[x][y] = 1

    rotated_matrix = [row[::-1] for row in matrix]
    
    transposed_matrix = [list(row) for row in zip(*rotated_matrix)]

    # Extract the rotated points from the rotated matrix
    rotated_points = [(i, j) for i, row in enumerate(transposed_matrix) for j, value in enumerate(row) if value == 1]

    return rotated_points
    
    
def mirror_points(points):
    if any(len(point) != 2 for point in points):
        raise ValueError("Each point must be a tuple of length 2")

    mirrored_points = [(x, 4 - y) for x, y in points]

    return mirrored_points

def mirror_board(matrix):
    if any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("Input matrix must be rectangular")
 
    mirrored_matrix = [row[::-1] for row in matrix]
 
    return np.array(mirrored_matrix)