import pickle
from typing import List, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw


def convert_to_binary_image() -> np.ndarray:
    """
    converts the image into a binary numpy array of shape (height, width)
    """
    outline = Image.open("logo_outline.png").convert("RGB")
    w, h = outline.size
    print(f"width and height are ({w}, {h})")

    # floodfill the image with green
    seed = (w // 2, h // 2)
    fill_color = (0, 255, 0)
    ImageDraw.floodfill(outline, seed, fill_color)
    pixels = np.asarray(outline)
    return np.apply_along_axis(lambda color: tuple(color) == fill_color, 2, pixels)


def bordering_cells(cell: Tuple[int, int], h: int, w: int):
    """
    returns the cells bordering the given one, in the form (cell_y, cell_x)
    """
    y, x = cell
    return list(
        filter(
            lambda coords: (0 <= coords[0] < h and 0 <= coords[1] < w),
            [
                (y + 1, x),
                (y - 1, x),
                (y, x + 1),
                (y, x - 1),
            ],
        )
    )


def find_taxicab_distance_to_well(a: np.ndarray) -> np.ndarray:
    """
    well is an area of the binary array a that is True
    """
    h, w = a.shape
    distance = np.full_like(a, fill_value=None, dtype=object)
    start_point = (h // 2, w // 2)
    distance[start_point] = 0

    # find border of well, bucket fill the well and take note of the border pixels
    border_cells: List[Tuple[int, int]] = []
    list_of_edges = [start_point]
    print("start finding edges")

    while True:
        new_list_of_edges: Set[Tuple[int, int]] = set()
        for edge_cell in list_of_edges:
            if a[edge_cell]:
                distance[edge_cell] = 0
            if any(
                map(lambda cell: a[cell] == False, bordering_cells(edge_cell, h, w))
            ):
                border_cells.append(edge_cell)

            unvisited_neighbors_inside_well = filter(
                lambda cell: distance[cell] is None and a[cell],
                bordering_cells(edge_cell, h, w),
            )

            new_list_of_edges = new_list_of_edges.union(unvisited_neighbors_inside_well)

        list_of_edges = list(new_list_of_edges)
        if len(list_of_edges) == 0:
            break

    print("found all the edges")

    list_of_edges = border_cells
    while True:
        new_list_of_edges: Set[Tuple[int, int]] = set()
        for edge_cell in list_of_edges:
            if a[edge_cell]:
                # a is part of the well
                distance[edge_cell] = 0
            else:
                distance[edge_cell] = (
                    min(
                        map(
                            lambda cell: distance[cell],
                            filter(
                                lambda cell: distance[cell] is not None,
                                bordering_cells(edge_cell, h, w),
                            ),
                        )
                    )
                    + 1
                )
            unvisited_neighbors_of_current_cell = filter(
                lambda cell: distance[cell] is None,
                bordering_cells(edge_cell, h, w),
            )

            new_list_of_edges = new_list_of_edges.union(
                unvisited_neighbors_of_current_cell
            )
        # print(f"new list: {new_list_of_edges}\nold list:{list_of_edges}")
        list_of_edges = list(new_list_of_edges)
        if len(new_list_of_edges) == 0:
            break
        distance_from_start = max(
            map(
                lambda cell: abs(cell[0] - start_point[0])
                + abs(cell[1] - start_point[1]),
                new_list_of_edges,
            )
        )
        if distance_from_start % 20 == 0:
            print(f"progress: {distance_from_start}")

    return distance


# def find_distance_to_well(a: np.ndarray) -> np.ndarray:
#     h, w = a.shape
#     output = np.zeros_like(a)
#     for i in range(h):
#         for j in range(w):
#             if a[i, j]:
#                 output[i, j] = 0.0
#             else:
#                 pass

#     return output


def main():
    bin_array = np.asarray(Image.open("bucket_fill.png").convert("1"))

    distance = find_taxicab_distance_to_well(bin_array)

    distance_image = Image.fromarray(
        np.array((254.0 / np.max(distance)) * distance, dtype=np.uint8)
    )
    distance_image.show()


if __name__ == "__main__":
    main()
