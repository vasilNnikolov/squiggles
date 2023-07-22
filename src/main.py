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


def find_taxicab_distance_to_well(
    bin_array: np.ndarray, point_in_well: Tuple[int, int]
) -> np.ndarray:
    """
    well is an area of the binary array a that is True
    """
    h, w = bin_array.shape
    distance = np.full_like(bin_array, fill_value=None, dtype=object)
    point_in_well = (h // 2, w // 2)
    distance[point_in_well] = 0

    # find border of well, bucket fill the well and take note of the border pixels
    border_cells: List[Tuple[int, int]] = []
    list_of_edges = [point_in_well]
    print("start finding edges")

    while True:
        new_list_of_edges: Set[Tuple[int, int]] = set()
        for edge_cell in list_of_edges:
            if bin_array[edge_cell]:
                distance[edge_cell] = 0
            if any(
                map(
                    lambda cell: bin_array[cell] == False,
                    bordering_cells(edge_cell, h, w),
                )
            ):
                border_cells.append(edge_cell)

            unvisited_neighbors_inside_well = filter(
                lambda cell: distance[cell] is None and bin_array[cell],
                bordering_cells(edge_cell, h, w),
            )

            new_list_of_edges = new_list_of_edges.union(unvisited_neighbors_inside_well)

        list_of_edges = list(new_list_of_edges)
        if len(list_of_edges) == 0:
            break

    print(f"found all the edges, there are {len(border_cells)} many edge pixels")

    list_of_edges = border_cells
    while True:
        new_list_of_edges: Set[Tuple[int, int]] = set()
        for edge_cell in list_of_edges:
            if bin_array[edge_cell]:
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
                lambda cell: abs(cell[0] - point_in_well[0])
                + abs(cell[1] - point_in_well[1]),
                new_list_of_edges,
            )
        )
        if distance_from_start % 20 == 0:
            print(f"progress: {distance_from_start}")

    return distance


def find_distance_to_well(bin_array: np.ndarray) -> np.ndarray:
    h, w = bin_array.shape

    N_ROTATIONS = 10
    # applicable for every image, slower
    # diagonal_length = int(np.sqrt(h**2 + w**2)) + 1
    # applicable for circular designs, not applicable for every image
    diagonal_length = max(h, w)
    larger_original = np.zeros((diagonal_length, diagonal_length), dtype=bool)
    # fill larger original with content of originial, keeping the center of the original in the center of the larger one
    offset = ((diagonal_length - h) // 2, (diagonal_length - w) // 2)
    larger_original[offset[0] : offset[0] + h, offset[1] : offset[1] + w] = bin_array[
        :, :
    ]

    rotated_arrays = np.zeros(
        (diagonal_length, diagonal_length, N_ROTATIONS), dtype=bool
    )
    distance = np.full(
        (diagonal_length, diagonal_length, N_ROTATIONS), 10 * diagonal_length
    )

    center_of_rotation = (diagonal_length // 2, diagonal_length // 2)

    def rotate_array(a: np.ndarray, angle):
        h, w = a.shape

        rotated_a = np.zeros_like(a)
        for i in range(h):
            for j in range(w):
                x = j - w // 2
                y = i - h // 2
                point_in_old_array = (
                    int(h // 2 + x * np.sin(-angle) + y * np.cos(-angle)),
                    int(w // 2 + x * np.cos(-angle) - y * np.sin(-angle)),
                )
                if 0 <= point_in_old_array[0] < h and 0 <= point_in_old_array[1] < w:
                    rotated_a[i, j] = a[point_in_old_array]
        return rotated_a

    for rotation_index in range(0, N_ROTATIONS):
        # 2/3 since the logo has 120 degree symmetry
        angle_to_rotate = 2.0 / 3.0 * np.pi * rotation_index / N_ROTATIONS

        rotated_arrays[:, :, rotation_index] = rotate_array(
            larger_original, angle_to_rotate
        )
        # Image.fromarray(rotated_arrays[:, :, rotation_index]).convert("RGB").show()
        # run taxicab distance on rotated array and update the distance array
        rotated_distance = find_taxicab_distance_to_well(
            rotated_arrays[:, :, rotation_index], center_of_rotation
        )

        distance[:, :, rotation_index] = rotate_array(
            rotated_distance, -angle_to_rotate
        )

    return np.min(distance, axis=2)


def main():
    bin_array = np.asarray(Image.open("bucket_fill.png").convert("1"))
    h, w = bin_array.shape

    distance = find_distance_to_well(bin_array)

    distance_image = Image.fromarray(
        np.array((254.0 / np.max(distance)) * distance, dtype=np.uint8)
    )
    distance_image.show()
    distance_image.save("approx_eucl_distance_image.png")


if __name__ == "__main__":
    main()
