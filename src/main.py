from typing import List, Set, Tuple

import matplotlib

matplotlib.use("TkAgg")
import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


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

    N_ROTATIONS = 30
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


def find_gradient(image: np.ndarray) -> np.ndarray:
    """
    finds the gradient of the grayscale image and returns it in the form array((image.shape,2))
    """
    # scale image down
    scale_down_factor = 255 / np.max(image)
    image_scaled = scale_down_factor * image.astype(np.float64)
    grad_x = cv2.Sobel(src=image_scaled, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=9)
    grad_y = cv2.Sobel(src=image_scaled, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=9)
    gradient = (
        1
        / scale_down_factor
        * np.stack(
            [
                np.transpose(grad_x.astype(np.float64)),
                np.transpose(grad_y.astype(np.float64)),
            ],
            axis=2,
        )
    )
    return gradient


def get_potential(image: np.ndarray) -> np.ndarray:
    return 10e-3 * (image.astype(np.float64)) ** 4


def num_integrate(distance_image):
    force = -find_gradient(potential(distance_image))

    # f_y, f_x = force[:, :, 0], force[:, :, 1]
    # main simulation
    B = 0.5
    E_strength = 10
    angle = np.pi / 2
    q = 1
    pos = np.array([400.0, 400.0])
    vel = np.array([10.0, 0.0])
    dt = 0.01
    T_end = 100
    N_points = int(T_end / dt)
    positions = np.zeros((N_points, 2))
    for frame_index in tqdm(range(N_points)):
        F = (
            force[int(pos[0]), int(pos[1])]
            + q * E_strength * np.array([np.cos(angle), np.sin(angle)])
            + q * B * np.array([-vel[1], vel[0]])
        )
        vel += F * dt
        pos += vel * dt
        positions[frame_index] = pos

    return positions


def main():
    main_image = cv2.GaussianBlur(
        np.array(Image.open("approx_eucl_distance_image_full.png")), (17, 17), 0
    )
    distance_image = np.array(main_image)
    image_center = np.array(distance_image.shape) / 2

    # initialize graphics
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(
    #     # autoscale_on=False,
    #     xlim=(0, distance_image.shape[1]),
    #     ylim=(0, distance_image.shape[0]),
    # )
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax.set_xlim((0, distance_image.shape[1]))
    ax.set_ylim((0, distance_image.shape[0]))
    ax.set_aspect("equal")

    (trace,) = ax.plot([], [], linewidth=1)
    (arrow,) = ax.plot([], [])
    time_text = ax.text(0.05, 0.9, "", color="white")

    (potential_trace,) = ax2.plot([], [])
    ax2.set_xlim((0, 40))
    ax2.set_ylim((0, 36621862.56))

    # animation
    potential_field = get_potential(distance_image)
    force = -find_gradient(potential_field)
    print(f"max of potential is {np.max(potential_field)}")

    class SimulationState:
        def __init__(self):
            self.B = 2
            self.E_strength = 10
            self.angle = np.pi / 2
            self.q = 1
            self.pos = np.array([400.0, 400.0])
            self.vel = np.array([100.0, 0.0])
            self.dt = 0.1
            self.current_time = 0
            self.T_end = 1000
            self.N_points = int(self.T_end / self.dt)

        def update_parameters(self, potential_field: np.ndarray):
            int_position = (int(self.pos[0]), int(self.pos[1]))
            potential = potential_field[int_position]
            self.dt = 1 / (self.vel[0] ** 2 + self.vel[1] ** 2) ** 0.5 + 0.001
            # self.B = 2 / (1 + potential)
            self.current_time += self.dt
            self.angle += 2 * np.pi * self.dt / 70
            F = (
                force[int_position]
                + self.q
                * self.E_strength
                * np.array([np.cos(self.angle), np.sin(self.angle)])
                + self.q * self.B * np.array([-self.vel[1], self.vel[0]])
            )
            # friction
            # F +=
            self.vel += F * self.dt
            self.pos += self.vel * self.dt

    # main simulation
    s = SimulationState()
    positions = np.zeros((s.N_points, 2))
    potentials = np.zeros(s.N_points)
    times = np.zeros(s.N_points)

    def animate(frame_index, s: SimulationState):
        s.update_parameters(potential_field)
        int_position = (int(s.pos[0]), int(s.pos[1]))
        positions[frame_index] = s.pos

        # draw to screen
        trace.set_data(positions[:frame_index, 0], positions[:frame_index, 1])
        # draw arrow in direction of E field
        arrow.set_data(
            [image_center[0], image_center[0] + 100 * np.cos(s.angle)],
            [image_center[1], image_center[1] + 100 * np.sin(s.angle)],
        )
        # text with progress
        time_text.set_text(
            f"""
            Progress: {frame_index/s.N_points:.4f}%
            Simulation time {s.current_time:.4f} s
            B is {s.B:.4f}
            dt is {s.dt:.4f}
            Potential is {potential_field[int_position]}
            position is {int_position}"""
        )

        potentials[frame_index] = potential_field[int_position]
        times[frame_index] = s.current_time
        potential_trace.set_data(times[:frame_index], potentials[:frame_index])
        return (
            trace,
            arrow,
            time_text,
            potential_trace,
        )

    log_potential_field = np.log(1 + potential_field)
    # plt.imshow(255 / np.max(log_potential_field) * log_potential_field)
    ax.imshow(log_potential_field)
    ani = animation.FuncAnimation(
        fig,
        lambda frame_index: animate(frame_index, s),
        s.N_points,
        interval=0,
        blit=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
