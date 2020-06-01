import cv2
import numpy as np
from utils import constants


def get_slopes(points):
    """
    Get slope of each edge of the polygon
    Polygon can have either have 4 or 6 edges
    :param points: coordinates of the polygon
    :return: a list of slopes of the edges of the polygon
    """
    # Get no. of points
    points_length = len(points)
    i = 0
    # Define an empty list to store slopes of all edges
    slopes = []
    while i < points_length:
        # Get indices of the two points of the edge
        if i != points_length - 1:
            j = i + 1
        else:
            j = 0
        # Calculate slope and append it to the list
        slopes.append((points[j][1] - points[i][1]) / (points[j][0] - points[i][0]))
        i += 1

    return slopes


def get_y_values(x, slopes, coordinates, edge_count):
    """
    Calculate the y value of the current x from each edge
    :param x: x-coordinate of the current node
    :param slopes:a list of slopes of all edges of the polygon
    :param coordinates: a list of vertices of the polygon
    :param edge_count: no. of edges in the polygon
    :return: a list of all y-values
    """
    # Define an empty list to store all y-values
    dist = []
    for i in range(edge_count):
        dist.append(slopes[i] * (x - coordinates[i][0]) + coordinates[i][1])
    # Return the list of y-values
    return dist


class Map:
    def __init__(self, radius, clearance):
        deg_30 = np.pi / 6
        deg_60 = np.pi / 3
        # Various class parameters
        self.height = constants.MAP_SIZE[0]
        self.width = constants.MAP_SIZE[1]
        self.thresh = radius + clearance
        # Coordinates of the convex polygon
        self.coord_polygon = np.array([(20, self.height - 120),
                                       (25, self.height - 185),
                                       (75, self.height - 185),
                                       (100, self.height - 150),
                                       (75, self.height - 120),
                                       (50, self.height - 150)], dtype=np.int32)
        # Coordinates of the rectangle
        self.coord_rectangle = np.array([(95 - 75 * np.cos(deg_30), self.height - 75 * np.sin(deg_30) - 30),
                                         (95 - 75 * np.cos(deg_30) + 10 * np.cos(deg_60), self.height
                                          - 75 * np.sin(deg_30) - 10 * np.sin(deg_60) - 30),
                                         (95 + 10 * np.cos(deg_60), self.height - 10 * np.sin(deg_60) - 30),
                                         (95, self.height - 30)],
                                        dtype=np.int32).reshape((-1, 2))
        # Coordinates of the rhombus
        self.coord_rhombus = np.array([(300 - 75 - (50 / 2), self.height - (30 / 2) - 10),
                                       (300 - 75, self.height - 30 - 10),
                                       (300 - 75 + (50 / 2), self.height - (30 / 2) - 10),
                                       (300 - 75, self.height - 10)],
                                      dtype=np.int32).reshape((-1, 2))
        # Define parameters of curved obstacles
        self.circle = [25, (225, 50)]
        self.ellipse = [(40, 20), (150, self.height - 100)]
        # Get slopes of all the edges of the convex polygon, rectangle, and rhombus
        self.slopes_poly = get_slopes(self.coord_polygon)
        self.slopes_rect = get_slopes(self.coord_rectangle)
        self.slopes_rhom = get_slopes(self.coord_rhombus)
        # Define empty world and add obstacles to it
        self.obstacle_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.obstacle_img = self.draw_obstacles()
        # Get image to search for obstacles
        self.check_img = self.erode_image()

    def draw_circle(self):
        """
        Draw the circle obstacle on the map-image
        :return: nothing
        """
        # Define center of the circle
        a = self.circle[1][0]
        b = self.circle[1][1]
        # Define radius of the circle
        r = self.circle[0]
        # Draw the circle
        for y in range(self.height):
            for x in range(self.width):
                if (x - a) ** 2 + (y - b) ** 2 <= r ** 2:
                    self.obstacle_img[y][x] = (0, 0, 0)

    def draw_ellipse(self):
        """
                Draw the circle obstacle on the map-image
                :return: nothing
                """
        # Get axes length of the ellipse
        a = self.ellipse[0][0]
        b = self.ellipse[0][1]
        # Get center of the ellipse
        center_a = self.ellipse[1][0]
        center_b = self.ellipse[1][1]
        # Draw the ellipse
        for y in range(self.height):
            for x in range(self.width):
                if ((x - center_a) / a) ** 2 + ((y - center_b) / b) ** 2 <= 1:
                    self.obstacle_img[y][x] = (0, 0, 0)

    def draw_polygons(self):
        """
        Draw the convex polygon, rectangle and rhombus on the map-image
        :return: nothing
        """
        last_poly_slope = ((self.coord_polygon[2][1] - self.coord_polygon[5][1]) /
                           (self.coord_polygon[2][0] - self.coord_polygon[5][0]))
        for y in range(self.height):
            for x in range(self.width):
                # Get y values for each edge of the convex polygon
                y_poly = get_y_values(x, self.slopes_poly, self.coord_polygon, 6)
                y_poly.append(last_poly_slope * (x - self.coord_polygon[5][0]) + self.coord_polygon[5][1])
                # Get y values for each edge of the rectangle
                y_rect = get_y_values(x, self.slopes_rect, self.coord_rectangle, 4)
                # Get y values for each edge of the rhombus
                y_rhom = get_y_values(x, self.slopes_rhom, self.coord_rhombus, 4)
                # Draw the convex polygon
                if y_poly[0] <= y <= y_poly[6] and y_poly[1] <= y <= y_poly[5]:
                    self.obstacle_img[y][x] = (0, 0, 0)
                elif y_poly[2] <= y <= y_poly[4] and y_poly[6] <= y <= y_poly[3]:
                    self.obstacle_img[y][x] = (0, 0, 0)
                # Draw the tilted rectangle
                elif y_rect[0] <= y <= y_rect[2] and y_rect[1] <= y <= y_rect[3]:
                    self.obstacle_img[y][x] = (0, 0, 0)
                # Draw the rhombus
                elif y_rhom[0] <= y <= y_rhom[3] and y_rhom[1] <= y <= y_rhom[2]:
                    self.obstacle_img[y][x] = (0, 0, 0)

    def check_node_validity(self, x, y):
        """
        Method to check whether point lies within any obstacle
        :param x: x-coordinate of the current node
        :param y: y-coordinate of the current node
        :return: false if point lies within any obstacle
        """
        # Check whether the current node lies within the map
        if x >= self.width or y >= self.height:
            return False
        # Check whether the current node lies within any obstacle
        elif self.check_img[y, x].all() == 0:
            return False

        return True

    def erode_image(self):
        """
        Get eroded image to check for obstacles considering the robot radius and clearance
        :return: image with obstacle space expanded to distance threshold between robot and obstacle
        """
        # Get map with obstacles
        eroded_img = self.obstacle_img.copy()
        # Erode map image for rigid robot
        if self.thresh:
            kernel_size = (self.thresh * 2) + 1
            erode_kernel = np.ones((kernel_size, kernel_size), np.uint8)
            eroded_img = cv2.erode(eroded_img, erode_kernel, iterations=1)
            # Include border in obstacle space
            for y in range(self.height):
                for x in range(self.width):
                    if (0 <= y < self.thresh or self.width - self.thresh <= x < self.width
                            or 0 <= x < self.thresh or self.height - self.thresh <= y < self.height):
                        eroded_img[y][x] = (0, 0, 0)

        return eroded_img

    def draw_obstacles(self):
        """
        Draw map using half-plane equations
        :return: map-image with all obstacles
        """
        # Fill map-image with white color
        self.obstacle_img.fill(255)
        # Draw various obstacles on the map
        self.draw_circle()
        self.draw_ellipse()
        self.draw_polygons()

        return self.obstacle_img
