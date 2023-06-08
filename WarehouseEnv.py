import random
from copy import copy
import pygame

board_size = 5
import time

def manhattan_distance(p0, p1):
    return abs(p0[0] - p1[0]) + abs(p0[1] - p1[1])


class Robot(object):
    def __init__(self, position, battery, credit):
        self.position = position
        self.battery = battery
        self.credit = credit
        self.package = None

    def __repr__(self):
        return 'position:' + str(self.position) + ' battery: ' + str(self.battery) + \
            ' credit: ' + str(self.credit) + ' package: [' + str(self.package) + ']'


class Package(object):
    def __init__(self, position, destination):
        self.position = position
        self.destination = destination
        self.on_board = False

    def __repr__(self):
        return 'position:' + str(self.position) + ' destination: ' + str(self.destination)


class ChargeStation(object):
    def __init__(self, position):
        self.position = position

    def __repr__(self):
        return 'position:' + str(self.position)


class WarehouseEnv(object):
    def __init__(self):
        self.charge_stations = None
        self.packages = None
        self.robots = None
        self.seed = None
        self.num_steps = None

        # for animation purposes
        self.window = None
        self.clock = None

    def generate(self, seed, num_steps):
        self.num_steps = num_steps
        self.seed = seed
        self.robots = [Robot(p, 20, 0) for p in self.random_cells(2)]
        self.packages = [Package(p, d) for _ in range(4) for p in self.random_cells(1) for d in
                         self.random_cells(1)]
        for i in range(2):
            self.packages[i].on_board = True

        self.charge_stations = [ChargeStation(p) for p in self.random_cells(2)]

    def clone(self):
        cloned = WarehouseEnv()
        cloned.num_steps = self.num_steps
        cloned.seed = self.seed
        cloned.robots = [copy(t) for t in self.robots]
        cloned.packages = [copy(p) for p in self.packages]
        cloned.charge_stations = [copy(g) for g in self.charge_stations]
        return cloned

    def random_cells(self, count: int):
        random.seed(self.seed)
        self.seed = random.randint(0, 255)
        return random.sample([(x, y) for x in range(board_size) for y in range(board_size)], count)

    def get_robot(self, robot_id):
        return self.robots[robot_id]

    def get_robot_in(self, position):
        robots = [robot for robot in self.robots if robot.position == position]
        if len(robots) == 0:
            return None
        else:
            return robots[0]

    def get_charge_station_in(self, position):
        charge_stations = [charge_station for charge_station in self.charge_stations if
                           charge_station.position == position]
        if len(charge_stations) == 0:
            return None
        return charge_stations[0]

    def get_package_in(self, position):
        packages = [package for package in self.packages[0:2] if package.position == position]
        if len(packages) == 0:
            return None
        return packages[0]

    def get_legal_operators(self, robot_index: int):
        ops = []
        robot = self.robots[robot_index]
        robot_pos = robot.position
        if robot.battery > 0:
            for op_move, op_disp in [('move north', (0, -1)), ('move south', (0, 1)),
                                     ('move west', (-1, 0)), ('move east', (1, 0))]:
                new_pos = (robot_pos[0] + op_disp[0], robot_pos[1] + op_disp[1])
                if board_size > new_pos[0] >= 0 and board_size > new_pos[1] >= 0 \
                        and self.get_robot_in(new_pos) is None:
                    ops.append(op_move)
        else:
            ops.append('park')
        if self.get_charge_station_in(robot_pos) and robot.credit > 0:
            ops.append("charge")
        if robot.package is not None and robot.package.destination == robot_pos:
            ops.append("drop off")
        package = self.get_package_in(robot.position)
        if robot.package is None and package is not None and package.on_board:
            ops.append("pick up")
        return ops

    def move_robot(self, robot_index: int, offset):
        p = self.robots[robot_index].position
        self.robots[robot_index].position = p[0] + offset[0], p[1] + offset[1]
        self.robots[robot_index].battery -= 1

    def spawn_package(self):
        ps = self.random_cells(2)
        return self.packages.append(Package(ps[0], ps[1]))

    def apply_operator(self, robot_index: int, operator: str):
        self.num_steps -= 1
        robot = self.robots[robot_index]
        assert operator in self.get_legal_operators(robot_index)
        assert not self.num_steps < 0
        if operator == 'park':
            pass
        elif operator == 'move north':
            self.move_robot(robot_index, (0, -1))
        elif operator == 'move south':
            self.move_robot(robot_index, (0, 1))
        elif operator == 'move east':
            self.move_robot(robot_index, (1, 0))
        elif operator == 'move west':
            self.move_robot(robot_index, (-1, 0))
        elif operator == 'pick up':
            package = self.get_package_in(robot.position)
            self.robots[robot_index].package = package
            self.packages.remove(package)
        elif operator == 'charge':
            robot.battery += robot.credit
            robot.credit = 0
        elif operator == 'drop off':
            robot.credit += manhattan_distance(robot.package.position, robot.package.destination) * 2
            self.spawn_package()
            if not self.packages[0].on_board:
                self.packages[0].on_board = True
            else:
                self.packages[1].on_board = True

            robot.package = None
        else:
            assert False

    def done(self):
        return len([robot for robot in self.robots if robot.battery > 0]) == 0 or self.num_steps <= 0

    def get_balances(self):
        return [t.credit for t in self.robots]

    def robot_is_occupied(self, robot_index):
        return self.robots[robot_index].package is not None

    def print(self):
        for y in range(board_size):
            for x in range(board_size):
                p = (x, y)
                robot = self.get_robot_in(p)
                package = self.get_package_in(p)
                charge_station = self.get_charge_station_in(p)
                package_destination = [package for package in self.packages[0:2] if package.destination == p and package.on_board]
                robot_package_destination = [i for i, robot in enumerate(self.robots) if robot.package is not None
                                             and robot.package.destination == p]
                if robot:
                    print('[R' + str(self.robots.index(robot)) + ']', end='')
                elif package and package.on_board:
                    print('[P' + str(self.packages[0:2].index(package)) + ']', end='')
                elif charge_station:
                    print('[C' + str(self.charge_stations.index(charge_station)) + ']', end='')
                elif len(package_destination) > 0:
                    print('[D' + str(self.packages[0:2].index(package_destination[0])) + ']', end='')
                elif len(robot_package_destination) > 0:
                    print('[X' + str(robot_package_destination[0]) + ']', end='')
                else:
                    print('[  ]', end='')
            print('')
        print('robots: ', self.robots)
        print('packages on street: ', self.packages)

        for package in self.packages:
            print(package.position, package.destination, package.on_board)

        print('charge stations: ', self.charge_stations)

    def _pygame_print_robot_data(self, robot, canvas, robot_index):

        robot_pos = robot.position
        robot_pos_txt = 'position: ' + str(robot_pos)
        robot_battery = robot.battery
        robot_battery_txt = 'battery: ' + str(robot_battery)
        robot_credit = robot.credit
        robot_credit_txt = 'credit: ' + str(robot_credit)
        robot_package = robot.package

        font = pygame.font.Font('freesansbold.ttf', 16)
        black = (0, 0, 0)
        red = (255, 0, 0)
        blue = (0, 0, 255)

        if robot_index == 0:
            canvas.blit(pygame.transform.scale(self.blue_robot_icon, (95, 95)), (95, 80))
            text = font.render(robot_pos_txt, True, black)
            canvas.blit(text, (185, 95))
            text = font.render(robot_battery_txt, True, black)
            canvas.blit(text, (185, 115))
            text = font.render(robot_credit_txt, True, black)
            canvas.blit(text, (185, 135))
            if robot_package is not None:
                robot_package_txt = 'package: ' + str(robot_package.position) + ' -> ' + str(robot_package.destination)
                text = font.render(robot_package_txt, True, black)
                canvas.blit(text, (185, 155))
        else:
            canvas.blit(pygame.transform.scale(self.red_robot_icon, (86, 86)), (355, 80))
            text = font.render(robot_pos_txt, True, black)
            canvas.blit(text, (445, 95))
            text = font.render(robot_battery_txt, True, black)
            canvas.blit(text, (445, 115))
            text = font.render(robot_credit_txt, True, black)
            canvas.blit(text, (445, 135))
            if robot_package is not None:
                robot_package_txt = 'package: ' + str(robot_package.position) + ' -> ' + str(robot_package.destination)
                text = font.render(robot_package_txt, True, black)
                canvas.blit(text, (445, 155))

    def pygame_print(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((720, 720))
            self.clock = pygame.time.Clock()
            self.blue_robot_icon = pygame.image.load("icons/robot_b.jpeg").convert()
            self.red_robot_icon = pygame.image.load("icons/robot_r.jpeg").convert()
            self.blue_robot_package_icon = pygame.image.load("icons/robot_b_package.jpeg").convert()
            self.red_robot_package_icon = pygame.image.load("icons/robot_r_package.jpeg").convert()
            self.charge_stations_icon = pygame.image.load("icons/charge_station.jpeg").convert()
            self.package_1_icon = pygame.image.load("icons/package_1.jpeg").convert()
            self.package_2_icon = pygame.image.load("icons/package_2.jpeg").convert()
            self.dest_1 = pygame.image.load("icons/dest_1.jpeg").convert()
            self.dest_2 = pygame.image.load("icons/dest_2.jpeg").convert()
            self.dest_red = pygame.image.load("icons/dest_red.jpeg").convert()
            self.dest_blue = pygame.image.load("icons/dest_blue.jpeg").convert()



        canvas = pygame.Surface((720, 720))
        canvas.fill((255, 255, 255))

        # print robot's data
        for robot_index in range(len(self.robots)):
            self._pygame_print_robot_data(self.robots[robot_index], canvas, robot_index)

        # print title
        font = pygame.font.Font('freesansbold.ttf', 28)
        black = (0, 0, 0)
        text = font.render('AI warehouse', True, black)
        canvas.blit(text, (215, 20))

        for x in range(6):
            pygame.draw.line(
                canvas,
                0,
                (110, x * 100 + 190),
                (610, x * 100 + 190),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (x * 100 + 110, 190),
                (x * 100 + 110, 690),
                width=3,
            )

        # print board elements
        for y in range(board_size):
            for x in range(board_size):
                p = (x, y)
                robot = self.get_robot_in(p)
                package = self.get_package_in(p)
                charge_station = self.get_charge_station_in(p)
                package_destination = [package for package in self.packages if package.destination == p and package.on_board]
                robot_package_destination = [i for i, robot in enumerate(self.robots) if
                                             robot.package is not None
                                             and robot.package.destination == p]
                if robot:
                    if self.robots.index(robot) == 0:
                        if robot.package is not None:
                            canvas.blit(pygame.transform.scale(self.blue_robot_package_icon, (95, 95)),
                                        (x * 100 + 112, y * 100 + 192))
                        else:
                            canvas.blit(pygame.transform.scale(self.blue_robot_icon, (95, 95)),
                                        (x * 100 + 112, y * 100 + 192))
                    else:
                        if robot.package is not None:
                            canvas.blit(pygame.transform.scale(self.red_robot_package_icon, (95, 95)),
                                    (x * 100 + 112, y * 100 + 192))
                        else:
                            canvas.blit(pygame.transform.scale(self.red_robot_icon, (95, 95)),
                                        (x * 100 + 112, y * 100 + 192))
                elif charge_station:
                    canvas.blit(pygame.transform.scale(self.charge_stations_icon, (80, 80)),
                                (x * 100 + 120, y * 100 + 200))

                elif package and package.on_board:
                    if self.packages.index(package) == 0:
                        canvas.blit(pygame.transform.scale(self.package_1_icon, (80, 80)),
                                    (x * 100 + 120, y * 100 + 200))
                    else:
                        canvas.blit(pygame.transform.scale(self.package_2_icon, (80, 80)),
                                    (x * 100 + 120, y * 100 + 200))

                elif len(package_destination) > 0:
                    if self.packages.index(package_destination[0]) == 0:
                        canvas.blit(pygame.transform.scale(self.dest_1, (80, 80)),
                                    (x * 100 + 120, y * 100 + 200))
                    else:
                        canvas.blit(pygame.transform.scale(self.dest_2, (80, 80)),
                                    (x * 100 + 120, y * 100 + 200))
                elif len(robot_package_destination) > 0:
                    if robot_package_destination[0] == 0:
                        canvas.blit(pygame.transform.scale(self.dest_blue, (80, 80)),
                                    (x * 100 + 120, y * 100 + 200))
                    else:
                        canvas.blit(pygame.transform.scale(self.dest_red, (80, 80)),
                                    (x * 100 + 120, y * 100 + 200))



        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        self.clock.tick(5)
        # time.sleep(1)
