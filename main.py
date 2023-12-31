from __future__ import annotations

import bisect
import functools
import itertools
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import pygame

'''
32x32 board, we assume green pieces up top and red pieces on the bottom.
'''

SCALE = 30
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
BACKGROUND_YELLOW = (200, 200, 0)
BACKGROUND_RED = (200, 0, 0)
BACKGROUND_GREEN = (0, 200, 0)
DOT_RADIUS = .75*SCALE
DOT_BORDER_THICKNESS = .15*SCALE
SHOW_PREVIOUS_MOVE = True
SHOW_BEST_MOVES = True
LINE_THICKNESS = .1*SCALE
# to make a perfect hexagon we need this magic number. DO NOT TOUCH IT!
XSCALE = 1.16

class Piece(Enum):
    EMPTY = 0
    RED = 1
    GREEN = 2

@dataclass
class Node:
    x: float # between -16 and 16
    y: float # between -16 and 16
    neighbours: set[Node] = field(default_factory=set)
    uid: int = field(default_factory=itertools.count().__next__)
    piece: Piece = Piece.EMPTY

    def __repr__(self):
        return str(self.uid)

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return self.uid

    def pos(self):
        '''
        Convert our internal coordinates (-16 to 16 on X and Y) to pygame
        coordinates we can draw to.
        '''
        return (self.x+16)*SCALE + SCALE, (self.y+16)*SCALE + SCALE

    def rotate(self, rad: float):
        '''
        Rotate our X and Y by `rad` radians around origin at (0,0).
        '''
        ox, oy = self.x, self.y
        self.x = ox*math.cos(rad) - oy*math.sin(rad)
        self.y = ox*math.sin(rad) + oy*math.cos(rad)

    def connect(self, n: Node):
        n.neighbours.add(self)
        self.neighbours.add(n)

    def cleanup(self):
        '''
        Deletes edges. Should be called when node is removed from graph,
        otherwise ghost edges remain and very bad things happen.
        '''
        for n in self.neighbours:
            n.neighbours.remove(self)
        self.neighbours.clear()

def triangle() -> list[list[Node]]:
    '''
    Produces a triangle with a height of 5 nodes. All nodes are connected to
    nodes in their immediate vicinity.
    '''
    layers = [[Node((x*2 - y if y%2 == 0 else x*2 - y) * XSCALE, y*2 - 16) for x in range(y+1)] for y in range(5)]
    for y, layer in enumerate(layers):
        # connect bottom
        if y != len(layers)-1:
            for x in range(len(layer)):
                layer[x].connect(layers[y+1][x])
                layer[x].connect(layers[y+1][x+1])

        # connect sideways
        for x in range(len(layer)-1):
            layer[x].connect(layer[x+1])
    return layers

def double_triangle() -> list[list[Node]]:
    '''
    Produces a triangle stitched together at the bases with another triangle
    flipped upside down.
    '''
    t1 = triangle()
    t2 = triangle()

    # remove last and widest layer
    for node in t2[-1]:
        node.cleanup()
    t2 = t2[:-1]

    # connect the 4-node last layer of t2 with the 5-node last layer of t1.
    for n, a, b in zip(t2[-1], t1[-1], t1[-1][1:]):
        n.connect(a)
        n.connect(b)

    # flip the triangle upside down.
    for layer in t2:
        for node in layer:
            node.y = 16-node.y - 32

    return t1 + list(reversed(t2))

def board() -> tuple[list[Node], list[Node]]:
    '''
    Produces the hexagon board. Returns a list of the top of each of the 6
    triangles and also just a list of all (around 180) nodes.
    '''
    triangles = [double_triangle() for _ in range(6)]

    for i in range(2, 6):
        triangles[0][-i][0].cleanup()
        triangles[0][-i].pop(0)
    for i in range(2, 6):
        triangles[-1][-i][-1].connect(triangles[0][-i][0])
        triangles[-1][-i][-1].connect(triangles[0][-i-1][0])

    for t1, t2 in zip(triangles, triangles[1:]):
        for i in range(1, 6):
            t2[-i][0].cleanup()
            t2[-i].pop(0)
        for i in range(2, 6):
            t1[-i][-1].connect(t2[-i][0])
            t1[-i][-1].connect(t2[-i-1][0])

    for i in range(1, 6):
        triangles[0][-1][-1].connect(triangles[i][-2][0])

    for i, t in enumerate(triangles):
        for layer in t:
            for node in layer:
                node.rotate(math.radians(60*i))
    new = []
    for t in triangles:
        for l in t:
            for n in l:
                new.append(n)
    return ([t[0][0] for t in triangles], new)


def draw_lines(node: Node, seen: set[Node]|None=None):
    '''
    Traverse all nodes with DFS, drawing lines representing the
    connections/edges.
    '''
    if seen is None:
        seen = set()
    seen.add(node)
    for n in node.neighbours:
        pygame.draw.line(screen, BLACK, node.pos(), n.pos(), int(LINE_THICKNESS))
        if n not in seen:
            draw_lines(n, seen)

def draw_dots(node: Node, seen: set[Node]|None=None):
    '''
    Traverse all nodes with DFS, drawing dots representing the nodes.
    '''
    if seen is None:
        seen = set()
    seen.add(node)

    # draw the inner dot.
    color = WHITE
    if node.piece != Piece.EMPTY:
        color = {
            Piece.RED: RED,
            Piece.GREEN: GREEN,
        }[node.piece]
        if node.piece != turn.get():
            color = (
                color[0] / 1.2,
                color[1] / 1.2,
                color[2] / 1.2,
            )
    radius = DOT_RADIUS
    if node == hovered:
        radius *= 1.3
    elif node == selected:
        radius *= 1.3
    elif can_move_to(node):
        radius *= 1.3
    pygame.draw.circle(screen, color, node.pos(), radius)

    # draw the outline.
    thickness = DOT_BORDER_THICKNESS
    color = BLACK
    if node == selected:
        thickness *= 1.4
    elif can_move_to(node):
        thickness *= 2
    pygame.draw.circle(screen, color, node.pos(), radius, int(thickness))

    for n in node.neighbours:
        if n not in seen:
            draw_dots(n, seen)

def setup(head: Node, piece: Piece):
    '''
    Starting at the top of a triangle, with BFS we move exactly 5 nodes down
    and set the 15 nodes we encounter to the specified piece.
    '''
    visited = set([head])
    q = deque()
    q.appendleft(head)
    while q:
        node = q.pop()
        # overriding an already existing piece probably isn't intended.
        if node.piece != Piece.EMPTY:
            raise ValueError('you are overriding a piece.')
        node.piece = piece
        if len(visited) < 15:
            for n in node.neighbours:
                if n not in visited:
                    q.appendleft(n)
                    visited.add(n)

def possible_paths(src: Node) -> list[list[Node]]:
    '''
    Return list of all possible paths from `src`. Uses standard BFS.
    '''
    moves = []
    visited = set([src])

    # the nodes we move to in one step
    for n in src.neighbours:
        if n.piece == Piece.EMPTY:
            visited.add(n)
            moves.append([src, n])

    q = deque()
    q.appendleft([src])
    while q:
        path = q.pop()
        for n in path[-1].neighbours:
            # get all neighbours that have pieces on them.
            if n not in visited and n.piece != Piece.EMPTY:
                visited.add(n)
                # if any of these neighbours have an empty space directly in
                # front of `node`, then add it.
                for n2 in n.neighbours:
                    if n2.piece == Piece.EMPTY:
                        visited.add(n2)
                        # This seems bad. The only way we can hop is if our
                        # `node` is directly apart from `n2`, seperated by `n`.
                        # So what we do is get the actual, physical distance
                        # and if this distance falls within two nodes (aka 5)
                        # within a threshold of 0.5, then `node` is directly
                        # facing `n2` and it's considered a valid move.
                        if abs(math.sqrt((path[-1].x-n2.x)**2 + (path[-1].y-n2.y)**2) - 5) < .5:
                            moves.append(path + [n2])
                            q.appendleft(path + [n2])
    return moves

def possible_paths_optimized(src: Node) -> list[list[Node]]:
    '''
    Return list of most promising paths from `src`. Uses standard BFS. For use
    in the `minimax` function. More optimized than using `possible_paths` would
    be.
    '''
    moves = []
    visited = set([src])

    # the nodes we move to in one step
    for n in src.neighbours:
        if n.piece == Piece.EMPTY:
            visited.add(n)
            moves.append([src, n])

    q = deque()
    q.appendleft([src])
    while q:
        path = q.pop()
        for n in path[-1].neighbours:
            if n not in visited and n.piece != Piece.EMPTY:
                visited.add(n)
                for n2 in n.neighbours:
                    if n2.piece == Piece.EMPTY:
                        visited.add(n2)
                        if abs(math.sqrt((path[-1].x-n2.x)**2 + (path[-1].y-n2.y)**2) - 5) < .5:
                            bisect.insort(moves, path + [n2], key=functools.cmp_to_key(lambda a, b: len(b) - len(a)))
                            q.appendleft(path + [n2])
    #random.shuffle(moves)
    return moves

def move_piece_with_animation(src: Node, dst: Node, path):
    original = src.piece
    color = {
        Piece.RED: RED,
        Piece.GREEN: GREEN,
    }[src.piece]
    src.piece = Piece.EMPTY
    global suggested_moves
    suggested_moves = []
    for src, dst in zip(path, path[1:]):
        sx, sy = src.pos()
        dx, dy = dst.pos()
        pacex, pacey = (dx-sx)/18, (dy-sy)/18
        for i in range(18):
            draw()
            pos = (sx + pacex*i, sy + pacey*i)
            pygame.draw.circle(screen, color, pos, DOT_RADIUS*1.3)
            pygame.draw.circle(screen, BLACK, pos, DOT_RADIUS*1.3, int(DOT_BORDER_THICKNESS*1.3))
            pygame.display.flip()
            clock.tick(60)
    src.piece = original
    move_piece(src, dst)

    path, _ = minimax(heads[0], (
        pieces(heads[0], turn.get_next()),
        pieces(heads[0], turn.get())
    ), depth=3, consider=2)
    suggested_moves = [path]

def move_piece(src: Node, dest: Node):
    dest.piece = src.piece
    src.piece = Piece.EMPTY

def pieces(board: Node, color: Piece) -> set[Node]:
    '''
    Get list of nodes of a certain color with BFS.
    '''
    visited = set([board])
    pieces = set()
    q = deque()
    q.appendleft(board)
    while q:
        node = q.pop()
        if node.piece == color:
            pieces.add(node)
        for n in node.neighbours:
            if n not in visited:
                q.appendleft(n)
                visited.add(n)
    return pieces

def distance_from_target(n: Node, home: Piece):
    # this function relies on the fact that the red pieces' home are at the
    # bottom and the greens' at the top.
    return abs(n.y + 16) if home == Piece.RED else abs(n.y - 16)

class countcalls(object):
   def __init__(self, f):
       self.f = f
       self.calls = 0

   def __call__(self, *args, **kwargs):
       self.calls += 1
       return self.f(*args, **kwargs)

@countcalls
def minimax(
    board: Node,
    pieces: tuple[list[Node],list[Node]],
    depth=4,
    consider=4,
    maximizing=True,
    alpha=-10*100,
    beta=10*100
) -> tuple[list[Node],int]:
    '''
    A simple implementation of minimax with alpha-beta pruning. `pieces[0]`
    should be a list of all the same-colored nodes on the board that we shall
    maximize for.
    '''
    if depth == 0:
        # the farther our opponent is from the target the better
        res = sum(distance_from_target(n, n.piece) for n in pieces[1])
        # we penalize according to how far we are from the target.
        res -= sum(distance_from_target(n, n.piece) for n in pieces[0])
        # favour the center of the board instead of valuing the fringes
        for n in pieces[0]:
            if abs(n.x) > 8:
                res -= abs(n.x)
        return [], res

    score = -10**100 if maximizing else 10**100
    bestpath = []

    for n in pieces[int(not maximizing)]:
        # only consider the most promising paths
        for path in possible_paths_optimized(n)[:consider]:
            src, dst = path[0], path[-1]

            # do not consider the moves that move the piece back.
            if distance_from_target(dst, dst.piece) > distance_from_target(src, src.piece):
                continue
            move_piece(src, dst)

            pieces[int(not maximizing)].remove(src)
            pieces[int(not maximizing)].add(dst)
            *_, newscore = minimax(board, pieces, depth-1, consider, not maximizing, alpha, beta)
            pieces[int(not maximizing)].remove(dst)
            pieces[int(not maximizing)].add(src)

            move_piece(dst, src)

            if maximizing and newscore > score:
                score = newscore
                bestpath = path
                alpha = max(alpha, newscore)
            elif not maximizing and newscore < score:
                score = newscore
                bestpath = path
                beta = min(beta, newscore)

            if beta <= alpha:
                return bestpath, score
    return bestpath, score

@dataclass
class Turn:
    data: Piece

    def get(self):
        return self.data

    def get_next(self):
        return Piece.GREEN if self.data == Piece.RED else Piece.RED

    def next(self):
        self.data = self.get_next()

heads, nodes = board()
running = True
turn = Turn(Piece.GREEN)
history: list[list[Node]] = []
suggested_moves: list[list[Node]] = []
selected: Node|None = None
hovered: Node|None = None
highlight: list[list[Node]] = []
setup(heads[0], Piece.GREEN)
setup(heads[3], Piece.RED)

pygame.init()
screen = pygame.display.set_mode((34*SCALE, 34*SCALE))
clock = pygame.time.Clock()

def collide() -> Node|None:
    for n in nodes:
        sx, sy = n.pos()
        dx, dy = pygame.mouse.get_pos()
        if math.sqrt((sx-dx)**2 + (sy-dy)**2) <= DOT_RADIUS*1.4:
            return n
    return None

def deselect():
    global highlight, selected
    highlight = []
    selected = None

def can_select(n: Node):
    return n.piece == turn.get()

def can_move_to(n: Node):
    return n in [path[-1] for path in highlight]

def draw():
    screen.fill((255, 255, 255))

    # I know magic numbers are bad, but these happen to work from trial and error.
    pygame.draw.polygon(screen, BACKGROUND_GREEN, ((17*SCALE, SCALE), (21.7*SCALE, 9*SCALE), (12.3*SCALE, 9*SCALE)))
    pygame.draw.polygon(screen, BACKGROUND_GREEN, ((17*SCALE, 33*SCALE), (21.7*SCALE, 25*SCALE), (12.3*SCALE, 25*SCALE)))
    pygame.draw.polygon(screen, BACKGROUND_YELLOW, ((31*SCALE, 9*SCALE), (21.7*SCALE, 9*SCALE), (26.3*SCALE, 17*SCALE)))
    pygame.draw.polygon(screen, BACKGROUND_YELLOW, ((3*SCALE, 25*SCALE), (12.3*SCALE, 25*SCALE), (7.8*SCALE, 17*SCALE)))
    pygame.draw.polygon(screen, BACKGROUND_RED, ((3*SCALE, 9*SCALE), (12.3*SCALE, 9*SCALE), (7.8*SCALE, 17*SCALE)))
    pygame.draw.polygon(screen, BACKGROUND_RED, ((31*SCALE, 25*SCALE), (21.6*SCALE, 25*SCALE), (26.2*SCALE, 17*SCALE)))

    draw_lines(heads[0])
    if SHOW_PREVIOUS_MOVE and history:
        for src, dst in zip(history[-1], history[-1][1:]):
            pygame.draw.line(screen, BLACK, src.pos(), dst.pos(), int(LINE_THICKNESS*3))
    if SHOW_BEST_MOVES:
        for path in suggested_moves:
            for src, dst in zip(path, path[1:]):
                pygame.draw.line(screen, BLUE, src.pos(), dst.pos(), int(LINE_THICKNESS*3))
    draw_dots(heads[0])

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                heads, nodes = board()
                running = True
                turn = Turn(Piece.GREEN)
                history = []
                suggested_moves = []
                selected = None
                hovered = None
                highlight = []
                setup(heads[0], Piece.GREEN)
                setup(heads[3], Piece.RED)
            if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_LCTRL:
                deselect()
                if history:
                    src, *_, dst = history.pop()
                    move_piece(dst, src)
            if event.key == pygame.K_SPACE:
                deselect()
                draw()

                # darken screen
                s = pygame.Surface(pygame.display.get_surface().get_size())
                s.set_alpha(128)
                s.fill((0, 0, 0))
                screen.blit(s, (0, 0))
                pygame.display.flip()

                # use minimax
                minimax.calls = 0
                start = time.time()
                path, score = minimax(heads[0], (
                    pieces(heads[0], turn.get()),
                    pieces(heads[0], turn.get_next())
                ))
                time_elapsed = time.time() - start
                assert score != -10**100 and score != 10**100
                print(f'{minimax.calls} possibilities in {time_elapsed:.3}s (score: {score:.3}).')
                assert path

                history.append(path)
                move_piece_with_animation(path[0], path[-1], path)
                turn.next()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if (mouse_node := collide()) is None:
                deselect()
            else:
                if can_move_to(mouse_node):
                    for path in highlight:
                        if path[0] == selected and path[-1] == mouse_node:
                            history.append(path)
                            break

                    move_piece_with_animation(selected, mouse_node, history[-1])
                    deselect()
                    turn.next()
                elif can_select(mouse_node):
                    highlight = possible_paths(mouse_node)
                    selected = mouse_node
                else:
                    deselect()
        else:
            if (mouse_node := collide()) is not None and mouse_node.piece == turn.get():
                hovered = mouse_node
            else:
                hovered = None

    draw()
    pygame.display.flip()
    clock.tick(60)
pygame.quit()
