"""
Intermediate Tangram Puzzle Environment (Polytope-based)
=========================================================
2 squares + 2 triangles (SE and NE types) on a ~10×10 grid.
4 pieces total — harder than Easy (4 squares) due to mixed geometry,
simpler than Hard (6 pieces).

Target silhouette (L-shape):
  Piece 0  Square  (3, 3)         Red
  Piece 1  Square  (3, 5)         Orange
  Piece 2  SE tri  (3, 7)  →  (3,7),(5,7),(3,9)   Green
  Piece 3  NE tri  (5, 3)  →  (5,3),(5,5),(7,5)   Blue

Starting positions (corners):
  Piece 0  Square  (0, 0)
  Piece 1  Square  (8, 0)
  Piece 2  SE tri  (0, 8)  →  (0,8),(2,8),(0,10)
  Piece 3  NE tri  (8, 7)  →  (8,7),(8,9),(10,9)

Observation (dict):
  h_rep : float32 [4, 5, 3]  — up to 5 constraints × 3 params per piece
  v_rep : float32 [4, 4, 2]  — up to 4 vertices × 2 coords per piece
  adj   : float32 [4, 5, 5]  — constraint adjacency per piece

Action space: Discrete(16) — 4 pieces × 4 directions (up/down/left/right)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ppl
from ppl import Variable, C_Polyhedron, Constraint_System
import gym
from gym import spaces
import os

BOARD_MAX = 11.0   # normalisation constant for b-values
MAX_STEPS = 300

# Baseline-encoder constants (flat MLP / CNN strawmen — see MLPRL.py / CNNRL.py)
NUM_PIECES     = 4
POSE_DIM       = 6                       # per piece: cx, cy, shape_id, orient_id, target_dx, target_dy
FLAT_POSE_DIM  = NUM_PIECES * POSE_DIM   # 24
GRID           = 13                      # occupancy grid side (board coords are bounded to [0, 12])
GRID_CHANNELS  = 2 * NUM_PIECES + 1      # per-piece occupancy + per-piece target + locked mask
# Square=0, triangle orientations SE/SW/NE/NW=1..4 (squares get orientation_id=0)
_ORIENTATION_ID = {None: 0, "SE": 1, "SW": 2, "NE": 3, "NW": 4}

# Cached cell-center coordinates for vectorized rasterization (avoids a
# per-cell, per-constraint PPL call — that O(GRID^2 * constraints) Python
# loop is the dominant cost of computing grid_image on every env step).
_CX, _CY = np.meshgrid(np.arange(GRID) + 0.5, np.arange(GRID) + 0.5, indexing='ij')


class IntermediateTangramEnv:
    """Inner physics / geometry layer."""

    def __init__(self):
        self.x, self.y = Variable(0), Variable(1)

        self.target_pieces = [
            self._create_square(3, 3),          # P0 Red
            self._create_square(3, 5),          # P1 Orange
            self._create_triangle(3, 7, "SE"),  # P2 Green
            self._create_triangle(5, 3, "NE"),  # P3 Blue
        ]

        self.target_areas    = [self._calculate_area(tp) for tp in self.target_pieces]
        self.target_centroids = [self._poly_centroid(tp) for tp in self.target_pieces]
        self.reset()

    # ── Polytope constructors ─────────────────────────────────────────────────
    def _create_square(self, x, y):
        cs = Constraint_System()
        cs.insert(self.x >= x);     cs.insert(self.x <= x + 2)
        cs.insert(self.y >= y);     cs.insert(self.y <= y + 2)
        return C_Polyhedron(cs)

    def _create_triangle(self, x, y, type="NW"):
        cs = Constraint_System()
        cs.insert(self.x >= x);     cs.insert(self.x <= x + 2)
        cs.insert(self.y >= y);     cs.insert(self.y <= y + 2)
        if type == "SE": cs.insert(self.y <= -self.x + (x + y + 2))
        if type == "SW": cs.insert(self.y <= self.x  + (y - x + 2))
        if type == "NE": cs.insert(self.y >= self.x  + (y - x))
        if type == "NW": cs.insert(self.y >= -self.x + (x + y))
        return C_Polyhedron(cs)

    # ── Geometry helpers ──────────────────────────────────────────────────────
    def _poly_centroid(self, poly):
        verts = [(float(g.coefficient(self.x)), float(g.coefficient(self.y)))
                 for g in poly.generators() if g.is_point()]
        return np.array(verts).mean(axis=0) if verts else np.zeros(2)

    def _calculate_area(self, poly):
        if poly.is_empty():
            return 0.0
        verts = [(float(g.coefficient(self.x)), float(g.coefficient(self.y)))
                 for g in poly.generators() if g.is_point()]
        if len(verts) < 3:
            return 0.0
        centroid = np.mean(verts, axis=0)
        verts.sort(key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))
        xs, ys = zip(*verts)
        return 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

    def _is_in_target(self, piece_idx):
        overlap = C_Polyhedron(self.pieces[piece_idx])
        overlap.intersection_assign(self.target_pieces[piece_idx])
        if overlap.is_empty():
            return False
        return self._calculate_area(overlap) / self.target_areas[piece_idx] >= 0.99

    # ── State ─────────────────────────────────────────────────────────────────
    def reset(self):
        self.pieces = [
            self._create_square(0, 0),          # P0
            self._create_square(8, 0),          # P1
            self._create_triangle(0, 8, "SE"),  # P2
            self._create_triangle(8, 7, "NE"),  # P3
        ]
        self.locked = [False] * 4
        # Static per-piece metadata for the flat-MLP baseline
        self.piece_shape_id = [0, 0, 1, 1]
        self.piece_orientation_id = [
            _ORIENTATION_ID[None], _ORIENTATION_ID[None],
            _ORIENTATION_ID["SE"], _ORIENTATION_ID["NE"],
        ]

    # ── Step ──────────────────────────────────────────────────────────────────
    def move_piece(self, piece_idx, dx, dy):
        if self.locked[piece_idx]:
            return 1.0, "Locked"

        new_poly = C_Polyhedron(self.pieces[piece_idx])
        new_poly.affine_image(self.x, self.x + dx)
        new_poly.affine_image(self.y, self.y + dy)
        self.pieces[piece_idx] = new_poly

        overlap = C_Polyhedron(self.pieces[piece_idx])
        overlap.intersection_assign(self.target_pieces[piece_idx])
        normalized_overlap = self._calculate_area(overlap) / self.target_areas[piece_idx]

        if self._is_in_target(piece_idx):
            self.locked[piece_idx] = True

        return normalized_overlap, f"Success: {normalized_overlap:.2f}"

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self, save_path):
        if not os.path.splitext(save_path)[1]:
            save_path += ".png"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(os.path.basename(save_path))
        for tp in self.target_pieces:
            self._plot_poly(ax, tp, color="gray", alpha=0.1, linestyle="--")
        colors = ["#FF5733", "#FFBD33", "#33FF57", "#3357FF"]
        for i, p in enumerate(self.pieces):
            c = "#27ae60" if self.locked[i] else colors[i]
            self._plot_poly(ax, p, color=c, alpha=0.6)
        plt.xlim(-1, 13); plt.ylim(-1, 13)
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    def _plot_poly(self, ax, poly, **kwargs):
        verts = [(float(g.coefficient(self.x)), float(g.coefficient(self.y)))
                 for g in poly.generators() if g.is_point()]
        if len(verts) >= 3:
            centroid = np.mean(verts, axis=0)
            verts.sort(key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))
            ax.add_patch(patches.Polygon(verts, **kwargs))


# ── Gym wrapper ───────────────────────────────────────────────────────────────

class IntermediateTangramGym(gym.Env):
    def __init__(self):
        super().__init__()
        self.inner      = IntermediateTangramEnv()
        self.num_pieces = 4
        self.max_steps  = MAX_STEPS
        self.step_count = 0
        self.gamma      = 0.99

        self.action_space = spaces.Discrete(self.num_pieces * 4)   # 16

        # h_rep: 5 constraints (triangles need 5), v_rep: 4 vertices
        self.observation_space = spaces.Dict({
            "h_rep": spaces.Box(low=-1, high=1, shape=(4, 5, 3), dtype=np.float32),
            "v_rep": spaces.Box(low=0,  high=1, shape=(4, 4, 2), dtype=np.float32),
            "adj"  : spaces.Box(low=0,  high=1, shape=(4, 5, 5), dtype=np.float32),
            "flat_pose"  : spaces.Box(low=-1, high=1, shape=(FLAT_POSE_DIM,), dtype=np.float32),
            "grid_image" : spaces.Box(low=0,  high=1, shape=(GRID_CHANNELS, GRID, GRID), dtype=np.float32),
        })
        self._target_channels = self._rasterize_targets()

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_obs(self):
        return {
            "h_rep": self._extract_h_rep(),
            "v_rep": self._extract_v_rep(),
            "adj"  : self._build_graph_adj(),
            "flat_pose"  : self._extract_flat_pose(),
            "grid_image" : self._extract_grid_image(),
        }

    # ── Flat-MLP baseline observation ─────────────────────────────────────────
    def _extract_flat_pose(self):
        """Per-piece [cx, cy, shape_id, orientation_id, target_dx, target_dy], flattened."""
        pose = []
        for i, piece in enumerate(self.inner.pieces):
            c  = self.inner._poly_centroid(piece)
            tc = self.inner.target_centroids[i]
            pose.extend([
                c[0] / BOARD_MAX,
                c[1] / BOARD_MAX,
                float(self.inner.piece_shape_id[i]),
                self.inner.piece_orientation_id[i] / 4.0,
                (tc[0] - c[0]) / BOARD_MAX,
                (tc[1] - c[1]) / BOARD_MAX,
            ])
        return np.array(pose, dtype=np.float32)

    # ── CNN baseline observation ──────────────────────────────────────────────
    def _rasterize(self, poly):
        """Boolean occupancy grid [GRID, GRID] for a single PPL polyhedron.

        Vectorized: each constraint's PPL coefficients are read once, then
        evaluated over the whole cell-center grid with numpy — not per cell.
        """
        X, Y = self.inner.x, self.inner.y
        mask = np.ones((GRID, GRID), dtype=bool)
        for c in poly.minimized_constraints():
            a1 = float(c.coefficient(X))
            a2 = float(c.coefficient(Y))
            b  = float(c.inhomogeneous_term())
            mask &= (a1 * _CX + a2 * _CY + b >= 0)
        return mask.astype(np.float32)

    def _rasterize_targets(self):
        """Target-silhouette channels are static — computed once and cached."""
        return np.stack([self._rasterize(tp) for tp in self.inner.target_pieces])

    def _extract_grid_image(self):
        n = self.num_pieces
        img = np.zeros((GRID_CHANNELS, GRID, GRID), dtype=np.float32)
        for i, piece in enumerate(self.inner.pieces):
            img[i] = self._rasterize(piece)
        img[n:2 * n] = self._target_channels
        for i in range(n):
            if self.inner.locked[i]:
                img[2 * n] = np.maximum(img[2 * n], img[i])
        return img

    def _extract_h_rep(self):
        h_rep = []
        for p in self.inner.pieces:
            rows = []
            for c in p.minimized_constraints():
                a1   = -float(c.coefficient(self.inner.x))
                a2   = -float(c.coefficient(self.inner.y))
                b    = float(c.inhomogeneous_term())
                norm = np.sqrt(a1**2 + a2**2) if (a1**2 + a2**2) > 0 else 1.0
                rows.append([a1 / norm, a2 / norm, (b / norm) / BOARD_MAX])
            while len(rows) < 5:
                rows.append([0.0, 0.0, 0.0])
            h_rep.append(rows[:5])
        return np.array(h_rep, dtype=np.float32)

    def _extract_v_rep(self):
        v_rep = []
        for p in self.inner.pieces:
            verts = []
            for g in p.generators():
                if g.is_point():
                    verts.append([
                        float(g.coefficient(self.inner.x)) / BOARD_MAX,
                        float(g.coefficient(self.inner.y)) / BOARD_MAX,
                    ])
            while len(verts) < 4:
                verts.append([0.0, 0.0])
            v_rep.append(verts[:4])
        return np.array(v_rep, dtype=np.float32)

    def _build_graph_adj(self):
        all_adj = []
        eps     = 1e-5
        for p in self.inner.pieces:
            constraints = list(p.minimized_constraints())
            vertices    = [g for g in p.generators() if g.is_point()]
            num_c       = min(len(constraints), 5)
            adj         = np.zeros((5, 5), dtype=np.float32)
            for i in range(num_c):
                for j in range(i + 1, num_c):
                    for v in vertices:
                        if (abs(self._eval_c(constraints[i], v)) < eps and
                                abs(self._eval_c(constraints[j], v)) < eps):
                            adj[i, j] = adj[j, i] = 1.0
                            break
            all_adj.append(adj)
        return np.array(all_adj, dtype=np.float32)

    def _eval_c(self, constraint, vertex):
        x_val = float(vertex.coefficient(self.inner.x)) / vertex.divisor()
        y_val = float(vertex.coefficient(self.inner.y)) / vertex.divisor()
        a1    = -float(constraint.coefficient(self.inner.x))
        a2    = -float(constraint.coefficient(self.inner.y))
        b     = float(constraint.inhomogeneous_term())
        return a1 * x_val + a2 * y_val + b

    # ── Potential shaping ─────────────────────────────────────────────────────
    def _potential(self):
        total = 0.0
        for i in range(self.num_pieces):
            if self.inner.locked[i]:
                continue
            c      = self.inner._poly_centroid(self.inner.pieces[i])
            total += np.linalg.norm(c - self.inner.target_centroids[i])
        return -total / self.num_pieces

    # ── Gym API ───────────────────────────────────────────────────────────────
    def step(self, action):
        self.step_count += 1
        piece_idx = action // 4
        direction = action  % 4
        dx, dy    = [(0, 1), (0, -1), (-1, 0), (1, 0)][direction]

        phi_before    = self._potential() / 10.0
        locked_before = list(self.inner.locked)

        self.inner.move_piece(piece_idx, dx, dy)

        phi_after = self._potential() / 10.0

        reward  = -0.01
        reward += self.gamma * phi_after - phi_before
        for i in range(self.num_pieces):
            if self.inner.locked[i] and not locked_before[i]:
                reward += 1.0

        done = False
        if all(self.inner.locked):
            reward += 10.0
            done    = True
        elif self.step_count >= self.max_steps:
            done = True

        completion = sum(self.inner.locked) / self.num_pieces
        return self._get_obs(), reward, done, {"completion": completion}

    def reset(self):
        self.step_count = 0
        self.inner.reset()
        return self._get_obs()

    def get_action_mask(self):
        """Blocks locked pieces and moves that push any vertex outside [0, 12]."""
        mask = np.ones(self.action_space.n, dtype=bool)
        for action in range(self.action_space.n):
            piece_idx = action // 4
            if self.inner.locked[piece_idx]:
                mask[action] = False
                continue
            direction = action % 4
            dx, dy    = [(0, 1), (0, -1), (-1, 0), (1, 0)][direction]

            new_poly = C_Polyhedron(self.inner.pieces[piece_idx])
            new_poly.affine_image(self.inner.x, self.inner.x + dx)
            new_poly.affine_image(self.inner.y, self.inner.y + dy)

            for g in new_poly.generators():
                if g.is_point():
                    x_val = float(g.coefficient(self.inner.x))
                    y_val = float(g.coefficient(self.inner.y))
                    if not (0 <= x_val <= 12 and 0 <= y_val <= 12):
                        mask[action] = False
                        break
        return mask
