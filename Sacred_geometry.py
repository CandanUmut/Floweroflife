import pygame
import numpy as np
import math
import datetime
import os
from itertools import combinations

# —————————————————————————————————————————————————————————————————————————————
# Configuration & Constants
# —————————————————————————————————————————————————————————————————————————————
WIDTH, HEIGHT = 1000, 1000
FPS_TARGET = 60

# Sacred color palette
INDIGO  = (75,   0, 130)
GOLD    = (255, 215,   0)
EMERALD = (  0, 128,   0)
CYAN    = (  0, 255, 255)
WHITE   = (255, 255, 255)

# Controls hint
CONTROLS_HINT = "SPACE:Next Solid   M:Toggle Mode   F:+Flower   P:ProjToggle   S:Save PNG"

# —————————————————————————————————————————————————————————————————————————————
# Utility Projections
# —————————————————————————————————————————————————————————————————————————————
def project_3d(point, angle_x, angle_y, zoom=200):
    """3D perspective projection of a 3D point to 2D screen coords."""
    x, y, z = point
    # Rotate around X
    cosx, sinx = math.cos(angle_x), math.sin(angle_x)
    y, z = y * cosx - z * sinx, y * sinx + z * cosx
    # Rotate around Y
    cosy, siny = math.cos(angle_y), math.sin(angle_y)
    x, z = x * cosy + z * siny, -x * siny + z * cosy
    # Perspective divide
    factor = zoom / (z + 5)
    sx = x * factor + WIDTH // 2
    sy = -y * factor + HEIGHT // 2
    return int(sx), int(sy)

def project_2d(point, *_args, zoom=200):
    """Simple orthographic projection: ignore Z."""
    x, y, _ = point
    sx = x * zoom + WIDTH // 2
    sy = -y * zoom + HEIGHT // 2
    return int(sx), int(sy)

# —————————————————————————————————————————————————————————————————————————————
# Solid Factory
# —————————————————————————————————————————————————————————————————————————————
class Solid:
    """Generates Platonic solids and cycles through them."""
    def __init__(self):
        self._solids = [
            ("Tetrahedron", self._tetrahedron),
            ("Cube",        self._cube),
            ("Octahedron",  self._octahedron),
            ("Dodecahedron",self._dodecahedron),
            ("Icosahedron", self._icosahedron),
        ]
        self.index = 0
        self.name, self.vertices, self.edges = None, None, None
        self.next()

    def next(self):
        """Advance to the next Platonic solid."""
        self.index = (self.index + 1) % len(self._solids)
        self.name, gen = self._solids[self.index]
        self.vertices, self.edges = gen()

    def _tetrahedron(self):
        a = 1
        h = math.sqrt(6)/3
        verts = [
            (0, 0, h),
            (a/2, math.sqrt(3)/6*a, 0),
            (-a/2, math.sqrt(3)/6*a, 0),
            (0, -math.sqrt(3)/3*a, 0)
        ]
        edges = list(combinations(range(4), 2))
        return verts, edges

    def _cube(self):
        pts = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
               (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
        edges = [(i,j) for i,j in combinations(range(8),2)
                 if sum(abs(pts[i][k]-pts[j][k]) for k in range(3))==2]
        return pts, edges

    def _octahedron(self):
        verts = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        return verts, list(combinations(range(6),2))

    def _dodecahedron(self):
        phi = (1+math.sqrt(5))/2
        a, b = 1, 1/phi
        pts = []
        for x in (-a,a):
            for y in (-a,a):
                for z in (-a,a):
                    pts.append((x,y,z))
        for i in (-b,b):
            for j in (-b*phi,b*phi):
                for k in (0,):
                    pts += [(i,j,k),(i,k,j),(k,i,j)]
        edges = [(i,j) for i,j in combinations(range(len(pts)),2)
                 if np.linalg.norm(np.subtract(pts[i],pts[j]))<2.1]
        return pts, edges

    def _icosahedron(self):
        phi = (1+math.sqrt(5))/2
        pts = []
        for i in (-1,1):
            for j in (-1,1):
                pts += [(0, i*phi, j), (i, 0, j*phi), (i*phi, j, 0)]
        edges = [(i,j) for i,j in combinations(range(len(pts)),2)
                 if np.linalg.norm(np.subtract(pts[i],pts[j]))<2.1]
        return pts, edges

# —————————————————————————————————————————————————————————————————————————————
# Mandala Engine
# —————————————————————————————————————————————————————————————————————————————
class MandalaEngine:
    """Draws a rotating mandala of a given solid."""
    def __init__(self, solid: Solid, center, radius=180, copies=6, scale=60):
        self.solid = solid
        self.center = center
        self.radius = radius
        self.copies = copies
        self.scale = scale
        self.angle_off = 0.0
        self.angle_x, self.angle_y = 0.0, 0.0

    def update(self, dt):
        self.angle_off += dt * 0.5
        self.angle_x   += dt * 0.8
        self.angle_y   += dt * 0.6

    def draw(self, surf, proj):
        verts, edges = self.solid.vertices, self.solid.edges
        for i in range(self.copies):
            theta = self.angle_off + i * (2*math.pi/self.copies)
            cx = self.center[0] + self.radius * math.cos(theta)
            cy = self.center[1] + self.radius * math.sin(theta)
            for e in edges:
                p1 = tuple(self.scale * v for v in verts[e[0]])
                p2 = tuple(self.scale * v for v in verts[e[1]])
                x1,y1 = proj(p1, self.angle_x, self.angle_y)
                x2,y2 = proj(p2, self.angle_x, self.angle_y)
                pygame.draw.line(surf, CYAN, (x1+cx,y1+cy), (x2+cx,y2+cy), 1)

# —————————————————————————————————————————————————————————————————————————————
# Flower of Life Engine
# —————————————————————————————————————————————————————————————————————————————
class FlowerEngine:
    """Animates recursive Flower of Life or Seed of Life."""
    def __init__(self, center, base_radius=40):
        self.center = center
        self.base_r = base_radius
        self.layers = 1
        self._build()
        self.current = 0

    def _build(self):
        """Precompute all circle centers up to self.layers."""
        self.points = [self.center]
        for L in range(1, self.layers):
            new_pts = []
            for pt in self.points[:len(self.points)]:
                for k in range(6):
                    ang = k * (2*math.pi/6)
                    dx = self.base_r * L * math.cos(ang)
                    dy = self.base_r * L * math.sin(ang)
                    nppt = (pt[0]+dx, pt[1]+dy)
                    if nppt not in self.points and nppt not in new_pts:
                        new_pts.append(nppt)
            self.points += new_pts
        self.circles = [(pt, self.base_r) for pt in self.points]

    def update(self):
        """Reveal one more circle per frame."""
        if self.current < len(self.circles):
            self.current += 1

    def draw(self, surf, color=EMERALD):
        """Draw revealed circles."""
        for pt,r in self.circles[:self.current]:
            pygame.draw.circle(surf, color, (int(pt[0]),int(pt[1])), int(r), 1)

# —————————————————————————————————————————————————————————————————————————————
# Metatron's Cube Engine
# —————————————————————————————————————————————————————————————————————————————
class MetatronEngine:
    """Draws the Seed of Life + connecting lines (Metatron's Cube)."""
    def __init__(self, center, base_radius=40):
        self.center = center
        self.base_r = base_radius
        # 7 seed points: center + 6 around
        self.points = [center] + [
            (center[0] + base_radius * math.cos(k*2*math.pi/6),
             center[1] + base_radius * math.sin(k*2*math.pi/6))
            for k in range(6)
        ]
        self.edges = list(combinations(range(7),2))

    def draw(self, surf):
        # circles
        for pt in self.points:
            pygame.draw.circle(surf, INDIGO, (int(pt[0]),int(pt[1])), self.base_r, 1)
        # connecting lines
        for i,j in self.edges:
            p1,p2 = self.points[i], self.points[j]
            pygame.draw.line(surf, GOLD, (int(p1[0]),int(p1[1])),
                                      (int(p2[0]),int(p2[1])), 1)

# —————————————————————————————————————————————————————————————————————————————
# Torus Engine
# —————————————————————————————————————————————————————————————————————————————
class TorusEngine:
    """Approximates a 3D torus wireframe."""
    def __init__(self, center, R=1.5, r=0.5, res1=24, res2=12):
        self.center = center
        self.R, self.r = R, r
        self.res1, self.res2 = res1, res2
        self.vertices, self.edges = [], []
        self.angle_x = self.angle_y = 0.0
        self._build()

    def _build(self):
        for i in range(self.res1):
            u = 2*math.pi * i/self.res1
            for j in range(self.res2):
                v = 2*math.pi * j/self.res2
                x = (self.R + self.r*math.cos(v))*math.cos(u)
                y = (self.R + self.r*math.cos(v))*math.sin(u)
                z = self.r * math.sin(v)
                self.vertices.append((x,y,z))
        for i in range(self.res1):
            for j in range(self.res2):
                idx = i*self.res2 + j
                next_i = ((i+1)%self.res1)*self.res2 + j
                next_j = i*self.res2 + ((j+1)%self.res2)
                self.edges.append((idx, next_i))
                self.edges.append((idx, next_j))

    def update(self, dt):
        self.angle_x += dt * 0.4
        self.angle_y += dt * 0.3

    def draw(self, surf, proj):
        for i,j in self.edges:
            p1 = self.vertices[i]
            p2 = self.vertices[j]
            x1,y1 = proj(p1, self.angle_x, self.angle_y)
            x2,y2 = proj(p2, self.angle_x, self.angle_y)
            pygame.draw.line(surf, EMERALD, (x1,y1), (x2,y2), 1)

# —————————————————————————————————————————————————————————————————————————————
# UI Overlay
# —————————————————————————————————————————————————————————————————————————————
class UIOverlay:
    """Displays mode, names, values, FPS, and controls."""
    def __init__(self, font):
        self.font = font

    def draw(self, surf, info):
        lines = [
            f"Mode: {info['mode']}",
            f"Solid: {info.get('solid','-')}",
            f"Flower Layers: {info.get('layers','-')}",
            f"Mandala Copies: {info.get('copies','-')}",
            f"Projection: {info.get('proj','?')}",
            f"FPS: {info['fps']:.1f}"
        ]
        y = 10
        for L in lines:
            txt = self.font.render(L, True, WHITE)
            surf.blit(txt, (10, y))
            y += 20
        # controls hint at bottom
        hint = self.font.render(CONTROLS_HINT, True, WHITE)
        surf.blit(hint, (10, HEIGHT-30))

# —————————————————————————————————————————————————————————————————————————————
# Main Application
# —————————————————————————————————————————————————————————————————————————————
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sacred Geometry Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    # Engines & State
    solid   = Solid()
    mandala = MandalaEngine(solid, center=(0,0))
    flower  = FlowerEngine((WIDTH/2, HEIGHT/2))
    metatron= MetatronEngine((WIDTH/2, HEIGHT/2))
    torus   = TorusEngine((0,0))
    overlay = UIOverlay(font)

    mode = "mandala"  # mandala, flower, seed, metatron, torus
    projection_3d = True

    running = True
    while running:
        dt = clock.tick(FPS_TARGET) / 1000.0
        fps = clock.get_fps()

        # — Event Handling —
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    solid.next()
                elif e.key == pygame.K_m:
                    # cycle modes
                    modes = ["mandala","flower","seed","metatron","torus"]
                    mode = modes[(modes.index(mode)+1) % len(modes)]
                    if mode == "seed":
                        # reset seed’s FlowerEngine to 2 layers
                        flower.layers = 2
                        flower.current = len(flower.circles)  # show all at once
                elif e.key == pygame.K_f and mode=="flower":
                    flower.layers += 1
                    flower.current = 0
                    flower._build()
                elif e.key == pygame.K_p:
                    projection_3d = not projection_3d
                elif e.key == pygame.K_s:
                    # save screenshot
                    fname = f"sacred_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
                    pygame.image.save(screen, fname)
                    print("Saved:", fname)

        # — Update Engines —
        mandala.update(dt)
        flower.update()
        torus.update(dt)

        # — Draw Frame —
        screen.fill((5,5,20))

        if mode == "mandala":
            mandala.draw(screen, project_3d if projection_3d else project_2d)
        elif mode == "flower":
            flower.draw(screen, EMERALD)
        elif mode == "seed":
            metatron.draw(screen)
        elif mode == "metatron":
            metatron.draw(screen)
        elif mode == "torus":
            torus.draw(screen, project_3d if projection_3d else project_2d)

        # — UI Overlay —
        overlay.draw(screen, {
            "mode": mode.title(),
            "solid": solid.name,
            "layers": flower.layers,
            "copies": mandala.copies,
            "proj": "3D" if projection_3d else "2D",
            "fps": fps
        })

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
