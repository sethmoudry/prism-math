# fmt: off
"""
Known-value ground truth for geometry methods (Part 2).

Covers: coordinates, coordinates_2, coordinates_3, polygons, polygons_2,
        transformations, misc, misc_2, misc_3, misc_4.
"""
import math

KNOWN_VALUES = {
    # ========================================================================
    # COORDINATES
    # ========================================================================
    "distance": [
        # (0,0) to (3,4): sqrt(9+16)=5 -> 5
        {"input": None, "params": {"x1": 0, "y1": 0, "x2": 3, "y2": 4}, "expected": 5},
        # (1,1) to (4,5): sqrt(9+16)=5 -> 5
        {"input": None, "params": {"x1": 1, "y1": 1, "x2": 4, "y2": 5}, "expected": 5},
    ],
    "coordinate_geometry": [
        # distance: (0,0) to (3,4) = 5
        {"input": None, "params": {"operation": "distance", "p1": [0, 0], "p2": [3, 4]}, "expected": 5},
        # midpoint: (0,0)-(4,6), mid=(2,3), sum=5
        {"input": None, "params": {"operation": "midpoint", "p1": [0, 0], "p2": [4, 6]}, "expected": 5},
        # slope: (0,0) to (2,4) = slope 2.0 -> int(round(2.0*10)) = 20
        {"input": None, "params": {"operation": "slope", "p1": [0, 0], "p2": [2, 4]}, "expected": 20},
        # triangle_area: (0,0),(4,0),(0,3) -> |0*(0-3)+4*(3-0)+0*(0-0)|/2 = 12/2 = 6
        {"input": None, "params": {"operation": "triangle_area", "p1": [0, 0], "p2": [4, 0], "p3": [0, 3]}, "expected": 6},
    ],
    "coordinate_geometry_area": [
        # (0,0),(4,0),(0,3): area=|0*(0-3)+4*(3-0)+0*(0-0)|/2 = 6, result=int(round(6*2))=12
        {"input": None, "params": {"x1": 0, "y1": 0, "x2": 4, "y2": 0, "x3": 0, "y3": 3}, "expected": 12},
    ],
    "coordinate_geometry_synthetic": [
        # centroid: (0,0),(6,0),(3,4) -> gx*3+gy*3 = 0+6+3+0+0+4 = 13
        {"input": None, "params": {"x1": 0, "y1": 0, "x2": 6, "y2": 0, "x3": 3, "y3": 4, "operation": "centroid"}, "expected": 13},
    ],
    "distance_formula": [
        # (0,0) to (3,4) = 5
        {"input": None, "params": {"p1": [0, 0], "p2": [3, 4], "squared": False}, "expected": 5},
        # squared: (0,0) to (3,4) = 25
        {"input": None, "params": {"p1": [0, 0], "p2": [3, 4], "squared": True}, "expected": 25},
    ],
    "minimize_distance_plane": [
        # Plane x+y+z=10, point (1,1,1): |1+1+1-10|/int(sqrt(3)) = 7//1 = 7
        {"input": None, "params": {"a": 1, "b": 1, "c": 1, "d": 10, "x0": 1, "y0": 1, "z0": 1}, "expected": 7},
    ],
    "plane_geometry": [
        # 3-gon: (3-2)*180 = 180
        {"input": None, "params": {"n": 3}, "expected": 180},
        # 4-gon: (4-2)*180 = 360
        {"input": None, "params": {"n": 4}, "expected": 360},
        # 6-gon: (6-2)*180 = 720
        {"input": None, "params": {"n": 6}, "expected": 720},
    ],
    "power_point_coordinate": [
        # Circle at (0,0) r=5, point (8,0): dist²=64, power=64-25=39, result=abs(39)=39
        {"input": None, "params": {"h": 0, "k": 0, "r": 5, "px": 8, "py": 0}, "expected": 39},
        # Circle at (0,0) r=5, point (3,4): dist²=25, power=25-25=0, result=abs(0)=0
        {"input": None, "params": {"h": 0, "k": 0, "r": 5, "px": 3, "py": 4}, "expected": 0},
    ],
    "slope_collinearity": [
        # (0,0),(1,1),(2,2) collinear: det=(1-0)*(2-1)-(2-1)*(1-0)=1-1=0 -> 0
        {"input": None, "params": {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "x3": 2, "y3": 2}, "expected": 0},
        # (0,0),(1,1),(2,3) not collinear: det=(1-0)*(2-1)-(3-1)*(1-0)=1-2=-2 -> 1
        {"input": None, "params": {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "x3": 2, "y3": 3}, "expected": 1},
    ],

    # ========================================================================
    # COORDINATES_2
    # ========================================================================
    "coordinate_geometry_basic": [
        # distance: (0,0) to (3,4) = 5
        {"input": None, "params": {"operation": "distance", "x1": 0, "y1": 0, "x2": 3, "y2": 4}, "expected": 5},
        # midpoint: (0,0)-(6,4), mx=3,my=2 -> round(3+2)=5
        {"input": None, "params": {"operation": "midpoint", "x1": 0, "y1": 0, "x2": 6, "y2": 4}, "expected": 5},
        # slope: (0,0)-(2,6) -> slope=3.0 -> round(3.0*10)=30
        {"input": None, "params": {"operation": "slope", "x1": 0, "y1": 0, "x2": 2, "y2": 6}, "expected": 30},
        # section: (0,0)-(10,10) m=1:n=1 -> px=5,py=5 -> round(10)=10
        {"input": None, "params": {"operation": "section", "x1": 0, "y1": 0, "x2": 10, "y2": 10, "m": 1, "n": 1}, "expected": 10},
    ],
    "distance_point_to_line": [
        # Point (0,0) to line 3x+4y+5=0: |0+0+5|/sqrt(9+16) = 5/5 = 1.0
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "point": (0, 0)}, "expected": 1.0},
        # Point (1,2) to line x+0y-1=0: |1+0-1|/1 = 0.0
        {"input": None, "params": {"a": 1, "b": 0, "c": -1, "point": (1, 2)}, "expected": 0.0},
    ],
    "line_through_two_points": [
        # (0,0) to (1,1): a = y2-y1 = 1
        {"input": None, "params": {"p1": (0, 0), "p2": (1, 1)}, "expected": 1},
        # (0,0) to (1,3): a = 3-0 = 3
        {"input": None, "params": {"p1": (0, 0), "p2": (1, 3)}, "expected": 3},
    ],
    "vector_dot_product": [
        # (1,0).(0,1) = 0
        {"input": None, "params": {"v1": (1, 0), "v2": (0, 1)}, "expected": 0},
        # (3,4).(3,4) = 9+16 = 25
        {"input": None, "params": {"v1": (3, 4), "v2": (3, 4)}, "expected": 25},
        # (1,2).(3,4) = 3+8 = 11
        {"input": None, "params": {"v1": (1, 2), "v2": (3, 4)}, "expected": 11},
    ],
    "angle_from_dot_product": [
        # (1,0) and (0,1): perpendicular -> 90 degrees
        {"input": None, "params": {"v1": (1, 0), "v2": (0, 1)}, "expected": 90},
        # (1,0) and (1,0): same direction -> 0 degrees
        {"input": None, "params": {"v1": (1, 0), "v2": (1, 0)}, "expected": 0},
    ],
    "distance_between_points": [
        # (0,0) to (3,4): sqrt(25) = 5.0
        {"input": None, "params": {"p1": (0, 0), "p2": (3, 4)}, "expected": 5.0},
    ],
    "point_translation": [
        # (3,4) + (2,5) -> new_x = 5
        {"input": None, "params": {"point": (3, 4), "vector": (2, 5)}, "expected": 5},
        # (0,0) + (-3,7) -> new_x = -3
        {"input": None, "params": {"point": (0, 0), "vector": (-3, 7)}, "expected": -3},
    ],
    "slope_to_trig": [
        # slope=1: arctan(1) = 45 degrees
        {"input": None, "params": {"slope": 1}, "expected": 45},
        # slope=0: arctan(0) = 0 degrees
        {"input": None, "params": {"slope": 0}, "expected": 0},
    ],
    "perpendicular_to_line": [
        # Line 3x+4y+5=0: returns b=4
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 4},
        # Line 1x+2y+0=0: returns b=2
        {"input": None, "params": {"a": 1, "b": 2, "c": 0}, "expected": 2},
    ],
    "perpendicular_line_equation": [
        # Line 3x+4y+5=0 through (0,0): perp_c = 3*0-4*0 = 0
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "point": (0, 0)}, "expected": 0},
        # Line 1x+2y+3=0 through (1,1): perp_c = 1*1-2*1 = -1
        {"input": None, "params": {"a": 1, "b": 2, "c": 3, "point": (1, 1)}, "expected": -1},
    ],
    "construct_line_through_points": [
        # (0,0) to (2,4): slope = 4/2 = 2.0
        {"input": None, "params": {"p1": (0, 0), "p2": (2, 4)}, "expected": 2.0},
        # (1,1) to (3,1): slope = 0/2 = 0.0
        {"input": None, "params": {"p1": (1, 1), "p2": (3, 1)}, "expected": 0.0},
    ],
    "line_equation_from_slope_point": [
        # slope=2, point=(1,3): c = 3-2*1 = 1.0
        {"input": None, "params": {"slope": 2, "point": (1, 3)}, "expected": 1.0},
        # slope=0, point=(5,7): c = 7-0*5 = 7.0
        {"input": None, "params": {"slope": 0, "point": (5, 7)}, "expected": 7.0},
    ],

    # ========================================================================
    # COORDINATES_3
    # ========================================================================
    "intersection_two_lines": [
        # x+y+0=0 and x-y+0=0: intersection at (0,0) -> x=0
        {"input": None, "params": {"line1": (1, 1, 0), "line2": (1, -1, 0)}, "expected": 0},
        # 2x+0y-6=0 and 0x+3y-9=0: x=3,y=3 -> int(round(3))=3
        {"input": None, "params": {"line1": (2, 0, -6), "line2": (0, 3, -9)}, "expected": 3},
    ],
    "segment_parameterize": [
        # (0,0) to (10,0), t=0.5: px = 0+0.5*10 = 5.0
        {"input": None, "params": {"p1": (0, 0), "p2": (10, 0), "t": 0.5}, "expected": 5.0},
        # (0,0) to (6,8), t=0.0: px = 0.0
        {"input": None, "params": {"p1": (0, 0), "p2": (6, 8), "t": 0.0}, "expected": 0.0},
    ],
    "trisection_point": [
        # (0,0) to (9,0), which=1: px = 0 + (1/3)*9 = 3.0
        {"input": None, "params": {"p1": (0, 0), "p2": (9, 0), "which": 1}, "expected": 3.0},
        # (0,0) to (9,0), which=2: px = 0 + (2/3)*9 = 6.0
        {"input": None, "params": {"p1": (0, 0), "p2": (9, 0), "which": 2}, "expected": 6.0},
    ],
    "construct_perpendicular_line": [
        # Line 3x+4y+5=0 through (0,0): d = 3*0-4*0 = 0
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "point": (0, 0)}, "expected": 0},
        # Line 1x+2y+3=0 through (2,3): d = 1*3-2*2 = -1
        {"input": None, "params": {"a": 1, "b": 2, "c": 3, "point": (2, 3)}, "expected": -1},
    ],
    "vector_ob": [
        # O=(0,0), B=(3,4): magnitude = sqrt(9+16)=5.0
        {"input": None, "params": {"o": (0, 0), "b": (3, 4)}, "expected": 5.0},
    ],
    "vector_bc": [
        # B=(0,0), C=(5,12): magnitude = sqrt(25+144)=13.0
        {"input": None, "params": {"b": (0, 0), "c": (5, 12)}, "expected": 13.0},
    ],
    "point_o": [
        # Origin always returns 0
        {"input": None, "params": {}, "expected": 0},
    ],
    "point_b": [
        # coords=(7,3) -> returns x=7
        {"input": None, "params": {"coords": (7, 3)}, "expected": 7},
    ],
    "point_c": [
        # coords=(7,4) -> returns y=4
        {"input": None, "params": {"coords": (7, 4)}, "expected": 4},
    ],
    "collinear": [
        # (0,0),(1,1),(2,2) -> collinear -> 1
        {"input": None, "params": {"p1": (0, 0), "p2": (1, 1), "p3": (2, 2)}, "expected": 1},
        # (0,0),(1,1),(2,3) -> not collinear -> 0
        {"input": None, "params": {"p1": (0, 0), "p2": (1, 1), "p3": (2, 3)}, "expected": 0},
    ],
    "point_on_line": [
        # (1.5,1.5) on line (0,0)-(3,3) -> on line -> 1
        {"input": None, "params": {"point": (1.5, 1.5), "line_p1": (0, 0), "line_p2": (3, 3)}, "expected": 1},
        # (1,2) on line (0,0)-(3,3) -> not on line -> 0
        {"input": None, "params": {"point": (1, 2), "line_p1": (0, 0), "line_p2": (3, 3)}, "expected": 0},
    ],

    # ========================================================================
    # POLYGONS
    # ========================================================================
    "shoelace_area": [
        # Triangle (0,0),(10,0),(5,10): area=|0*0-10*0+10*10-5*0+5*0-0*10|/2 = |0+100+0|/2 = 50
        {"input": None, "params": {"vertices": [(0, 0), (10, 0), (5, 10)]}, "expected": 50},
        # Unit square (0,0),(1,0),(1,1),(0,1): area=1
        {"input": None, "params": {"vertices": [(0, 0), (1, 0), (1, 1), (0, 1)]}, "expected": 1},
    ],
    "angle_constraints_tilings": [
        # Hexagon: internal = (6-2)*180//6 = 120, k=360//120=3
        {"input": None, "params": {"n": 6}, "expected": 3},
        # Square: internal = (4-2)*180//4 = 90, k=360//90=4
        {"input": None, "params": {"n": 4}, "expected": 4},
        # Triangle: internal = (3-2)*180//3 = 60, k=360//60=6
        {"input": None, "params": {"n": 3}, "expected": 6},
    ],
    "convex_hull_analysis": [
        # Simple triangle (0,0),(4,0),(2,3): hull has 3 vertices
        {"input": None, "params": {"points": [(0, 0), (4, 0), (2, 3)], "operation": "vertices"}, "expected": 3},
        # area: shoelace of (0,0),(4,0),(2,3) = |0*0-4*0+4*3-2*0+2*0-0*3|/2 = |12|/2 = 6
        {"input": None, "params": {"points": [(0, 0), (4, 0), (2, 3)], "operation": "area"}, "expected": 6},
    ],
    "cyclic_quadrilaterals": [
        # angle=60: opposite = 180-60 = 120
        {"input": None, "params": {"angle": 60}, "expected": 120},
        # angle=90: opposite = 180-90 = 90
        {"input": None, "params": {"angle": 90}, "expected": 90},
    ],
    "regular_polygon_properties": [
        # n=6, s=4: perimeter = 24
        {"input": None, "params": {"n": 6, "s": 4}, "expected": 24},
        # n=3, s=5: perimeter = 15
        {"input": None, "params": {"n": 3, "s": 5}, "expected": 15},
    ],
    "cyclic_quadrilateral": [
        # sides 3,4,5,6: Ptolemy product = 3*5 + 4*6 = 15+24 = 39
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "d": 6}, "expected": 39},
        # sides 5,6,7,8: = 5*7+6*8 = 35+48 = 83
        {"input": None, "params": {"a": 5, "b": 6, "c": 7, "d": 8}, "expected": 83},
    ],

    # ========================================================================
    # POLYGONS_2
    # ========================================================================
    "parallelogram_properties": [
        # a=10, b=10, angle=90: area=100*sin(90)=100, d1=sqrt(200+200*cos90)=sqrt(200)=14.14
        # result = round(100 + 14.14/2) = round(107.07) = 107
        {"input": None, "params": {"a": 10, "b": 10, "angle": 90}, "expected": 107},
    ],
    "polygon_area": [
        # Triangle (0,0),(4,0),(0,3): area = |0*0-4*0+4*3-0*0+0*0-0*3|/2 = 12/2 = 6.0
        {"input": None, "params": {"vertices": [(0, 0), (4, 0), (0, 3)]}, "expected": 6.0},
    ],
    "quadrilateral_area": [
        # Unit square (0,0),(1,0),(1,1),(0,1):
        # T1: (0,0),(1,0),(1,1) area=0.5, T2: (0,0),(1,1),(0,1) area=0.5 -> total=1.0
        {"input": None, "params": {"vertices": [(0, 0), (1, 0), (1, 1), (0, 1)]}, "expected": 1.0},
    ],
    "trapezoid_height": [
        # b1=10, b2=10, area=100: h = 2*100/(10+10) = 10.0
        {"input": None, "params": {"b1": 10, "b2": 10, "area": 100}, "expected": 10.0},
        # b1=6, b2=4, area=25: h = 2*25/10 = 5.0
        {"input": None, "params": {"b1": 6, "b2": 4, "area": 25}, "expected": 5.0},
    ],
    "trapezoid_midline": [
        # b1=10, b2=20: midline = 30/2 = 15.0
        {"input": None, "params": {"b1": 10, "b2": 20}, "expected": 15.0},
        # b1=7, b2=3: midline = 5.0
        {"input": None, "params": {"b1": 7, "b2": 3}, "expected": 5.0},
    ],
    "trapezoid_area": [
        # b1=10, b2=10, h=5: area = (10+10)*5/2 = 50.0
        {"input": None, "params": {"b1": 10, "b2": 10, "h": 5}, "expected": 50.0},
        # b1=3, b2=7, h=4: area = 10*4/2 = 20.0
        {"input": None, "params": {"b1": 3, "b2": 7, "h": 4}, "expected": 20.0},
    ],
    "rectangle_coordinates": [
        # width=10, height=8, center=(0,0): vertices at (+-5, +-4)
        # x-sum = -5+5+5-5 = 0.0
        {"input": None, "params": {"width": 10, "height": 8, "center": (0, 0)}, "expected": 0.0},
        # center=(3,0), width=4: vertices x = 1,5,5,1 -> sum = 12.0
        {"input": None, "params": {"width": 4, "height": 6, "center": (3, 0)}, "expected": 12.0},
    ],

    # ========================================================================
    # TRANSFORMATIONS
    # ========================================================================
    "reflection_geometry": [
        # (5,3) across x-axis: reflected=(5,-3), distance=sqrt(0+36)=6
        {"input": None, "params": {"px": 5, "py": 3, "axis_type": 0}, "expected": 6},
        # (5,3) across y-axis: reflected=(-5,3), distance=sqrt(100+0)=10
        {"input": None, "params": {"px": 5, "py": 3, "axis_type": 1}, "expected": 10},
        # (5,3) across y=x: reflected=(3,5), distance=sqrt(4+4)=sqrt(8)=2.83 -> round=3
        {"input": None, "params": {"px": 5, "py": 3, "axis_type": 2}, "expected": 3},
    ],
    "symmetry_rotation": [
        # n=6: rotation_angle=60, num_symmetries=6, total=round(60*6/10)=36
        {"input": None, "params": {"n": 6}, "expected": 36},
        # n=4: rotation_angle=90, num_symmetries=4, total=round(90*4/10)=36
        {"input": None, "params": {"n": 4}, "expected": 36},
        # n=3: rotation_angle=120, num_symmetries=3, total=round(120*3/10)=36
        {"input": None, "params": {"n": 3}, "expected": 36},
    ],
    "reflection_principle": [
        # m=3, n=4: C(7,3) = 35
        {"input": None, "params": {"m": 3, "n": 4}, "expected": 35},
        # m=2, n=2: C(4,2) = 6
        {"input": None, "params": {"m": 2, "n": 2}, "expected": 6},
    ],

    # ========================================================================
    # MISC
    # ========================================================================
    "picks_theorem": [
        # i=10, b=6: A = 10 + 6/2 - 1 = 12
        {"input": None, "params": {"interior": 10, "boundary": 6}, "expected": 12},
        # i=5, b=4: A = 5 + 2 - 1 = 6
        {"input": None, "params": {"interior": 5, "boundary": 4}, "expected": 6},
    ],
    "picks_inverse_interior": [
        # A=12, b=6: i = 12 - 3 + 1 = 10
        {"input": None, "params": {"area": 12, "boundary": 6}, "expected": 10},
    ],
    "lattice_points_on_segment": [
        # (0,0) to (6,8): gcd(6,8)+1 = 2+1 = 3
        {"input": None, "params": {"p1": (0, 0), "p2": (6, 8)}, "expected": 3},
        # (0,0) to (10,0): gcd(10,0)+1 = 10+1 = 11
        {"input": None, "params": {"p1": (0, 0), "p2": (10, 0)}, "expected": 11},
    ],
    "lattice_points_in_region": [
        # 5x5: (5+1)*(5+1) = 36
        {"input": None, "params": {"width": 5, "height": 5}, "expected": 36},
        # 3x4: (3+1)*(4+1) = 20
        {"input": None, "params": {"width": 3, "height": 4}, "expected": 20},
    ],
    "cross_ratio": [
        # Points 0,1,3,7: AC=3,BC=2,AD=7,BD=6 -> (3/2)/(7/6) = (3/2)*(6/7) = 18/14 = 1.286 -> 1
        {"input": None, "params": {"points": [0, 1, 3, 7]}, "expected": 1},
    ],
    "harmonic_bundle": [
        # A=-6, B=6, C=2: AC=8, BC=-4, 2BC-AC=-8-8=-16
        # D = (2*(-4)*(-6) - 8*6)/(-16) = (48-48)/(-16) = 0 -> 0
        {"input": None, "params": {"A": -6, "B": 6, "C": 2}, "expected": 0},
    ],
    "tetrahedron_volume": [
        # (0,0,0),(10,0,0),(0,10,0),(0,0,10): det = |[[10,0,0],[0,10,0],[0,0,10]]| = 1000
        # volume = 1000/6 = 166.67 -> 167
        {"input": None, "params": {"vertices": [(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)]}, "expected": 167},
    ],
    "cylinder_geometry": [
        # r=3, h=5: V = pi*9*5 = 141.371669...
        {"input": None, "params": {"r": 3, "h": 5}, "expected": 141.371669},
    ],
    "cutting_plane_equation": [
        # n=5: C(5,2) = 10
        {"input": None, "params": {"n": 5}, "expected": 10},
        # n=4: C(4,2) = 6
        {"input": None, "params": {"n": 4}, "expected": 6},
    ],
    "isogonal_conjugation": [
        # a=7,b=8,c=9, point (1,1,1): x'=49, y'=64, z'=81, sum=194
        {"input": None, "params": {"a": 7, "b": 8, "c": 9, "x": 1, "y": 1, "z": 1}, "expected": 194},
    ],

    # ========================================================================
    # MISC_2
    # ========================================================================
    "lattice_point_counting": [
        # rectangle 5x4: total = (5+1)*(4+1) = 30
        {"input": None, "params": {"shape": "rectangle", "width": 5, "height": 4}, "expected": 30},
        # segment (0,0) to (6,8): gcd(6,8)+1 = 3
        {"input": None, "params": {"shape": "segment", "x1": 0, "y1": 0, "x2": 6, "y2": 8}, "expected": 3},
    ],
    "projective_geometry": [
        # PG(2,3) points: 9+3+1 = 13
        {"input": None, "params": {"operation": "points", "n": 3}, "expected": 13},
        # points_per_line in PG(2,3): 3+1 = 4
        {"input": None, "params": {"operation": "points_per_line", "n": 3}, "expected": 4},
    ],
    "surface_area_calculation": [
        # cube a=5: 6*25 = 150
        {"input": None, "params": {"shape": "cube", "a": 5}, "expected": 150},
        # rectangular prism 3x4x5: 2*(12+20+15) = 94
        {"input": None, "params": {"shape": "rectangular_prism", "l": 3, "w": 4, "h": 5}, "expected": 94},
    ],
    "tiling_theory": [
        # unit_squares 4x6: 24
        {"input": None, "params": {"operation": "unit_squares", "m": 4, "n": 6}, "expected": 24},
        # dominoes 4x6: 24//2 = 12
        {"input": None, "params": {"operation": "dominoes", "m": 4, "n": 6}, "expected": 12},
    ],
    "trig_identities_synthetic": [
        # sin^2(x) + cos^2(x) = 1 always -> int(100*1) = 100
        {"input": None, "params": {"angle": 30}, "expected": 100},
        {"input": None, "params": {"angle": 45}, "expected": 100},
        {"input": None, "params": {"angle": 60}, "expected": 100},
    ],
    "unfolding_nets_pathfinding": [
        # 3x4x5 shortest path: min(sqrt(7^2+5^2), sqrt(8^2+4^2), sqrt(9^2+3^2))
        # = min(sqrt(74), sqrt(80), sqrt(90)) = sqrt(74) = 8.602
        # result = int(round(8.602*10)) = 86
        {"input": None, "params": {"l": 3, "w": 4, "h": 5, "operation": "shortest_path"}, "expected": 86},
    ],

    # ========================================================================
    # MISC_3
    # ========================================================================
    "volume_calculation": [
        # box 4x5x6: 120
        {"input": None, "params": {"shape": "box", "l": 4, "w": 5, "h": 6}, "expected": 120},
        # pyramid base=6, h=10: (1/3)*36*10 = 120
        {"input": None, "params": {"shape": "pyramid", "base": 6, "h": 10}, "expected": 120},
    ],
    "geometric_series_sum": [
        # finite_sum: a=1, r=1/2, n=5: 1*(1-(1/2)^5)/(1-1/2) = (1-1/32)/(1/2) = (31/32)*2 = 31/16 = 1.9375 -> 2
        {"input": None, "params": {"operation": "finite_sum", "a": 1, "r_num": 1, "r_den": 2, "n": 5}, "expected": 2},
        # infinite_sum: a=4, r=1/2: 4/(1-0.5) = 8
        {"input": None, "params": {"operation": "infinite_sum", "a": 4, "r_num": 1, "r_den": 2, "n": 1}, "expected": 8},
    ],
    "tangent_addition": [
        # sum: tan(30+45) = tan(75) = 2+sqrt(3) = 3.732 -> int(round(3.732*100)) = 373
        {"input": None, "params": {"angle_a": 30, "angle_b": 45, "operation": "sum"}, "expected": 373},
        # difference: tan(45-30) = tan(15) = 2-sqrt(3) = 0.2679 -> int(round(0.2679*100)) = 27
        {"input": None, "params": {"angle_a": 45, "angle_b": 30, "operation": "difference"}, "expected": 27},
        # double: tan(2*30) = tan(60) = sqrt(3) = 1.732 -> int(round(1.732*100)) = 173
        {"input": None, "params": {"angle_a": 30, "angle_b": 45, "operation": "double"}, "expected": 173},
    ],
    "3d_combinatorics": [
        # grid_points n=3: (3+1)^3 = 64
        {"input": None, "params": {"operation": "grid_points", "n": 3}, "expected": 64},
        # space_diagonals: always 4
        {"input": None, "params": {"operation": "space_diagonals", "n": 3}, "expected": 4},
        # face_diagonals: always 12
        {"input": None, "params": {"operation": "face_diagonals", "n": 3}, "expected": 12},
        # planes_from_points n=5: C(5,3) = 10
        {"input": None, "params": {"operation": "planes_from_points", "n": 5}, "expected": 10},
    ],
    "3d_distance_pythagorean": [
        # 3,4,5: dist_sq=50, dist=int(sqrt(50))=int(7.07)=7
        {"input": None, "params": {"x": 3, "y": 4, "z": 5}, "expected": 7},
        # 1,2,2: dist_sq=9, dist=int(sqrt(9))=3
        {"input": None, "params": {"x": 1, "y": 2, "z": 2}, "expected": 3},
    ],
    "coprime_diff_squares_param": [
        # Fixed computation k=1,x=2,y=1: a=7,b=8,c=6 -> product=336
        {"input": None, "params": {}, "expected": 336},
    ],
    "product_mod": [
        # 14*8*6 = 672, 672 % 100000 = 672
        {"input": None, "params": {"a": 14, "b": 8, "c": 6, "modulus": 100000}, "expected": 672},
        # 100*200*300 = 6000000, % 100000 = 0
        {"input": None, "params": {"a": 100, "b": 200, "c": 300, "modulus": 100000}, "expected": 0},
    ],
    "volume_ball": [
        # 3D sphere r=3: V=(4/3)*pi*27 = 113.097 -> but returns float
        # V = pi^(3/2) * 27 / Gamma(5/2) = 5.568*27/1.329 = 113.097
        {"input": None, "params": {"radius": 3, "dimension": 3}, "expected": (4.0/3.0)*math.pi*27},
        # 2D disk r=5: V = pi*25 = 78.54
        {"input": None, "params": {"radius": 5, "dimension": 2}, "expected": math.pi * 25},
    ],
    "ellipse_standard_form": [
        # a=10, b=5, center=(0,0): area = int(round(pi*10*5)) = int(round(157.08)) = 157
        {"input": None, "params": {"a": 10, "b": 5, "center": (0, 0)}, "expected": 157},
    ],
    "ellipse_vertices": [
        # a=10, b=5, center=(0,0): vertices (10,0),(-10,0),(0,5),(0,-5)
        # x-sum = 10+(-10)+0+0 = 0
        {"input": None, "params": {"a": 10, "b": 5, "center": (0, 0)}, "expected": 0},
        # center=(3,0): x-sum = 13+(-7)+3+3 = 12
        {"input": None, "params": {"a": 10, "b": 5, "center": (3, 0)}, "expected": 12},
    ],

    # ========================================================================
    # MISC_4
    # ========================================================================
    "hyperbola_semi_axes": [
        # a=3, b=4: product = 12
        {"input": None, "params": {"a": 3, "b": 4, "center": (0, 0)}, "expected": 12},
        # a=5, b=5: product = 25
        {"input": None, "params": {"a": 5, "b": 5, "center": (0, 0)}, "expected": 25},
    ],
    "hyperbola_foci_distance": [
        # a=3, b=4: c=sqrt(9+16)=5, distance=10.0
        {"input": None, "params": {"a": 3, "b": 4}, "expected": 10.0},
    ],
    "volume_from_bounds": [
        # [0,10]x[0,10]x[0,10]: volume = 10*10*10 = 1000
        {"input": None, "params": {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10, "z_min": 0, "z_max": 10}, "expected": 1000},
        # [1,4]x[2,7]x[3,6]: volume = 3*5*3 = 45
        {"input": None, "params": {"x_min": 1, "x_max": 4, "y_min": 2, "y_max": 7, "z_min": 3, "z_max": 6}, "expected": 45},
    ],
    "poles_and_polars": [
        # r=6, d=9: polar_distance = 36/9 = 4
        {"input": None, "params": {"radius": 6, "distance": 9}, "expected": 4},
        # r=10, d=5: polar_distance = 100/5 = 20
        {"input": None, "params": {"radius": 10, "distance": 5}, "expected": 20},
    ],
    # ========================================================================
    # HOMOTHETY (transformations.py) - scale point from center
    # ========================================================================
    "homothety": [
        # center=(0,0), scale=2, point=(3,4) -> (6,8)
        {"input": None, "params": {"center": (0, 0), "scale": 2, "point": (3, 4)}, "expected": (6, 8)},
    ],
    # ========================================================================
    # HOMOTHETY INVERSE (transformations.py) - find center from two points + scale
    # ========================================================================
    "homothety_inverse": [
        # p1=(3,4), p2=(9,12), scale=3 -> center=(0,0)
        {"input": None, "params": {"p1": (3, 4), "p2": (9, 12), "scale": 3}, "expected": (0.0, 0.0)},
    ],
    # ========================================================================
    # INVERSIVE GEOMETRY (transformations.py) - circle inversion
    # ========================================================================
    "inversive_geometry": [
        # center=(0,0), r=10, point=(5,0) -> (20,0): d=5, r^2/d=20
        {"input": None, "params": {"center": (0, 0), "radius": 10, "point": (5, 0)}, "expected": (20.0, 0.0)},
    ],
    # ========================================================================
    # REFLECTION (transformations.py) - reflect point over line y=mx
    # ========================================================================
    "reflection": [
        # slope=1 (y=x), point=(3,4) -> (4,3)
        {"input": None, "params": {"slope": 1, "point": (3, 4)}, "expected": (4.0, 3.0)},
    ],
    # ========================================================================
    # SPIRAL SIMILARITY (transformations.py) - rotate + scale
    # ========================================================================
    "spiral_similarity": [
        # angle=0, scale=2, (3,4) -> (6,8) (pure scaling)
        {"input": None, "params": {"center": (0, 0), "angle": 0, "scale": 2.0, "point": (3, 4)}, "expected": (6.0, 8.0)},
    ],
    # ========================================================================
    # LINE INTERSECTION (basic_primitives/geometry.py)
    # ========================================================================
    "line_intersection": [
        # y=x through (0,0) dir (1,1) meets y=-x+4 through (0,4) dir (1,-1) at (2,2)
        {"input": None, "params": {"line1": {"midpoint": (0, 0), "direction": (1, 1)}, "line2": {"midpoint": (0, 4), "direction": (1, -1)}}, "expected": (2.0, 2.0)},
    ],
    # ========================================================================
    # CONVEX HULL AREA (polygons_2.py)
    # ========================================================================
    "convex_hull_area": [
        # 3-4-5 right triangle: area = 6.0
        {"input": None, "params": {"points": [(0, 0), (4, 0), (0, 3)]}, "expected": 6.0},
        # 10x10 square: area = 100.0
        {"input": None, "params": {"points": [(0, 0), (10, 0), (10, 10), (0, 10)]}, "expected": 100.0},
    ],
    # ========================================================================
    # PERPENDICULAR BISECTOR (basic_primitives/geometry.py)
    # ========================================================================
    "perpendicular_bisector": [
        # (0,0) to (6,8): midpoint dist = sqrt(9+16) = 5, doubled = 10
        {"input": None, "params": {"p1": (0, 0), "p2": (6, 8)}, "expected": 10},
        # (0,0) to (10,0): midpoint dist = 5, doubled = 10
        {"input": None, "params": {"p1": (0, 0), "p2": (10, 0)}, "expected": 10},
    ],
    # ========================================================================
    # DISTANCE INVERSE POINT (coordinates.py) - find point at given distance
    # ========================================================================
    "distance_inverse_point": [
        # origin, distance=5, angle=0 -> (5,0)
        {"input": None, "params": {"p1": (0, 0), "distance": 5, "angle": 0}, "expected": (5.0, 0.0)},
        # (3,4), distance=5, angle=0 -> (8,4)
        {"input": None, "params": {"p1": (3, 4), "distance": 5, "angle": 0}, "expected": (8.0, 4.0)},
    ],
    # ========================================================================
    # EULER OG DISTANCE (triangles_2.py) - Euler's formula: OG^2 = R^2 - ...
    # ========================================================================
    "euler_og_distance": [
        # 3-4-5 right triangle
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 8},
        # 5-12-13 right triangle
        {"input": None, "params": {"a": 5, "b": 12, "c": 13}, "expected": 22},
        # Equilateral 10-10-10: O=G so distance = 0
        {"input": None, "params": {"a": 10, "b": 10, "c": 10}, "expected": 0},
    ],
    # ========================================================================
    # CIRCLE THROUGH THREE POINTS (circles_2.py) - circumradius
    # ========================================================================
    "circle_through_three_points": [
        # (0,0),(4,0),(0,3): right triangle, R = hyp/2 = 5/2 = 2.5
        {"input": None, "params": {"p1": (0, 0), "p2": (4, 0), "p3": (0, 3)}, "expected": 2.5},
        # (0,0),(10,0),(0,10): isosceles right, R = 10*sqrt(2)/2 = 5*sqrt(2)
        {"input": None, "params": {"p1": (0, 0), "p2": (10, 0), "p3": (0, 10)}, "expected": 7.0710678118654755},
    ],
    # ========================================================================
    # CIRCUMCIRCLE ANGLE BISECTOR (circles_2.py) - bisector/chord computations
    # ========================================================================
    "circumcircle_angle_bisector": [
        # 3-4-5 bisector_length: t_a=2*4*5*cos(A/2)/(4+5), encoded *10
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "bisector_length"}, "expected": 42},
        # 3-4-5 chord_bm: BM=2R*sin(A/2), encoded *10
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "chord_bm"}, "expected": 16},
        # Equilateral 10-10-10 bisector_length
        {"input": None, "params": {"a": 10, "b": 10, "c": 10, "operation": "bisector_length"}, "expected": 87},
    ],
    # ========================================================================
    # LATTICE CONVEX POLYGON (misc_2.py) - max vertices on NxN grid
    # ========================================================================
    "lattice_convex_polygon": [
        # n=10: uses Euler's totient to count lattice-visible directions
        {"input": None, "params": {"n": 10}, "expected": 11},
        # n=4
        {"input": None, "params": {"n": 4}, "expected": 3},
        # n=6
        {"input": None, "params": {"n": 6}, "expected": 5},
    ],
    # ========================================================================
    # SIDE TIMES SIN (triangles_4.py)
    # ========================================================================
    "side_times_sin": [
        # 10 * sin(30deg) = 5.0 (with float rounding)
        {"input": None, "params": {"side": 10, "angle": math.radians(30)}, "expected": 4.999999999999999},
        # 10 * sin(90deg) = 10.0
        {"input": None, "params": {"side": 10, "angle": math.radians(90)}, "expected": 10.0},
    ],
    # ========================================================================
    # SINE SUPPLEMENTARY ANGLE (triangles_4.py) - sin(pi - theta) = sin(theta)
    # ========================================================================
    "sine_supplementary_angle": [
        # sin(pi - 30deg) = sin(30deg) = 0.5
        {"input": None, "params": {"angle": math.radians(30)}, "expected": 0.49999999999999994},
        # sin(pi - 45deg) = sin(45deg) = sqrt(2)/2
        {"input": None, "params": {"angle": math.radians(45)}, "expected": 0.7071067811865476},
    ],
    # ========================================================================
    # TRIANGLES SHARED ANGLE (triangles_3.py) - ratio via shared angle
    # ========================================================================
    "triangles_shared_angle": [
        # area_ratio: (ab*ac)/(ad*ae) * 100 = (6*8)/(3*4)*100/... -> 25
        {"input": None, "params": {"operation": "area_ratio", "ab": 6, "ac": 8, "ad": 3, "ae": 4}, "expected": 25},
        # find_side: finds missing side from ratio constraint -> 12
        {"input": None, "params": {"operation": "find_side", "ab": 6, "ac": 8, "ad": 3, "ae": 4}, "expected": 12},
    ],
    # ========================================================================
    # TRIANGLE PLACE COORDINATES (triangles_4.py) - coordinate placement
    # ========================================================================
    "triangle_place_coordinates": [
        # 3-4-5 triangle: returns encoded height = 3.0
        {"input": None, "params": {"a": 5, "b": 4, "c": 3}, "expected": 3.0},
        # Equilateral 10-10-10: returns encoded value = 15.0
        {"input": None, "params": {"a": 10, "b": 10, "c": 10}, "expected": 15.0},
    ],
}
