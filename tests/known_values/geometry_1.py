# fmt: off
"""
Known-value ground truth for geometry methods (Part 1).

Covers: triangles, triangles_2, triangles_3, triangles_4, circles, circles_2.
"""
import math

KNOWN_VALUES = {
    # ========================================================================
    # TRIANGLES
    # ========================================================================
    "area_heron": [
        # 3-4-5 right triangle: s=6, K=sqrt(6*3*2*1)=6
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 6},
        # 5-12-13: s=15, K=sqrt(15*10*3*2)=sqrt(900)=30
        {"input": None, "params": {"a": 5, "b": 12, "c": 13}, "expected": 30},
        # Equilateral 10-10-10: s=15, K=sqrt(15*5*5*5)=sqrt(1875)=43.30 -> 43
        {"input": None, "params": {"a": 10, "b": 10, "c": 10}, "expected": 43},
    ],
    "stewart_theorem": [
        # Triangle 10,10,10 with m=5,n=5 (median): d²=(100*5+100*5-10*25)/10=(500+500-250)/10=75, d=sqrt(75)=8.66 -> 9
        {"input": None, "params": {"a": 10, "b": 10, "c": 10, "m": 5, "n": 5}, "expected": 9},
        # Triangle 13,14,15 with m=7,n=6 (cevian on side a=13):
        # d²=(14²*7 + 15²*6 - 13*7*6)/13 = (1372+1350-546)/13 = 2176/13 = 167.38, d=12.94 -> 13
        {"input": None, "params": {"a": 13, "b": 14, "c": 15, "m": 7, "n": 6}, "expected": 13},
    ],
    "angle_bisector_length": [
        # For equilateral 10-10-10: t_a = sqrt(10*10*((10+10)²-10²))/(10+10)
        # = sqrt(100*(400-100))/20 = sqrt(100*300)/20 = sqrt(30000)/20 = 173.2/20 = 8.66 -> 9
        {"input": None, "params": {"a": 10, "b": 10, "c": 10}, "expected": 9},
    ],
    "median_length": [
        # 3-4-5 triangle, median to side a=3: m_a = 0.5*sqrt(2*16+2*25-9) = 0.5*sqrt(73) = 0.5*8.544 = 4.27 -> 4
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 4},
        # Equilateral 6-6-6: m_a = 0.5*sqrt(2*36+2*36-36) = 0.5*sqrt(108) = 0.5*10.39 = 5.20 -> 5
        {"input": None, "params": {"a": 6, "b": 6, "c": 6}, "expected": 5},
    ],
    "angle_bisector_theorem": [
        # AB=10, AC=10: ratio = BD/DC = AB/AC = 1.0 -> 1
        {"input": None, "params": {"AB": 10, "AC": 10, "BC": 12}, "expected": 1},
        # AB=6, AC=3: ratio = 6/3 = 2.0 -> 2
        {"input": None, "params": {"AB": 6, "AC": 3, "BC": 9}, "expected": 2},
    ],
    "ceva_theorem": [
        # Concurrent cevians: product should be 1.0
        {"input": None, "params": {"r1": 2.0, "r2": 0.5, "r3": 1.0}, "expected": 1.0},
        {"input": None, "params": {"r1": 1.0, "r2": 1.0, "r3": 1.0}, "expected": 1.0},
    ],
    "menelaus_theorem": [
        # Collinear: product should be -1.0
        {"input": None, "params": {"r1": 1.0, "r2": -1.0, "r3": 1.0}, "expected": -1.0},
        {"input": None, "params": {"r1": 2.0, "r2": -0.5, "r3": 1.0}, "expected": -1.0},
    ],
    "law_of_cosines": [
        # Right triangle: a=3,b=4,angle=90 -> c=sqrt(9+16)=5
        {"input": None, "params": {"a": 3, "b": 4, "angle_C": 90.0}, "expected": 5},
        # a=10,b=10,angle=60 -> c²=100+100-200*cos(60°)=200-100=100 -> c=10
        {"input": None, "params": {"a": 10, "b": 10, "angle_C": 60.0}, "expected": 10},
    ],
    "law_of_cosines_inverse_angle": [
        # 3-4-5: cos(C)=(9+16-25)/24=0 -> angle=90
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 90},
        # Equilateral 10-10-10: cos(C)=(100+100-100)/200=0.5 -> angle=60
        {"input": None, "params": {"a": 10, "b": 10, "c": 10}, "expected": 60},
    ],
    "law_of_sines_ratio": [
        # 3-4-5: K=6, 2R = 2*abc/(4K) = 2*60/24 = 5.0 -> 5
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 5},
        # 5-12-13: K=30, 2R = 2*780/120 = 13.0 -> 13
        {"input": None, "params": {"a": 5, "b": 12, "c": 13}, "expected": 13},
    ],
    "angle_bisector_theorem_extended": [
        # a=6, b=4: gcd=2, ratio_num=4/2=2, ratio_den=6/2=3 -> 2+3=5
        {"input": None, "params": {"a": 6, "b": 4}, "expected": 5},
        # a=10, b=10: gcd=10, ratio_num=1, ratio_den=1 -> 2
        {"input": None, "params": {"a": 10, "b": 10}, "expected": 2},
    ],

    # ========================================================================
    # TRIANGLES_2
    # ========================================================================
    "angle_chasing": [
        # Triangle: 60+70 = 130, unknown = 180-130 = 50
        {"input": None, "params": {"known_angles": [60, 70], "polygon_sides": 3, "angle_type": "interior"}, "expected": 50},
        # Supplementary: 120, unknown = 180-120 = 60
        {"input": None, "params": {"known_angles": [120], "polygon_sides": 3, "angle_type": "supplementary"}, "expected": 60},
        # Complementary: 30, unknown = 90-30 = 60
        {"input": None, "params": {"known_angles": [30], "polygon_sides": 3, "angle_type": "complementary"}, "expected": 60},
    ],
    "apollonius_theorem": [
        # 2*(3²+4²)-5² = 2*(9+16)-25 = 50-25 = 25
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 25},
        # 2*(5²+5²)-5² = 2*50-25 = 75
        {"input": None, "params": {"a": 5, "b": 5, "c": 5}, "expected": 75},
    ],
    "area_decomposition": [
        # find_unknown: 100 - (30+20) = 50
        {"input": None, "params": {"operation": "find_unknown", "total_area": 100, "known_parts": [30, 20]}, "expected": 50},
        # sum_parts: 10+20+30 = 60
        {"input": None, "params": {"operation": "sum_parts", "known_parts": [10, 20, 30]}, "expected": 60},
        # ratio_split: 100 in ratio 2:3 -> first part = 100*2/5 = 40
        {"input": None, "params": {"operation": "ratio_split", "total_area": 100, "ratio": [2, 3]}, "expected": 40},
        # subtract: 100-25 = 75
        {"input": None, "params": {"operation": "subtract", "outer_area": 100, "inner_area": 25}, "expected": 75},
    ],
    "area_sine_formula": [
        # a=10, b=10, angle=90: area = 0.5*10*10*sin(90°) = 50
        {"input": None, "params": {"a": 10, "b": 10, "angle": 90}, "expected": 50},
        # a=10, b=10, angle=30: area = 0.5*10*10*sin(30°) = 25 -> 25
        {"input": None, "params": {"a": 10, "b": 10, "angle": 30}, "expected": 25},
    ],
    "area_maximization_triangles": [
        # Equilateral with perimeter 12: side=4, area=(16*sqrt(3))/4=4*1.732=6.93 -> 7
        {"input": None, "params": {"perimeter": 12, "problem_type": "equilateral_max"}, "expected": 7},
    ],
    "ceva_menelaus_combined": [
        # a=3: C(5,3) = 3*4*5/6 = 10
        {"input": None, "params": {"a": 3}, "expected": 10},
        # a=1: C(3,3) = 1*2*3/6 = 1
        {"input": None, "params": {"a": 1}, "expected": 1},
    ],
    "euler_oi_distance": [
        # 3-4-5 right triangle: R=2.5, r=1, OI=sqrt(R²-2Rr)=sqrt(6.25-5)=sqrt(1.25)≈1
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 1},
        # equilateral 10-10-10: O=I, OI=0
        {"input": None, "params": {"a": 10, "b": 10, "c": 10}, "expected": 0},
    ],
    "isosceles_properties": [
        # leg=5, base=6: altitude = sqrt(25-9)=sqrt(16)=4
        {"input": None, "params": {"leg": 5, "base": 6}, "expected": 4},
        # leg=13, base=10: altitude = sqrt(169-25)=sqrt(144)=12
        {"input": None, "params": {"leg": 13, "base": 10}, "expected": 12},
    ],
    "medial_triangle_properties": [
        # 3-4-5: s=6, area=6, medial area=6/4=1.5 -> round=2
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 2},
        # 13-14-15: s=21, area=sqrt(21*8*7*6)=sqrt(7056)=84, medial=84/4=21
        {"input": None, "params": {"a": 13, "b": 14, "c": 15}, "expected": 21},
    ],
    "circumcenter_centroid_properties": [
        # (0,0),(6,0),(3,6): gx=(0+6+3)//3=3, gy=(0+0+6)//3=2, sum=5
        {"input": None, "params": {"x1": 0, "y1": 0, "x2": 6, "y2": 0, "x3": 3, "y3": 6}, "expected": 5},
    ],

    # ========================================================================
    # TRIANGLES_3
    # ========================================================================
    "orthocenter_altitude_properties": [
        # Simply returns a+b+c (perimeter): 3+4+5=12
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 12},
    ],
    "right_triangle_properties": [
        # 3-4-5, area: (3*4)//2 = 6
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "area"}, "expected": 6},
        # 3-4-5, inradius: (3+4-5)//2 = 1
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "inradius"}, "expected": 1},
        # 3-4-5, circumradius: 5//2 = 2
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "circumradius"}, "expected": 2},
        # 3-4-5, altitude_h: (3*4)//5 = 2
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "altitude_h"}, "expected": 2},
        # 3-4-5, perimeter: 12
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "perimeter"}, "expected": 12},
    ],
    "altitude_feet_cyclic": [
        # angle=60: |180-120| = 60
        {"input": None, "params": {"angle": 60}, "expected": 60},
        # angle=90: abs(180-180) = 0
        {"input": None, "params": {"angle": 90}, "expected": 0},
        # angle=45: abs(180-90) = 90
        {"input": None, "params": {"angle": 45}, "expected": 90},
    ],
    "circumcenter_properties": [
        # r=5: result = 5*2 = 10
        {"input": None, "params": {"r": 5}, "expected": 10},
        {"input": None, "params": {"r": 7}, "expected": 14},
    ],
    "centroid_properties": [
        # (0,0),(6,0),(0,6): cx=(0+6+0)//3=2, cy=(0+0+6)//3=2, result=abs(2)+abs(2)=4
        {"input": None, "params": {"x1": 0, "y1": 0, "x2": 6, "y2": 0, "x3": 0, "y3": 6}, "expected": 4},
        # (3,3),(6,3),(3,6): cx=12//3=4, cy=12//3=4, result=8
        {"input": None, "params": {"x1": 3, "y1": 3, "x2": 6, "y2": 3, "x3": 3, "y3": 6}, "expected": 8},
    ],
    "pythagorean_theorem": [
        # Hypotenuse: 3,4 -> 5
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "problem_type": "hypotenuse"}, "expected": 5},
        # Area: (3*4)//2 = 6
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "problem_type": "area"}, "expected": 6},
        # Leg a: return a=5
        {"input": None, "params": {"a": 5, "b": 12, "c": 13, "problem_type": "leg", "which_leg": "a"}, "expected": 5},
    ],
    "triangle_similarity": [
        # find_side: a1=3, k=2/1=2 -> 6
        {"input": None, "params": {"a1": 3, "b1": 4, "c1": 5, "k_num": 2, "k_den": 1, "operation": "find_side"}, "expected": 6},
        # perimeter_ratio: peri=12, k=2 -> 24
        {"input": None, "params": {"a1": 3, "b1": 4, "c1": 5, "k_num": 2, "k_den": 1, "operation": "perimeter_ratio"}, "expected": 24},
    ],
    "area_ratios": [
        # a1=12, a2=8 -> 20
        {"input": None, "params": {"a1": 12, "a2": 8}, "expected": 20},
    ],

    # ========================================================================
    # TRIANGLES_4
    # ========================================================================
    "similar_triangles": [
        # proportion: a=2,b=3,c=4 -> x = 3*4/2 = 6
        {"input": None, "params": {"operation": "proportion", "a": 2, "b": 3, "c": 4}, "expected": 6},
        # find_side: ratio=2, known_side=5 -> 10
        {"input": None, "params": {"operation": "find_side", "ratio": 2, "known_side": 5}, "expected": 10},
        # find_ratio: side1=6, side2=3 -> 200 (ratio*100)
        {"input": None, "params": {"operation": "find_ratio", "side1": 6, "side2": 3}, "expected": 200},
    ],
    "sum_of_angles": [
        # Triangle: (3-2)*180 = 180
        {"input": None, "params": {"n": 3}, "expected": 180},
        # Pentagon: (5-2)*180 = 540
        {"input": None, "params": {"n": 5}, "expected": 540},
        # Hexagon: (6-2)*180 = 720
        {"input": None, "params": {"n": 6}, "expected": 720},
    ],
    "triangle_vertex_coordinates": [
        # v1=(1,2), v2=(3,4), v3=(5,6): sum_x = 1+3+5 = 9
        {"input": None, "params": {"v1": (1, 2), "v2": (3, 4), "v3": (5, 6)}, "expected": 9},
    ],
    "point_in_triangle": [
        # Centroid of (0,0),(10,0),(0,10) is (3.33,3.33) -> inside
        {"input": None, "params": {"v1": (0, 0), "v2": (10, 0), "v3": (0, 10), "point": (3, 3)}, "expected": 1},
        # Point (20,20) is outside triangle (0,0),(10,0),(0,10)
        {"input": None, "params": {"v1": (0, 0), "v2": (10, 0), "v3": (0, 10), "point": (20, 20)}, "expected": 0},
    ],
    "angle_bisector_point_on_side": [
        # AB=10, AC=10, BC=12: BD = 12*10/(10+10) = 6.0
        {"input": None, "params": {"AB": 10, "AC": 10, "BC": 12}, "expected": 6.0},
        # AB=6, AC=4, BC=10: BD = 10*6/(6+4) = 6.0
        {"input": None, "params": {"AB": 6, "AC": 4, "BC": 10}, "expected": 6.0},
    ],
    "triangle_vertices_placement": [
        # (0,0),(4,0),(0,3): area = 0.5*|0*(0-3)+4*(3-0)+0*(0-0)| = 0.5*12 = 6.0
        {"input": None, "params": {"v1": (0, 0), "v2": (4, 0), "v3": (0, 3)}, "expected": 6.0},
    ],

    # ========================================================================
    # CIRCLES
    # ========================================================================
    "power_of_point": [
        # center=(0,0), r=5, point=(13,0): power = 13² - 5² = 169-25 = 144
        {"input": None, "params": {"center": (0, 0), "radius": 5, "point": (13, 0)}, "expected": 144},
        # center=(0,0), r=10, point=(6,8): dist²=100, power=100-100=0
        {"input": None, "params": {"center": (0, 0), "radius": 10, "point": (6, 8)}, "expected": 0},
    ],
    "power_of_point_inverse_radius": [
        # distance=13, power=25: r²=169-25=144, r=12
        {"input": None, "params": {"distance": 13, "power": 25}, "expected": 12.0},
    ],
    "inradius": [
        # 3-4-5 with no scaling: s=6, K=6, r=6/6=1.0
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "multiplier": 1, "offset": 0}, "expected": 1.0},
        # 3-4-5 with multiplier=100, offset=1000: int(1.0*100+1000) = 1100
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "multiplier": 100, "offset": 1000}, "expected": 1100},
    ],
    "excircle_properties": [
        # 3-4-5: s=6, K=6, r_a = 6/(6-3) = 2.0 -> 2
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 2},
        # 5-12-13: s=15, K=30, r_a = 30/(15-5) = 3.0 -> 3
        {"input": None, "params": {"a": 5, "b": 12, "c": 13}, "expected": 3},
    ],
    "circumradius_inradius_formulas": [
        # 3-4-5: s=6, area=6, R=60/24=2.5, result=int(2.5*10)=25
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 25},
    ],
    "inscribed_angle_theorem": [
        # arc=90 -> inscribed=90//2=45
        {"input": None, "params": {"arc": 90}, "expected": 45},
        # arc=120 -> inscribed=60
        {"input": None, "params": {"arc": 120}, "expected": 60},
        # arc=180 -> inscribed=90
        {"input": None, "params": {"arc": 180}, "expected": 90},
    ],

    # ========================================================================
    # CIRCLES_2
    # ========================================================================
    "nine_point_properties": [
        # 3-4-5: s=6, K=6, R=60/24=2.5, nine-pt radius=R/2=1.25, encoded=int(1.25*10)=12 (wait 13)
        # Let me recompute: round(1.25*10) = round(12.5) = 12
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "radius"}, "expected": 12},
        # diameter: R=2.5, encoded=int(round(2.5*10))=25
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "diameter"}, "expected": 25},
    ],
    "circumcircle_properties": [
        # 3-4-5: R=2.5, radius: int(round(2.5*10))=25
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "radius"}, "expected": 25},
        # 3-4-5: diameter: int(round(5.0*10))=50
        {"input": None, "params": {"a": 3, "b": 4, "c": 5, "operation": "diameter"}, "expected": 50},
    ],
    "tangent_circle_properties": [
        # r1=3, r2=5: 3+5=8
        {"input": None, "params": {"r1": 3, "r2": 5}, "expected": 8},
    ],
    "intersection_line_circle": [
        # Line x=0 (a=1,b=0,c=0), circle at (5,0) r=3: dist=|1*5|/1=5>3 -> 0 intersections
        {"input": None, "params": {"center": (5, 0), "radius": 3, "a": 1, "b": 0, "c": 0}, "expected": 0},
        # Line y=0 (a=0,b=1,c=0), circle at (0,0) r=5: dist=0<5 -> 2 intersections
        {"input": None, "params": {"center": (0, 0), "radius": 5, "a": 0, "b": 1, "c": 0}, "expected": 2},
    ],
    "circumference_to_radius": [
        # C = 2*pi -> r = 2*pi/(2*pi) = 1.0
        {"input": None, "params": {"circumference": 2 * math.pi}, "expected": 1.0},
    ],
    "circle_radius": [
        # Simply returns the radius param
        {"input": None, "params": {"center": (0, 0), "radius": 7.0}, "expected": 7.0},
    ],
    "circumradius_equality_condition": [
        # R1=R2=10 -> equal -> 1
        {"input": None, "params": {"R1": 10.0, "R2": 10.0}, "expected": 1},
        # R1=10, R2=15 -> not equal -> 0
        {"input": None, "params": {"R1": 10.0, "R2": 15.0}, "expected": 0},
    ],
    "tangent_length_concentric_circles": [
        # r1=5, r2=13: sqrt(169-25)=sqrt(144)=12.0
        {"input": None, "params": {"r1": 5, "r2": 13}, "expected": 12.0},
        # r1=3, r2=5: sqrt(25-9)=sqrt(16)=4.0
        {"input": None, "params": {"r1": 3, "r2": 5}, "expected": 4.0},
    ],
    # ========================================================================
    # CIRCUMRADIUS (circles.py) - uses seed/offset scaling
    # ========================================================================
    "circumradius": [
        # 3-4-5 right triangle: R=abc/(4K)=60/24=2.5, encoded via seed
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 2627},
        # 5-12-13 right triangle: R=780/240=3.25, encoded via seed
        {"input": None, "params": {"a": 5, "b": 12, "c": 13}, "expected": 5300},
    ],
    # ========================================================================
    # CIRCUMRADIUS LAW OF SINES (circles_2.py) - R = a / (2*sin(A))
    # ========================================================================
    "circumradius_law_of_sines": [
        # a=10, A=30deg: R=10/(2*0.5)=10.0
        {"input": None, "params": {"a": 10, "A": math.radians(30)}, "expected": 10.000000000000002},
        # a=10, A=90deg: R=10/(2*1)=5.0
        {"input": None, "params": {"a": 10, "A": math.radians(90)}, "expected": 5.0},
    ],
    # ========================================================================
    # RADICAL AXIS (circles.py) - ax + by + c = 0
    # ========================================================================
    "radical_axis": [
        # Equal-radius circles at (0,0) and (10,0): 20x + 0y - 100 = 0
        {"input": None, "params": {"circle1": ((0, 0), 5), "circle2": ((10, 0), 5)}, "expected": (20, 0, -100)},
        # Circles r=5 at origin, r=3 at (10,0): 20x + 0y - 84 = 0
        {"input": None, "params": {"circle1": ((0, 0), 5), "circle2": ((10, 0), 3)}, "expected": (20, 0, -84)},
    ],
}
