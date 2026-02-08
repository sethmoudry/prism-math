"""
Geometry decompositions: 2D/3D shapes, coordinate geometry, trigonometry.

Sections covered:
- 6. GEOMETRY DECOMPOSITIONS
- 20. GEOMETRY DECOMPOSITIONS - EXTENDED
"""

from ..decomposition import Decomposition

GEOMETRY_DECOMPOSITIONS = {
    # =========================================================================
    # 6. GEOMETRY DECOMPOSITIONS
    # =========================================================================

    "triangle_area_base_height": Decomposition(
        expression="divide(multiply(base, height), 2)",
        param_map={"base": "base", "height": "height"},
        notes="Area = base * height / 2"
    ),

    "triangle_area_heron": Decomposition(
        expression="power(multiply(multiply(multiply(s, subtract(s, a)), subtract(s, b)), subtract(s, c)), 0.5)",
        param_map={"s": "s", "a": "a", "b": "b", "c": "c"},
        notes="sqrt(s(s-a)(s-b)(s-c)) where s = semiperimeter"
    ),

    "triangle_semiperimeter": Decomposition(
        expression="divide(add(add(a, b), c), 2)",
        param_map={"a": "a", "b": "b", "c": "c"},
        notes="s = (a+b+c)/2"
    ),

    "circle_area": Decomposition(
        expression="multiply(3.14159265359, power(r, 2))",
        param_map={"r": "r"},
        notes="pi * r^2"
    ),

    "circle_circumference": Decomposition(
        expression="multiply(multiply(2, 3.14159265359), r)",
        param_map={"r": "r"},
        notes="2 * pi * r"
    ),

    "sphere_volume": Decomposition(
        expression="multiply(divide(multiply(4, 3.14159265359), 3), power(r, 3))",
        param_map={"r": "r"},
        notes="(4/3) * pi * r^3"
    ),

    "sphere_surface_area": Decomposition(
        expression="multiply(multiply(4, 3.14159265359), power(r, 2))",
        param_map={"r": "r"},
        notes="4 * pi * r^2"
    ),

    "cylinder_volume": Decomposition(
        expression="multiply(multiply(3.14159265359, power(r, 2)), h)",
        param_map={"r": "r", "h": "h"},
        notes="pi * r^2 * h"
    ),

    "cone_volume": Decomposition(
        expression="divide(multiply(multiply(3.14159265359, power(r, 2)), h), 3)",
        param_map={"r": "r", "h": "h"},
        notes="(1/3) * pi * r^2 * h"
    ),

    "pythagorean": Decomposition(
        expression="power(add(power(a, 2), power(b, 2)), 0.5)",
        param_map={"a": "a", "b": "b"},
        notes="sqrt(a^2 + b^2)"
    ),

    "distance_2d": Decomposition(
        expression="power(add(power(subtract(x2, x1), 2), power(subtract(y2, y1), 2)), 0.5)",
        param_map={"x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2"},
        notes="sqrt((x2-x1)^2 + (y2-y1)^2)"
    ),

    "distance_3d": Decomposition(
        expression="power(add(add(power(subtract(x2, x1), 2), power(subtract(y2, y1), 2)), power(subtract(z2, z1), 2)), 0.5)",
        param_map={"x1": "x1", "y1": "y1", "z1": "z1", "x2": "x2", "y2": "y2", "z2": "z2"},
        notes="3D Euclidean distance"
    ),

    "midpoint_x": Decomposition(
        expression="divide(add(x1, x2), 2)",
        param_map={"x1": "x1", "x2": "x2"},
        notes="(x1 + x2) / 2"
    ),

    "midpoint_y": Decomposition(
        expression="divide(add(y1, y2), 2)",
        param_map={"y1": "y1", "y2": "y2"},
        notes="(y1 + y2) / 2"
    ),

    "slope": Decomposition(
        expression="divide(subtract(y2, y1), subtract(x2, x1))",
        param_map={"x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2"},
        notes="(y2 - y1) / (x2 - x1)"
    ),

    "rectangle_area": Decomposition(
        expression="multiply(length, width)",
        param_map={"length": "length", "width": "width"},
        notes="length * width"
    ),

    "rectangle_perimeter": Decomposition(
        expression="multiply(2, add(length, width))",
        param_map={"length": "length", "width": "width"},
        notes="2 * (length + width)"
    ),

    "parallelogram_area": Decomposition(
        expression="multiply(base, height)",
        param_map={"base": "base", "height": "height"},
        notes="base * height"
    ),

    "trapezoid_area": Decomposition(
        expression="divide(multiply(add(a, b), h), 2)",
        param_map={"a": "a", "b": "b", "h": "h"},
        notes="(a + b) * h / 2"
    ),

    # =========================================================================
    # ADDITIONAL GEOMETRY
    # =========================================================================

    "regular_polygon_area": Decomposition(
        expression="divide(multiply(multiply(n, s), s), multiply(4, tan(divide(3.14159265359, n))))",
        param_map={"n": "n", "s": "s"},
        notes="Area of regular n-gon with side s"
    ),

    "regular_polygon_interior_angle": Decomposition(
        expression="divide(multiply(subtract(n, 2), 180), n)",
        param_map={"n": "n"},
        notes="Interior angle = (n-2)*180/n degrees"
    ),

    "regular_polygon_exterior_angle": Decomposition(
        expression="divide(360, n)",
        param_map={"n": "n"},
        notes="Exterior angle = 360/n degrees"
    ),

    "regular_polygon_diagonal_count": Decomposition(
        expression="divide(multiply(n, subtract(n, 3)), 2)",
        param_map={"n": "n"},
        notes="Diagonals = n(n-3)/2"
    ),

    "triangle_circumradius": Decomposition(
        expression="divide(multiply(multiply(a, b), c), multiply(4, area))",
        param_map={"a": "a", "b": "b", "c": "c", "area": "area"},
        notes="R = abc/(4*Area)"
    ),

    "triangle_inradius": Decomposition(
        expression="divide(area, s)",
        param_map={"area": "area", "s": "s"},
        notes="r = Area/s where s = semiperimeter"
    ),

    "shoelace_formula": Decomposition(
        expression="divide(absolute_value(subtract(multiply(x1, subtract(y2, yn)), add(multiply(x2, subtract(y3, y1)), more_terms))), 2)",
        param_map={"x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2", "yn": "yn", "more_terms": "more_terms"},
        notes="Shoelace formula for polygon area"
    ),

    "picks_theorem": Decomposition(
        expression="add(subtract(area, divide(boundary_points, 2)), 1)",
        param_map={"area": "area", "boundary_points": "boundary_points"},
        notes="Interior points = Area - B/2 + 1"
    ),

    "ellipse_area": Decomposition(
        expression="multiply(multiply(3.14159265359, a), b)",
        param_map={"a": "a", "b": "b"},
        notes="pi * a * b"
    ),

    "sector_area": Decomposition(
        expression="divide(multiply(power(r, 2), theta), 2)",
        param_map={"r": "r", "theta": "theta"},
        notes="Area = r^2 * theta / 2 (theta in radians)"
    ),

    "arc_length": Decomposition(
        expression="multiply(r, theta)",
        param_map={"r": "r", "theta": "theta"},
        notes="Arc length = r * theta (theta in radians)"
    ),

    "chord_length": Decomposition(
        expression="multiply(multiply(2, r), sin(divide(theta, 2)))",
        param_map={"r": "r", "theta": "theta"},
        notes="Chord = 2r*sin(theta/2)"
    ),

    "law_of_cosines": Decomposition(
        expression="power(subtract(add(power(a, 2), power(b, 2)), multiply(multiply(multiply(2, a), b), cos_C)), 0.5)",
        param_map={"a": "a", "b": "b", "cos_C": "cos_C"},
        notes="c = sqrt(a^2 + b^2 - 2ab*cos(C))"
    ),

    "law_of_sines_side": Decomposition(
        expression="divide(multiply(a, sin_B), sin_A)",
        param_map={"a": "a", "sin_A": "sin_A", "sin_B": "sin_B"},
        notes="b = a * sin(B) / sin(A)"
    ),

    "triangle_area_two_sides_angle": Decomposition(
        expression="divide(multiply(multiply(a, b), sin_C), 2)",
        param_map={"a": "a", "b": "b", "sin_C": "sin_C"},
        notes="Area = (1/2) * a * b * sin(C)"
    ),

    "box_volume": Decomposition(
        expression="multiply(multiply(l, w), h)",
        param_map={"l": "l", "w": "w", "h": "h"},
        notes="Volume = l * w * h"
    ),

    "box_surface_area": Decomposition(
        expression="multiply(2, add(add(multiply(l, w), multiply(w, h)), multiply(h, l)))",
        param_map={"l": "l", "w": "w", "h": "h"},
        notes="Surface area = 2(lw + wh + hl)"
    ),

    "box_diagonal": Decomposition(
        expression="power(add(add(power(l, 2), power(w, 2)), power(h, 2)), 0.5)",
        param_map={"l": "l", "w": "w", "h": "h"},
        notes="Diagonal = sqrt(l^2 + w^2 + h^2)"
    ),

    "tetrahedron_volume": Decomposition(
        expression="divide(multiply(multiply(a, a), a), multiply(6, power(2, 0.5)))",
        param_map={"a": "a"},
        notes="Regular tetrahedron volume = a^3/(6*sqrt(2))"
    ),

    "pyramid_volume": Decomposition(
        expression="divide(multiply(base_area, h), 3)",
        param_map={"base_area": "base_area", "h": "h"},
        notes="Volume = (1/3) * base_area * height"
    ),

    "frustum_volume": Decomposition(
        expression="divide(multiply(multiply(h, 3.14159265359), add(add(power(R, 2), power(r, 2)), multiply(R, r))), 3)",
        param_map={"h": "h", "R": "R", "r": "r"},
        notes="Frustum volume = (h*pi/3)*(R^2 + r^2 + R*r)"
    ),

    "torus_volume": Decomposition(
        expression="multiply(multiply(multiply(2, power(3.14159265359, 2)), R), power(r, 2))",
        param_map={"R": "R", "r": "r"},
        notes="Torus volume = 2*pi^2*R*r^2"
    ),

    "torus_surface_area": Decomposition(
        expression="multiply(multiply(multiply(4, power(3.14159265359, 2)), R), r)",
        param_map={"R": "R", "r": "r"},
        notes="Torus surface area = 4*pi^2*R*r"
    ),

    "dot_product_2d": Decomposition(
        expression="add(multiply(x1, x2), multiply(y1, y2))",
        param_map={"x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2"},
        notes="(x1,y1).(x2,y2) = x1*x2 + y1*y2"
    ),

    "cross_product_2d": Decomposition(
        expression="subtract(multiply(x1, y2), multiply(y1, x2))",
        param_map={"x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2"},
        notes="(x1,y1) x (x2,y2) = x1*y2 - y1*x2"
    ),

    "point_to_line_distance": Decomposition(
        expression="divide(absolute_value(add(add(multiply(a, x0), multiply(b, y0)), c)), power(add(power(a, 2), power(b, 2)), 0.5))",
        param_map={"a": "a", "b": "b", "c": "c", "x0": "x0", "y0": "y0"},
        notes="Distance from (x0,y0) to ax+by+c=0"
    ),

    "reflection_point": Decomposition(
        expression="subtract(multiply(2, line_x), x)",
        param_map={"x": "x", "line_x": "line_x"},
        notes="Reflect x across vertical line x=line_x"
    ),

    "rotation_x": Decomposition(
        expression="subtract(multiply(x, cos_theta), multiply(y, sin_theta))",
        param_map={"x": "x", "y": "y", "cos_theta": "cos_theta", "sin_theta": "sin_theta"},
        notes="x' = x*cos(theta) - y*sin(theta)"
    ),

    "rotation_y": Decomposition(
        expression="add(multiply(x, sin_theta), multiply(y, cos_theta))",
        param_map={"x": "x", "y": "y", "cos_theta": "cos_theta", "sin_theta": "sin_theta"},
        notes="y' = x*sin(theta) + y*cos(theta)"
    ),

    "line_intersection_x": Decomposition(
        expression="divide(subtract(multiply(b1, c2), multiply(b2, c1)), subtract(multiply(a1, b2), multiply(a2, b1)))",
        param_map={"a1": "a1", "b1": "b1", "c1": "c1", "a2": "a2", "b2": "b2", "c2": "c2"},
        notes="x-coord of intersection of a1x+b1y=c1 and a2x+b2y=c2"
    ),

    "angle_between_vectors": Decomposition(
        expression="acos(divide(dot_product, multiply(mag1, mag2)))",
        param_map={"dot_product": "dot_product", "mag1": "mag1", "mag2": "mag2"},
        notes="theta = acos(u.v / (|u|*|v|))"
    ),

    "vector_magnitude": Decomposition(
        expression="power(add(power(x, 2), power(y, 2)), 0.5)",
        param_map={"x": "x", "y": "y"},
        notes="|v| = sqrt(x^2 + y^2)"
    ),

    "unit_vector_x": Decomposition(
        expression="divide(x, power(add(power(x, 2), power(y, 2)), 0.5))",
        param_map={"x": "x", "y": "y"},
        notes="x-component of unit vector"
    ),

    "unit_vector_y": Decomposition(
        expression="divide(y, power(add(power(x, 2), power(y, 2)), 0.5))",
        param_map={"x": "x", "y": "y"},
        notes="y-component of unit vector"
    ),

    "triangle_centroid_x": Decomposition(
        expression="divide(add(add(x1, x2), x3), 3)",
        param_map={"x1": "x1", "x2": "x2", "x3": "x3"},
        notes="Centroid x = (x1+x2+x3)/3"
    ),

    "triangle_centroid_y": Decomposition(
        expression="divide(add(add(y1, y2), y3), 3)",
        param_map={"y1": "y1", "y2": "y2", "y3": "y3"},
        notes="Centroid y = (y1+y2+y3)/3"
    ),

    "barycentric_to_cartesian_x": Decomposition(
        expression="add(add(multiply(u, x1), multiply(v, x2)), multiply(w, x3))",
        param_map={"u": "u", "v": "v", "w": "w", "x1": "x1", "x2": "x2", "x3": "x3"},
        notes="x = u*x1 + v*x2 + w*x3"
    ),

    "euler_line_length": Decomposition(
        expression="multiply(3, power(add(power(subtract(O_x, G_x), 2), power(subtract(O_y, G_y), 2)), 0.5))",
        param_map={"O_x": "O_x", "O_y": "O_y", "G_x": "G_x", "G_y": "G_y"},
        notes="Euler line: OH = 3*OG"
    ),

    "nine_point_circle_radius": Decomposition(
        expression="divide(R, 2)",
        param_map={"R": "R"},
        notes="Nine-point circle radius = R/2"
    ),

    "ptolemy_product": Decomposition(
        expression="multiply(AC, BD)",
        param_map={"AC": "AC", "BD": "BD"},
        notes="AC*BD = AB*CD + AD*BC for cyclic quad"
    ),

    "stewart_theorem": Decomposition(
        expression="add(add(multiply(multiply(b, b), m), multiply(multiply(c, c), n)), multiply(multiply(a, m), n))",
        param_map={"a": "a", "b": "b", "c": "c", "m": "m", "n": "n"},
        notes="b^2*m + c^2*n = a(d^2 + mn) Stewart's theorem"
    ),

    "power_of_point": Decomposition(
        expression="subtract(power(d, 2), power(r, 2))",
        param_map={"d": "d", "r": "r"},
        notes="Power = d^2 - r^2"
    ),

    "radical_axis_perpendicular_distance": Decomposition(
        expression="divide(subtract(add(power(d, 2), power(r1, 2)), power(r2, 2)), multiply(2, d))",
        param_map={"d": "d", "r1": "r1", "r2": "r2"},
        notes="Distance from center1 to radical axis"
    ),

    "external_tangent_length": Decomposition(
        expression="power(subtract(power(d, 2), power(subtract(r1, r2), 2)), 0.5)",
        param_map={"d": "d", "r1": "r1", "r2": "r2"},
        notes="External tangent length between circles"
    ),

    "internal_tangent_length": Decomposition(
        expression="power(subtract(power(d, 2), power(add(r1, r2), 2)), 0.5)",
        param_map={"d": "d", "r1": "r1", "r2": "r2"},
        notes="Internal tangent length between circles"
    ),
}
