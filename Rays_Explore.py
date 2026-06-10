import math

"""
障碍物检测和边界检测
计算射线和圆形障碍物的交点
"""
def ray_to_obstacle(origin,direction,obs_pos,obs_radius,max_dist):
    f=origin-obs_pos
    if f.length_squared() <= obs_radius ** 2:
        return 0
    a=direction.dot(direction)
    b=2*f.dot(direction)
    c=f.dot(f)-obs_radius**2

    discriminant= b**2-4*a*c

    if discriminant<0:
        return None
    sqr_discriminant= math.sqrt(discriminant)

    t1=(-b+sqr_discriminant)/2*a
    t2=(-b-sqr_discriminant)/2*a
    valid_hits = []
    if 0 <= t1 <= max_dist:
        valid_hits.append(t1)

    if 0 <= t2 <= max_dist:
        valid_hits.append(t2)

    if not valid_hits:
        return None

    return min(valid_hits)


def ray_boundary_distance(origin, direction, width, height, max_dist):
    distances = []

    if direction.x > 0:
        distances.append((width - origin.x) / direction.x)
    elif direction.x < 0:
        distances.append((0 - origin.x) / direction.x)

    if direction.y > 0:
        distances.append((height - origin.y) / direction.y)
    elif direction.y < 0:
        distances.append((0 - origin.y) / direction.y)

    positive_distances = [d for d in distances if d >= 0]

    if not positive_distances:
        return None

    distance = min(positive_distances)

    if distance <= max_dist:
        return distance

    return None