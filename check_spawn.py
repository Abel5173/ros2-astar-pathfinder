import math

origin_x = -7.75
origin_y = -6.25
res = 0.25

# Furniture from model.sdf (approximate bounding boxes)
furniture = [
    ('table',        -2.657,  2.424, 0.0, 0.75, 0.40),   # large table
    ('bookshelf',    -6.544,  5.195, 0.0, 0.45, 0.20),
    ('bookshelf_0',   4.724,  5.179, 0.0, 0.45, 0.20),
    ('bookshelf_1',   5.644,  5.179, 0.0, 0.45, 0.20),
    ('cabinet',      -5.472, -1.576, 0.0, 0.23, 0.23),
    ('cabinet_0',    -5.473, -2.065, 0.0, 0.23, 0.23),
    ('cabinet_1',    -7.184,  1.248, 1.5708, 0.23, 0.23),
    ('cafe_table',    6.359, -3.192, 0.0, 0.46, 0.46),
    ('cafe_table_0',  6.359, -2.278, 0.0, 0.46, 0.46),
    ('trash_can',     1.883,  1.912, 0.0, 0.20, 0.20),
    ('trash_can_0',  -4.694,  4.894, 0.0, 0.20, 0.20),
    ('table_marble',  4.883,  2.926, 0.0, 0.50, 0.50),
    ('mailbox',       0.883, -0.576, 0.0, 0.20, 0.20),
]

def is_near_furniture(wx, wy, margin=0.6):
    for name, cx, cy, yaw, hl, ht in furniture:
        cos_y = math.cos(yaw); sin_y = math.sin(yaw)
        dx = wx - cx; dy = wy - cy
        lx = dx*cos_y + dy*sin_y
        ly = -dx*sin_y + dy*cos_y
        if abs(lx) <= hl + margin and abs(ly) <= ht + margin:
            return name
    return None

# Candidate waypoints — one per room, well away from furniture and walls
candidates = [
    ('Left room',              -6.0,  1.5),
    ('Left room alt',          -6.0, -1.0),
    ('Left room alt2',         -6.5,  1.5),
    ('Centre corridor upper',  -2.5,  2.0),
    ('Centre corridor alt',    -3.5,  2.5),
    ('Centre corridor alt2',   -1.5,  2.5),
    ('Right upper room',        4.0,  3.5),
    ('Right upper alt',         3.0,  4.0),
    ('Right upper alt2',        5.5,  4.0),
    ('Centre lower',           -2.5, -2.5),
    ('Centre lower alt',       -1.0, -3.0),
    ('Bottom corridor',         1.0, -4.0),
    ('Bottom corridor alt',     2.0, -4.5),
]

print(f'{"Status":8s}  {"Label":30s}  {"World":18s}  {"Grid":10s}  {"Furniture"}')
print('-'*80)
for label, wx, wy in candidates:
    r = round((wy - origin_y) / res)
    c = round((wx - origin_x) / res)
    furn = is_near_furniture(wx, wy)
    status = 'BLOCKED' if furn else 'OK'
    print(f'{status:8s}  {label:30s}  ({wx:5.1f},{wy:5.1f})  ({r:2d},{c:2d})  {furn or ""}')
