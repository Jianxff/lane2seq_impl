from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2

DCOLORS = [(110, 30, 30), (75, 25, 230), (75, 180, 60), (200, 130, 0), (48, 130, 245), (180, 30, 145),
           (0, 0, 255), (24, 140, 34), (255, 0, 0), (0, 255, 255),  # the main ones
           (40, 110, 170), (200, 250, 255), (255, 190, 230), (0, 0, 128), (195, 255, 170),
           (0, 128, 128), (195, 255, 170), (75, 25, 230)]

def get_dcolors(total_length):
    center = 8
    dr = (total_length - 1) // 2
    dl = (total_length - 1) - dr
    colors = DCOLORS[center - dl:center + dr + 1]
    return colors


def llamas_extend_lane(lane):
    """Extends marker closest to the camera
    Adds an extra marker that reaches the end of the image
    Parameters
    ----------
    lane : iterable of markers
    """
    # Unfortunately, we did not store markers beyond the image plane. That hurts us now
    # z is the orthongal distance to the car. It's good enough

    # The markers are automatically detected, mapped, and labeled. There exist faulty ones,
    # e.g., horizontal markers which need to be filtered
    filtered_markers = filter(
        lambda x: (x['pixel_start']['y'] != x['pixel_end']['y'] and x[
            'pixel_start']['x'] != x['pixel_end']['x']), lane['markers'])
    filtered_markers = list(filtered_markers)
    if len(filtered_markers) == 0:
        return lane
    # might be the first marker in the list but not guaranteed
    closest_marker = min(filtered_markers, key=lambda x: x['world_start']['z'])

    if closest_marker['world_start'][
            'z'] < 0:  # This one likely equals "if False"
        return lane

    # World marker extension approximation
    x_gradient = (closest_marker['world_end']['x'] - closest_marker['world_start']['x']) /\
        (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])
    y_gradient = (closest_marker['world_end']['y'] - closest_marker['world_start']['y']) /\
        (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])

    zero_x = closest_marker['world_start']['x'] - (
        closest_marker['world_start']['z'] - 1) * x_gradient
    zero_y = closest_marker['world_start']['y'] - (
        closest_marker['world_start']['z'] - 1) * y_gradient

    # Pixel marker extension approximation
    pixel_x_gradient = (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x']) /\
        (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y'])
    pixel_y_gradient = (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y']) /\
        (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x'])

    pixel_zero_x = closest_marker['pixel_start']['x'] + (
        716 - closest_marker['pixel_start']['y']) * pixel_x_gradient
    if pixel_zero_x < 0:
        left_y = closest_marker['pixel_start'][
            'y'] - closest_marker['pixel_start']['x'] * pixel_y_gradient
        new_pixel_point = (0, left_y)
    elif pixel_zero_x > 1276:
        right_y = closest_marker['pixel_start']['y'] + (
            1276 - closest_marker['pixel_start']['x']) * pixel_y_gradient
        new_pixel_point = (1276, right_y)
    else:
        new_pixel_point = (pixel_zero_x, 716)

    new_marker = {
        'lane_marker_id': 'FAKE',
        'world_end': {
            'x': closest_marker['world_start']['x'],
            'y': closest_marker['world_start']['y'],
            'z': closest_marker['world_start']['z']
        },
        'world_start': {
            'x': zero_x,
            'y': zero_y,
            'z': 1
        },
        'pixel_end': {
            'x': closest_marker['pixel_start']['x'],
            'y': closest_marker['pixel_start']['y']
        },
        'pixel_start': {
            'x': int(round(new_pixel_point[0])),
            'y': int(round(new_pixel_point[1]))
        }
    }
    lane['markers'].insert(0, new_marker)

    return lane


def llamas_sample_points_horizontal(lane):
    """ Markers are given by start and endpoint. This one adds extra points
    which need to be considered for the interpolation. Otherwise the spline
    could arbitrarily oscillate between start and end of the individual markers
    Parameters
    ----------
    lane: polyline, in theory but there are artifacts which lead to inconsistencies
            in ordering. There may be parallel lines. The lines may be dashed. It's messy.
    ypp: y-pixels per point, e.g. 10 leads to a point every ten pixels
    between_markers : bool, interpolates inbetween dashes
    Notes
    -----
    Especially, adding points in the lower parts of the image (high y-values) because
    the start and end points are too sparse.
    Removing upper lane markers that have starting and end points mapped into the same pixel.
    """

    # Collect all x values from all markers along a given line. There may be multiple
    # intersecting markers, i.e., multiple entries for some y values
    x_values = [[] for i in range(717)]
    for marker in lane['markers']:
        x_values[marker['pixel_start']['y']].append(
            marker['pixel_start']['x'])

        height = marker['pixel_start']['y'] - marker['pixel_end']['y']
        if height > 2:
            slope = (marker['pixel_end']['x'] -
                        marker['pixel_start']['x']) / height
            step_size = (marker['pixel_start']['y'] -
                            marker['pixel_end']['y']) / float(height)
            for i in range(height + 1):
                x = marker['pixel_start']['x'] + slope * step_size * i
                y = marker['pixel_start']['y'] - step_size * i
                x_values[int(round(y))].append(int(round(x)))

    # Calculate average x values for each y value
    for y, xs in enumerate(x_values):
        if not xs:
            x_values[y] = -1
        else:
            x_values[y] = sum(xs) / float(len(xs))

    # # interpolate between markers
    current_y = 0
    while x_values[current_y] == -1:  # skip missing first entries
        current_y += 1

    # Also possible using numpy.interp when accounting for beginning and end
    next_set_y = 0
    try:
        while current_y < 717:
            if x_values[current_y] != -1:  # set. Nothing to be done
                current_y += 1
                continue

            # Finds target x value for interpolation
            while next_set_y <= current_y or x_values[next_set_y] == -1:
                next_set_y += 1
                if next_set_y >= 717:
                    raise StopIteration

            x_values[current_y] = x_values[current_y - 1] + (x_values[next_set_y] - x_values[current_y - 1]) /\
                (next_set_y - current_y + 1)
            current_y += 1

    except StopIteration:
        pass  # Done with lane

    return x_values


def remove_consecutive_duplicates(x):
    x = sorted(x, key=lambda x: (x[0], x[1]))
    y = []
    for t in x:
        if len(y) > 0 and y[-1] == t:
            continue
        y.append(t)
    return y


def interpolate_lane(points, n: int = 50):
    """Spline interpolation of a lane. Used on the predictions"""
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., n)
    return np.array(splev(u, tck)).T


def draw_lane(lane, img=None, img_shape=None, width=30):
    """Draw a lane (a list of points) on an image by drawing a line with width `width` through each
    pair of points i and i+i"""
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(1, ), thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(717, 1276)):
    """For each lane in xs, compute its Intersection Over Union (IoU) with each lane in ys by drawing the lanes on
    an image"""
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # IoU by the definition: sum all intersections (binary and) and divide by the sum of the union (binary or)
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def culane_metric(pred,
                  anno,
                  width=30,
                  iou_thresholds=[0.5],
                  img_shape=(717, 1276)):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]
    
    if len(pred) == 0 or len(anno) == 0:
        return _metric
    
    ious = discrete_cross_iou(pred,
                            anno,
                            width=width,
                            img_shape=img_shape)
    row_ind, col_ind = linear_sum_assignment(1 - ious)
    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        _metric[thr] = [tp, fp, fn]

    return _metric