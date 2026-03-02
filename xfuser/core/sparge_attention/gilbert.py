# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2018 Jakub Červený

import numpy
from numba import njit
import torch
from typing import Tuple


@njit
def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


@njit
def _generate3d_impl(out, idx, x, y, z,
                    ax, ay, az,
                    bx, by, bz,
                    cx, cy, cz):
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    (dax, day, daz) = (sgn(ax), sgn(ay), sgn(az))
    (dbx, dby, dbz) = (sgn(bx), sgn(by), sgn(bz))
    (dcx, dcy, dcz) = (sgn(cx), sgn(cy), sgn(cz))

    # trivial row/column fills
    if h == 1 and d == 1:
        for i in range(0, w):
            out[idx, 0] = x
            out[idx, 1] = y
            out[idx, 2] = z
            idx += 1
            (x, y, z) = (x + dax, y + day, z + daz)
        return idx

    if w == 1 and d == 1:
        for i in range(0, h):
            out[idx, 0] = x
            out[idx, 1] = y
            out[idx, 2] = z
            idx += 1
            (x, y, z) = (x + dbx, y + dby, z + dbz)
        return idx

    if w == 1 and h == 1:
        for i in range(0, d):
            out[idx, 0] = x
            out[idx, 1] = y
            out[idx, 2] = z
            idx += 1
            (x, y, z) = (x + dcx, y + dcy, z + dcz)
        return idx

    (ax2, ay2, az2) = (ax//2, ay//2, az//2)
    (bx2, by2, bz2) = (bx//2, by//2, bz//2)
    (cx2, cy2, cz2) = (cx//2, cy//2, cz//2)

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    # prefer even steps
    if (w2 % 2) and (w > 2):
       (ax2, ay2, az2) = (ax2 + dax, ay2 + day, az2 + daz)

    if (h2 % 2) and (h > 2):
       (bx2, by2, bz2) = (bx2 + dbx, by2 + dby, bz2 + dbz)

    if (d2 % 2) and (d > 2):
       (cx2, cy2, cz2) = (cx2 + dcx, cy2 + dcy, cz2 + dcz)

    # wide case, split in w only
    if (2*w > 3*h) and (2*w > 3*d):
       idx = _generate3d_impl(out, idx, x, y, z,
                              ax2, ay2, az2,
                              bx, by, bz,
                              cx, cy, cz)

       idx = _generate3d_impl(out, idx, x+ax2, y+ay2, z+az2,
                              ax-ax2, ay-ay2, az-az2,
                              bx, by, bz,
                              cx, cy, cz)
       return idx

    # do not split in d
    elif 3*h > 4*d:
       idx = _generate3d_impl(out, idx, x, y, z,
                              bx2, by2, bz2,
                              cx, cy, cz,
                              ax2, ay2, az2)

       idx = _generate3d_impl(out, idx, x+bx2, y+by2, z+bz2,
                              ax, ay, az,
                              bx-bx2, by-by2, bz-bz2,
                              cx, cy, cz)

       idx = _generate3d_impl(out, idx,
                              x+(ax-dax)+(bx2-dbx),
                              y+(ay-day)+(by2-dby),
                              z+(az-daz)+(bz2-dbz),
                              -bx2, -by2, -bz2,
                              cx, cy, cz,
                              -(ax-ax2), -(ay-ay2), -(az-az2))
       return idx

    # do not split in h
    elif 3*d > 4*h:
       idx = _generate3d_impl(out, idx, x, y, z,
                              cx2, cy2, cz2,
                              ax2, ay2, az2,
                              bx, by, bz)

       idx = _generate3d_impl(out, idx, x+cx2, y+cy2, z+cz2,
                              ax, ay, az,
                              bx, by, bz,
                              cx-cx2, cy-cy2, cz-cz2)

       idx = _generate3d_impl(out, idx,
                              x+(ax-dax)+(cx2-dcx),
                              y+(ay-day)+(cy2-dcy),
                              z+(az-daz)+(cz2-dcz),
                              -cx2, -cy2, -cz2,
                              -(ax-ax2), -(ay-ay2), -(az-az2),
                              bx, by, bz)
       return idx

    # regular case, split in all w/h/d
    else:
       idx = _generate3d_impl(out, idx, x, y, z,
                              bx2, by2, bz2,
                              cx2, cy2, cz2,
                              ax2, ay2, az2)

       idx = _generate3d_impl(out, idx, x+bx2, y+by2, z+bz2,
                              cx, cy, cz,
                              ax2, ay2, az2,
                              bx-bx2, by-by2, bz-bz2)

       idx = _generate3d_impl(out, idx,
                              x+(bx2-dbx)+(cx-dcx),
                              y+(by2-dby)+(cy-dcy),
                              z+(bz2-dbz)+(cz-dcz),
                              ax, ay, az,
                              -bx2, -by2, -bz2,
                              -(cx-cx2), -(cy-cy2), -(cz-cz2))

       idx = _generate3d_impl(out, idx,
                              x+(ax-dax)+bx2+(cx-dcx),
                              y+(ay-day)+by2+(cy-dcy),
                              z+(az-daz)+bz2+(cz-dcz),
                              -cx, -cy, -cz,
                              -(ax-ax2), -(ay-ay2), -(az-az2),
                              bx-bx2, by-by2, bz-bz2)

       idx = _generate3d_impl(out, idx,
                              x+(ax-dax)+(bx2-dbx),
                              y+(ay-day)+(by2-dby),
                              z+(az-daz)+(bz2-dbz),
                              -bx2, -by2, -bz2,
                              cx2, cy2, cz2,
                              -(ax-ax2), -(ay-ay2), -(az-az2))
       return idx


@njit
def gilbert3d(width, height, depth):
    """
    Generalized Hilbert ('Gilbert') space-filling curve for arbitrary-sized
    3D rectangular grids. Returns discrete 3D coordinates filling a cuboid
    of size (width x height x depth). Even sizes are recommended in 3D.

    Returns:
        out: array of shape (width*height*depth, 3), dtype int64, rows are (x,y,z).
    """
    n = width * height * depth
    out = numpy.empty((n, 3), dtype=numpy.int64)

    if width >= height and width >= depth:
        _generate3d_impl(out, 0, 0, 0, 0,
                         width, 0, 0,
                         0, height, 0,
                         0, 0, depth)
    elif height >= width and height >= depth:
        _generate3d_impl(out, 0, 0, 0, 0,
                         0, height, 0,
                         width, 0, 0,
                         0, 0, depth)
    else:  # depth >= width and depth >= height
        _generate3d_impl(out, 0, 0, 0, 0,
                         0, 0, depth,
                         width, 0, 0,
                         0, height, 0)

    return out


def curve(depth: int, height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = gilbert3d(width, height, depth)
    inverse_order = coords[:, 2] * height * width + coords[:, 1] * width + coords[:, 0]
    inverse_order = torch.from_numpy(inverse_order).to(device=device)
    order = torch.argsort(inverse_order)
    return order, inverse_order
