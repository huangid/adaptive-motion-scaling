"""Visual lab models from LabUtopia: a test tube rack (6 holes, shortened on request) and one test tube.

LabUtopia (Li et al., 2025, https://arxiv.org/abs/2505.22634) publishes its lab assets at
https://huggingface.co/datasets/Ruinwalker/LabUtopia-Dataset under CC BY-NC 4.0
(non-commercial use, with attribution). They are downloaded into the Hugging Face cache at
runtime and are never committed here.

The models are used for appearance only. Their thin glass and plastic shells make poor
collision geometry, so the task keeps its own solid colliders (a cylinder and thick boxes),
made invisible, and lays these models over them.

All dimensions below were measured from the pinned revision.
"""

from pathlib import Path

import numpy as np

LABUTOPIA_REPO = "Ruinwalker/LabUtopia-Dataset"
LABUTOPIA_REVISION = "e622f0ec8123c9098d913ea56a4116fd83a29662"
_PATTERNS = ["Instruments/TubeRack/**", "Instruments/TestTube/TestTube_02/**"]

RACK_FILE = "Instruments/TubeRack/TubeRack.usd"
TUBE_SET_FILE = "Instruments/TestTube/TestTube_02/TestTube_02.usd"

# --- TubeRack.usd (metres, z up). Six holes in a row along its local y axis, at x = 0. ---
RACK = dict(
    n_holes=6,
    hole_radius=0.0098,        # round holes, both plates
    hole_pitch=0.0235,         # centre to centre
    first_hole_y=-0.0743,      # holes at first_hole_y + i * hole_pitch (the origin is off-centre)
    depth=0.0626,              # extent across the row (local x)
    length=0.1564,             # extent along the row (local y)
    top_z=0.0947,              # top of the upper plate
    base_z=-0.0016,            # underside of the base
)


def rack_row_center_y(n_holes: int) -> float:
    """Local y of the middle of the hole row once the rack is shortened to ``n_holes``."""
    return RACK["first_hole_y"] + (n_holes - 1) / 2.0 * RACK["hole_pitch"]


def rack_length(n_holes: int) -> float:
    """Model length along the row once shortened to ``n_holes`` (end panels unchanged)."""
    return RACK["length"] - (RACK["n_holes"] - n_holes) * RACK["hole_pitch"]


# --- TestTube_02.usd (centimetres, z up). Five tubes in a glass stand, in a row along x. ---
TUBE_SET = dict(
    tube_x=(-8.53, -4.29, 0.01, 4.29, 8.55),   # tube axes; all at y = 0
    keep_radius=1.6,           # faces within this of a tube's axis belong to that tube
    stand_meshes=("mesh_002", "mesh_004"),   # the stand's posts and plates, not tubes
)
# Liquid colour per tube, left to right, for choosing one.
TUBE_LIQUID = ("orange", "orange", "purple", "purple", "orange")


# Bump when _extract_tube / _prepare_rack change, so stale cached files are not reused.
_EXTRACT_VERSION = "v2"
_RACK_VERSION = "v2"


def labutopia_root() -> Path:
    """Local snapshot of the two LabUtopia files (downloaded on first use)."""
    from huggingface_hub import snapshot_download

    kwargs = dict(repo_type="dataset", revision=LABUTOPIA_REVISION, allow_patterns=_PATTERNS)
    try:
        return Path(snapshot_download(LABUTOPIA_REPO, local_files_only=True, **kwargs))
    except Exception:
        return Path(snapshot_download(LABUTOPIA_REPO, **kwargs))


def rack_usd(n_holes: int) -> Path:
    """Path to the prepared rack model with ``n_holes`` holes (1-6), built once and cached.

    Two fixes are applied to LabUtopia's model:

    1. Its plate surfaces around the holes are stored as large polygons (27-44 corners)
       with curved, concave outlines. The renderer splits those into a fan of triangles,
       which is only correct for convex shapes, so triangles cut across the holes and
       make them look irregular. They are re-triangulated here by ear clipping.
    2. The model has six holes. Fewer are made by collapsing the band holding the last
       ``6 - n_holes`` holes to zero width and sliding the far end panel inward, so the
       plates and side panels stay continuous instead of being cut open.
    """
    if not 1 <= n_holes <= RACK["n_holes"]:
        raise ValueError(f"n_holes must be 1..{RACK['n_holes']}, got {n_holes}")
    root = labutopia_root()
    out = root / f"Instruments/TubeRack/TubeRack_{n_holes}holes_{_RACK_VERSION}.usd"
    if not out.exists():
        _prepare_rack(root / RACK_FILE, out, n_holes)
    return out


def _prepare_rack(src: Path, out: Path, n_holes: int):
    from pxr import Usd, UsdGeom, Vt  # needs a running Isaac Sim app

    pitch = RACK["hole_pitch"]
    # Band to remove: from midway after the last kept hole, one pitch per removed hole.
    c1 = RACK["first_hole_y"] + (n_holes - 0.5) * pitch
    width = (RACK["n_holes"] - n_holes) * pitch
    c2 = c1 + width

    layer = Usd.Stage.Open(str(src)).Flatten()
    tmp = out.with_suffix(".tmp.usd")
    layer.Export(str(tmp))
    stage = Usd.Stage.Open(str(tmp))
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        _triangulate_large_faces(mesh)
        if width == 0.0:
            continue
        local = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        world = np.c_[local, np.ones(len(local))] @ xf
        y = world[:, 1]
        # Band vertices collapse onto c1 (their faces become zero-width); beyond it, shift back.
        world[:, 1] = np.where(y >= c2, y - width, np.where(y > c1, c1, y))
        local = (world @ np.linalg.inv(xf))[:, :3]
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
        mesh.GetExtentAttr().Set(Vt.Vec3fArray([tuple(local.min(0)), tuple(local.max(0))]))
    stage.GetRootLayer().Export(str(out))
    del stage
    tmp.unlink(missing_ok=True)


def _ear_clip(poly: np.ndarray) -> list:
    """Triangulate a simple 2D polygon (n, 2) by ear clipping. Returns corner index triples."""
    n = len(poly)
    area2 = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    sign = 1.0 if area2 >= 0 else -1.0
    eps = 1e-12 * max(1.0, float(np.abs(poly).max()) ** 2)
    remaining = list(range(n))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    while len(remaining) > 3:
        m = len(remaining)
        clipped = False
        for k in range(m):
            i0, i1, i2 = remaining[k - 1], remaining[k], remaining[(k + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if sign * cross(a, b, c) <= eps:
                continue                                  # reflex or degenerate corner
            inside = False
            for j in remaining:
                if j in (i0, i1, i2):
                    continue
                q = poly[j]
                if (sign * cross(a, b, q) > eps and sign * cross(b, c, q) > eps
                        and sign * cross(c, a, q) > eps):
                    inside = True
                    break
            if inside:
                continue
            tris.append((i0, i1, i2))
            remaining.pop(k)
            clipped = True
            break
        if not clipped:
            # Only collinear or duplicated corners left: drop the flattest one.
            k = min(range(m), key=lambda k: abs(cross(poly[remaining[k - 1]], poly[remaining[k]],
                                                      poly[remaining[(k + 1) % m]])))
            tris.append((remaining[k - 1], remaining[k], remaining[(k + 1) % m]))
            remaining.pop(k)
    tris.append(tuple(remaining))
    return tris


def _triangulate_large_faces(mesh, max_corners: int = 4):
    """Replace faces with more than ``max_corners`` corners by properly clipped triangles.

    Face-varying primvars (normals, UVs) follow their face corners; uniform ones are
    repeated per new triangle. Raises if the triangles do not cover the original area.
    """
    from pxr import UsdGeom, Vt

    if UsdGeom.Subset.GetAllGeomSubsets(mesh):
        raise RuntimeError(f"{mesh.GetPath()} has GeomSubsets; re-triangulation would break them")
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    if counts.max() <= max_corners:
        return
    starts = np.r_[0, np.cumsum(counts)[:-1]]

    new_counts, corner_src, face_src = [], [], []   # corner_src: old face-vertex slot per new corner
    for f, (st, c) in enumerate(zip(starts, counts)):
        if c <= max_corners:
            new_counts.append(c)
            corner_src.extend(range(st, st + c))
            face_src.append(f)
            continue
        poly = pts[idx[st:st + c]]
        # Newell normal; project onto the plane of its two smallest components.
        nrm = np.cross(poly, np.roll(poly, -1, axis=0)).sum(0)
        drop = int(np.argmax(np.abs(nrm)))
        keep = [a for a in range(3) if a != drop]
        poly2 = poly[:, keep]
        tris = _ear_clip(poly2)
        tri_area = sum(abs(np.cross(poly2[b] - poly2[a], poly2[d] - poly2[a])) / 2.0 for a, b, d in tris)
        x, y = poly2[:, 0], poly2[:, 1]
        poly_area = abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0
        if not np.isclose(tri_area, poly_area, rtol=1e-3, atol=1e-10):
            raise RuntimeError(f"{mesh.GetPath()} face {f}: triangles cover {tri_area:.3e}, "
                               f"polygon is {poly_area:.3e}")
        for tri in tris:
            new_counts.append(3)
            corner_src.extend(st + j for j in tri)
            face_src.append(f)

    corner_src = np.asarray(corner_src)
    face_src = np.asarray(face_src)
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([int(c) for c in new_counts]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(idx[corner_src].tolist()))
    normals = mesh.GetNormalsAttr()
    if normals.HasValue():
        interp = mesh.GetNormalsInterpolation()
        vals = np.asarray(normals.Get())
        if interp == UsdGeom.Tokens.faceVarying:
            normals.Set(Vt.Vec3fArray.FromNumpy(vals[corner_src].astype(np.float32)))
        elif interp == UsdGeom.Tokens.uniform:
            normals.Set(Vt.Vec3fArray.FromNumpy(vals[face_src].astype(np.float32)))
    for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        interp = pv.GetInterpolation()
        sel = corner_src if interp == UsdGeom.Tokens.faceVarying else face_src if interp == UsdGeom.Tokens.uniform else None
        if sel is None:
            continue
        if pv.IsIndexed():
            pv.SetIndices(Vt.IntArray(np.asarray(pv.GetIndices())[sel].tolist()))
        else:
            vals = pv.Get()
            pv.Set(type(vals)([vals[int(i)] for i in sel]))


def single_tube_usd(tube_index: int) -> Path:
    """Path to a USD holding one tube (glass, stopper, liquid) cut out of TestTube_02.

    Built once and cached next to the download. Faces are kept when their centroid lies
    within ``keep_radius`` of the chosen tube's axis; the stand meshes are dropped. The
    result stays in the source's centimetre units and coordinates.
    """
    if not 0 <= tube_index < len(TUBE_SET["tube_x"]):
        raise ValueError(f"tube_index must be 0..{len(TUBE_SET['tube_x']) - 1}, got {tube_index}")
    root = labutopia_root()
    out = root / f"Instruments/TestTube/TestTube_02/single_tube_{tube_index}_{_EXTRACT_VERSION}.usd"
    if not out.exists():
        _extract_tube(root / TUBE_SET_FILE, out, TUBE_SET["tube_x"][tube_index])
    return out


def _extract_tube(src: Path, out: Path, axis_x: float):
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt  # needs a running Isaac Sim app

    layer = Usd.Stage.Open(str(src)).Flatten()
    tmp = out.with_suffix(".tmp.usd")
    layer.Export(str(tmp))
    stage = Usd.Stage.Open(str(tmp))

    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.GetName() in TUBE_SET["stand_meshes"]:
            stage.RemovePrim(prim.GetPath())
            continue
        mesh = UsdGeom.Mesh(prim)
        if UsdGeom.Subset.GetAllGeomSubsets(mesh):
            raise RuntimeError(f"{prim.GetPath()} has GeomSubsets; face filtering would break them")
        local = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        pts = (np.c_[local, np.ones(len(local))] @ xf)[:, :3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        centroid = np.add.reduceat(pts[idx], starts) / counts[:, None]
        keep = np.hypot(centroid[:, 0] - axis_x, centroid[:, 1]) < TUBE_SET["keep_radius"]
        if not keep.any():
            stage.RemovePrim(prim.GetPath())
            continue
        fv = np.repeat(keep, counts)
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts[keep].tolist()))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(idx[fv].tolist()))
        normals = mesh.GetNormalsAttr()
        if normals.HasValue():
            interp = mesh.GetNormalsInterpolation()
            if interp == UsdGeom.Tokens.faceVarying:
                normals.Set(Vt.Vec3fArray.FromNumpy(np.asarray(normals.Get())[fv].astype(np.float32)))
            elif interp == UsdGeom.Tokens.uniform:
                normals.Set(Vt.Vec3fArray.FromNumpy(np.asarray(normals.Get())[keep].astype(np.float32)))
        for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            interp = pv.GetInterpolation()
            mask = fv if interp == UsdGeom.Tokens.faceVarying else keep if interp == UsdGeom.Tokens.uniform else None
            if mask is None:
                continue
            if pv.IsIndexed():
                pv.SetIndices(Vt.IntArray(np.asarray(pv.GetIndices())[mask].tolist()))
            else:
                vals = pv.Get()
                pv.Set(type(vals)([vals[int(i)] for i in np.flatnonzero(mask)]))
        kept = local[np.unique(idx[fv])]   # extent is in the mesh's own frame
        mesh.GetExtentAttr().Set(Vt.Vec3fArray([tuple(kept.min(0)), tuple(kept.max(0))]))

    # Drop materials nothing uses any more. Several belong to the removed stand and point
    # at textures on the dataset author's machine, which would only log errors.
    used = set()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if mat:
                used.add(mat.GetPath())
    unused = [p.GetPath() for p in stage.Traverse() if p.IsA(UsdShade.Material) and p.GetPath() not in used]
    for path in unused:
        stage.RemovePrim(path)

    stage.GetRootLayer().Export(str(out))
    del stage
    tmp.unlink(missing_ok=True)
    Sdf.Layer.FindOrOpen(str(out))  # fail now, not at spawn time, if the file is unreadable


def usd_bounds(path) -> tuple:
    """(min_xyz, max_xyz) of everything renderable in a USD file, in its own units."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        stage.GetPseudoRoot()).ComputeAlignedRange()
    return tuple(box.GetMin()), tuple(box.GetMax())
