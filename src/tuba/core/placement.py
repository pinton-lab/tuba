"""Transducer placement on a skull surface.

Given a target point (world-mm), an outer-bone surface point cloud
(world-mm), and an optional cavity centroid for tilt geometry, finds:

* a dorsal apex on the bone surface, within a lateral-search cylinder
  around the target's (x, y) projection;
* an apex lift along +z that satisfies either ``scalp_clearance_mm``
  (water column above bone) or ``apex_to_target_mm`` (geometric focus
  distance, Pythagorean against lateral offset);
* an optional sagittal-plane tilt about the cavity centroid.

The returned dict mirrors the legacy mouse/macaque placement helpers
for drop-in compatibility with downstream solvers; the ``_lps`` suffix
is a historical artefact of the human (Halle) pipeline and is retained
for back-compat, but the contents are in RAS world-mm.
"""
import numpy as np


def _rotate_about_x_at_pivot(p, pivot, theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0,   c,  -s],
                  [0.0,   s,   c]])
    return pivot + R @ (np.asarray(p, dtype=np.float64) - pivot)


def place_on_skull(target_world_mm, surface_world_mm,
                   cavity_centroid_world_mm,
                   scalp_clearance_mm=None,
                   apex_to_target_mm=None,
                   beam_tilt_deg_yz=0.0,
                   lateral_search_mm=3.0,
                   target_name=None,
                   frame_name='aligned_native_RAS',
                   verbose=True):
    """Compute placement (apex, beam direction, scalp clearance).

    Parameters
    ----------
    target_world_mm : (3,) array-like
        Target point in world-mm (RAS, bone-centroid origin).
    surface_world_mm : (N, 3) array
        Outer-bone surface point cloud in the same world frame
        (e.g. from :func:`tuba.core.surface.outer_bone_surface_world_mm`).
    cavity_centroid_world_mm : (3,) array-like
        Pivot for the sagittal-plane tilt; pass [0, 0, 0] if no cavity
        is available (tilt then rotates about origin).
    scalp_clearance_mm : float, optional
        Water-column distance above the outer-bone surface.
        Ignored if ``apex_to_target_mm`` is set.
    apex_to_target_mm : float, optional
        If set, lift the apex along +z so its Euclidean distance to the
        target equals this value (geometric-focus bowl with focal
        point at the target).
    beam_tilt_deg_yz : float
        Sagittal-plane (y-z) tilt about the cavity centroid; positive =
        apex rotated rostrally.
    lateral_search_mm : float
        Cylinder radius around (target_x, target_y) for apex candidate
        selection.

    Returns
    -------
    placement : dict with keys
        target_name, target_lps, xdc_center_lps, focus_lps,
        beam_dir_3d, scalp_target_dist_mm, scalp_clearance_mm,
        apex_on_bone_voxel_mm, target_world_mm, beam_tilt_deg_yz, frame.
    """
    surf = np.asarray(surface_world_mm, dtype=np.float64)
    target_world = np.asarray(target_world_mm, dtype=np.float64).copy()
    if verbose:
        print(f'  target world-mm = {target_world.round(3)}')
        print(f'    {len(surf):,} outer-bone surface voxels')

    tx, ty, tz = target_world
    dorsal = surf[surf[:, 2] > tz]
    if len(dorsal) == 0:
        raise RuntimeError('No dorsal outer-bone surface voxels above target')
    radial = np.hypot(dorsal[:, 0] - tx, dorsal[:, 1] - ty)
    cyl = dorsal[radial < lateral_search_mm]
    if len(cyl) == 0:
        if verbose:
            print(f'    no surface within {lateral_search_mm} mm, falling '
                  f'back to all dorsal points')
        cyl = dorsal
    apex_world = cyl[np.argmax(cyl[:, 2])].copy()
    if verbose:
        print(f'    apex on bone world-mm = {apex_world.round(3)}')

    apex_lifted = apex_world.copy()
    if apex_to_target_mm is not None:
        lat2 = ((target_world[0] - apex_world[0])**2
                + (target_world[1] - apex_world[1])**2)
        if apex_to_target_mm**2 < lat2:
            raise ValueError(
                f'Lateral offset {np.sqrt(lat2):.2f} mm > apex->target '
                f'{apex_to_target_mm} mm; pick a deeper target.')
        apex_z = target_world[2] + np.sqrt(apex_to_target_mm**2 - lat2)
        L = apex_z - apex_world[2]
        if L < 0:
            raise ValueError(
                f'Required apex lift is negative ({L:.2f} mm); '
                f'increase apex_to_target_mm.')
        apex_lifted[2] = apex_z
        scalp_clearance_mm = float(L)
    else:
        if scalp_clearance_mm is None:
            scalp_clearance_mm = 3.0
        apex_lifted[2] += scalp_clearance_mm

    beam = target_world - apex_lifted
    dist = float(np.linalg.norm(beam))
    beam_unit = beam / dist
    if verbose:
        print(f'    apex lifted world-mm = {apex_lifted.round(3)}  '
              f'(lift {scalp_clearance_mm:.2f} mm)')
        print(f'    beam unit            = {beam_unit.round(4)}')
        print(f'    apex -> target dist  = {dist:.2f} mm')

    if abs(beam_tilt_deg_yz) > 1e-6:
        pivot = np.asarray(cavity_centroid_world_mm, dtype=np.float64)
        theta = np.deg2rad(+beam_tilt_deg_yz)
        apex_lifted = _rotate_about_x_at_pivot(apex_lifted, pivot, theta)
        target_world = _rotate_about_x_at_pivot(target_world, pivot, theta)
        apex_world = _rotate_about_x_at_pivot(apex_world, pivot, theta)
        beam = target_world - apex_lifted
        dist = float(np.linalg.norm(beam))
        beam_unit = beam / dist
        if verbose:
            tilt_vs_vertical = float(np.degrees(
                np.arccos(np.clip(beam_unit[2] * -1.0
                                   if beam_unit[2] < 0 else beam_unit[2],
                                   -1.0, 1.0))))
            print(f'  applied {beam_tilt_deg_yz:+.1f} deg y-z tilt about '
                  f'cavity centre {pivot.round(2)}')
            print(f'    new apex world-mm  = {apex_lifted.round(3)}')
            print(f'    new target         = {target_world.round(3)}')
            print(f'    new beam unit      = {beam_unit.round(4)}')
            print(f'    beam tilt vs |z|   = {tilt_vs_vertical:.2f} deg')

    return {
        'target_name': target_name,
        'target_lps': tuple(float(x) for x in target_world),
        'xdc_center_lps': tuple(float(x) for x in apex_lifted),
        'focus_lps': tuple(float(x) for x in target_world),
        'beam_dir_3d': tuple(float(x) for x in beam_unit),
        'scalp_target_dist_mm': dist,
        'scalp_clearance_mm': float(scalp_clearance_mm),
        'apex_on_bone_voxel_mm': tuple(float(x) for x in apex_world),
        'target_world_mm': tuple(float(x) for x in target_world),
        'beam_tilt_deg_yz': float(beam_tilt_deg_yz),
        'frame': frame_name,
    }


def place_perpendicular_to_surface(target_world_mm,
                                   surface_world_mm,
                                   surface_normals_world_mm,
                                   eeg_seed_world_mm=None,
                                   eeg_search_radius_mm=10.0,
                                   focal_length_mm=30.0,
                                   bowl_radius_mm=15.0,
                                   aim_focus_at_target=True,
                                   use_patch_normal=True,
                                   rim_safety_mm=2.0,
                                   target_name=None,
                                   frame_name='aligned_native_RAS',
                                   verbose=True):
    """Perpendicular-to-skull placement using surface normals.

    For each candidate outer-skull vertex (optionally restricted to a
    radius around an EEG seed point), compute the perpendicular
    distance from the inward-normal line through the vertex to the
    target; pick the vertex with the smallest perpendicular residual,
    requiring the target to be in front of the transducer.

    Bowl apex sits ``h = F - sqrt(F^2 - R^2)`` mm OUTSIDE the bone in
    the -beam direction so the rim is on the bone and the bowl is
    entirely outside (the angular-spectrum solver convention has
    ``z = 0`` at the apex). This is the human-pipeline placement
    geometry; mouse/macaque use the simpler dorsal-cylinder search in
    :func:`place_on_skull`.

    Parameters
    ----------
    target_world_mm : (3,) array-like
        Target point in world-mm (the species' canonical frame).
    surface_world_mm : (N, 3) array
        Outer-bone surface vertices.
    surface_normals_world_mm : (N, 3) array
        Outward unit normals per vertex.
    eeg_seed_world_mm : (3,) array-like, optional
        EEG site (warped from the atlas EEG dict into the species
        frame). Restricts candidate vertices to a sphere of radius
        ``eeg_search_radius_mm`` around the seed. If None, all
        outer-skull vertices are searched.
    eeg_search_radius_mm : float
        Sphere radius for EEG-constrained search. Widens to 2x if
        fewer than 50 candidates are found.
    focal_length_mm, bowl_radius_mm : float
        Bowl geometry. ``focal_length_mm`` is the focal length F;
        ``bowl_radius_mm`` is the aperture radius R.

    Returns
    -------
    placement : dict (keys mirror legacy halle_placement.place_on_skull)
    """
    from scipy.spatial import cKDTree
    target_lps = np.asarray(target_world_mm, dtype=np.float64).copy()
    surf = np.asarray(surface_world_mm, dtype=np.float64)
    normals = np.asarray(surface_normals_world_mm, dtype=np.float64)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    if verbose:
        print(f'  target world-mm = {target_lps.round(3)}')
        print(f'    {len(surf):,} outer-bone surface verts')

    eeg_seed = None
    if eeg_seed_world_mm is not None:
        eeg_seed = np.asarray(eeg_seed_world_mm, dtype=np.float64)
        tree = cKDTree(surf)
        cand_idx = np.array(
            tree.query_ball_point(eeg_seed, eeg_search_radius_mm),
            dtype=int)
        if len(cand_idx) < 50:
            if verbose:
                print(f'    EEG search radius widened from '
                      f'{eeg_search_radius_mm} to {eeg_search_radius_mm*2} '
                      f'mm (only {len(cand_idx)} verts within initial radius)')
            cand_idx = np.array(
                tree.query_ball_point(eeg_seed, eeg_search_radius_mm * 2),
                dtype=int)
        if verbose:
            print(f'    {len(cand_idx)} candidate verts within '
                  f'EEG search radius around {eeg_seed.round(2)}')
    else:
        cand_idx = np.arange(len(surf))

    cand_pts = surf[cand_idx]
    cand_normals = normals[cand_idx]

    diff = cand_pts - target_lps[None]
    inward = -cand_normals
    parallel = np.einsum('ij,ij->i', diff, inward)
    perp_vec = diff - parallel[:, None] * inward
    perp_dist = np.linalg.norm(perp_vec, axis=1)
    in_front = parallel < 0
    score = np.where(in_front, perp_dist, np.inf)
    best = int(np.argmin(score))
    bone_anchor = cand_pts[best]
    outward_normal = cand_normals[best].copy()

    # ------------------------------------------------------------------
    # Mean-normal over the contact patch: for wide bowls (R > a few mm)
    # the local skull curves noticeably across the bowl footprint.
    # Using a single vertex normal makes parts of the rim sink into bone
    # at high-curvature sites.  Average the outward normals of all bone
    # vertices that fall within radius ``bowl_radius_mm`` of the anchor
    # in the plane perpendicular to the anchor's normal (i.e., inside
    # the bowl footprint).  Iterate once on the updated normal so the
    # patch membership stabilises.
    # ------------------------------------------------------------------
    if use_patch_normal:
        n_local = outward_normal.copy()
        for _ in range(2):
            dvec = surf - bone_anchor
            axial = dvec @ n_local
            lat_vec = dvec - axial[:, None] * n_local[None]
            lat_dist = np.linalg.norm(lat_vec, axis=1)
            patch = ((lat_dist < bowl_radius_mm)
                     & (np.abs(axial) < bowl_radius_mm))
            if patch.sum() >= 5:
                patch_n = normals[patch]
                n_local = patch_n.mean(axis=0)
                n_local = n_local / (np.linalg.norm(n_local) + 1e-12)
        outward_normal = n_local
    beam_3d = -outward_normal

    h = float(focal_length_mm
              - np.sqrt(focal_length_mm**2 - bowl_radius_mm**2))

    # ------------------------------------------------------------------
    # Placement strategy.
    #
    # Geometry recap (beam = inward unit vector from outside of head):
    #     apex = scalp_contact - apex_to_scalp * beam        (in air/water)
    #     rim  = apex + h * beam                              (h mm below apex
    #                                                          along beam,
    #                                                          h = F - sqrt(F^2 - R^2))
    #     focus= apex + F * beam                              (geometric focus)
    #
    # Two constraints:
    #   (i)  rim must sit above the outer bone with at least
    #        ``rim_safety_mm`` margin --> apex_to_scalp >= h + rim_safety_mm
    #   (ii) the focus should land on the target when possible -->
    #        F - apex_to_scalp == target_depth_along_beam, i.e.
    #        apex_to_scalp = F - target_depth_along_beam
    #
    # When the target is shallow enough that constraint (ii) violates (i)
    # (F - target_depth < h + safety) the focus undershoots the target;
    # we set apex_to_scalp = h + rim_safety_mm and report
    # focus != target (the F=R bowl geometry physically can't reach a
    # shallow target without burying the rim).  When the target is deeper
    # than F (constraint (ii) negative), apex sits h + safety above scalp
    # and the focus is at apex + F*beam, well short of the target -- this
    # is the "transducer is wrong for this target" case and again gets
    # logged explicitly.
    #
    # ``aim_focus_at_target=False`` (legacy): apex sits exactly h mm
    # outside the bone (rim ON bone, no water column) and focus =
    # apex + F*beam regardless of target.  Kept for back-compat.
    # ------------------------------------------------------------------
    target_depth_along_beam = float(np.dot(target_lps - bone_anchor, beam_3d))
    if aim_focus_at_target:
        apex_to_scalp_mm_calc = max(h + float(rim_safety_mm),
                                     focal_length_mm - target_depth_along_beam)
        scalp_contact = bone_anchor
        apex_world = scalp_contact - apex_to_scalp_mm_calc * beam_3d
        focus_world = apex_world + focal_length_mm * beam_3d
    else:
        scalp_contact = bone_anchor
        apex_world = scalp_contact - h * beam_3d
        focus_world = apex_world + focal_length_mm * beam_3d

    # Distance between geometric focus and target along the beam line
    # (positive = focus beyond target along beam, negative = focus short
    # of target).  Use the beam-projected distance so the placement
    # quality check is independent of any small perp residual.
    focus_to_target_along_beam_mm = float(np.dot(target_lps - focus_world,
                                                    beam_3d))
    focus_to_target_mm = float(np.linalg.norm(target_lps - focus_world))

    perp_residual_mm = float(perp_dist[best])
    scalp_to_target_mm = float(np.linalg.norm(target_lps - scalp_contact))
    apex_to_target_mm = float(np.linalg.norm(target_lps - apex_world))
    apex_to_scalp_mm = float(np.linalg.norm(scalp_contact - apex_world))

    if verbose:
        print(f'    scalp contact world-mm = {scalp_contact.round(3)}')
        print(f'    outward normal         = {outward_normal.round(4)}')
        print(f'    beam (inward)          = {beam_3d.round(4)}')
        print(f'    bowl depth h           = {h:.3f} mm')
        print(f'    apex world-mm          = {apex_world.round(3)}')
        print(f'    focus world-mm         = {focus_world.round(3)}')
        print(f'    perp residual          = {perp_residual_mm:.3f} mm')
        print(f'    apex -> target dist    = {apex_to_target_mm:.2f} mm')

    # 3-D orthonormal beam basis (matches halle_placement.place_on_skull)
    n3 = beam_3d / (np.linalg.norm(beam_3d) + 1e-12)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(n3.dot(seed)) > 0.95:
        seed = np.array([0.0, 1.0, 0.0])
    t3 = seed - seed.dot(n3) * n3
    t3 = t3 / (np.linalg.norm(t3) + 1e-12)
    b3 = np.cross(n3, t3)
    b3 = b3 / (np.linalg.norm(b3) + 1e-12)
    # Legacy 2-D L-P projection (back-compat with consumers that take 2-D
    # normal/tangent).
    n_lp = np.array([beam_3d[0], beam_3d[1]])
    n_lp = n_lp / (np.linalg.norm(n_lp) + 1e-12)
    t_lp = np.array([-n_lp[1], n_lp[0]])

    return {
        'target_name': target_name,
        'target_lps': tuple(float(v) for v in target_lps),
        'xdc_center_lps': tuple(float(v) for v in apex_world),
        'scalp_contact_lps': tuple(float(v) for v in scalp_contact),
        'focus_lps': tuple(float(v) for v in focus_world),
        'beam_dir_3d': tuple(float(v) for v in beam_3d),
        'normal_3d': tuple(float(v) for v in n3),
        'tangent_3d': tuple(float(v) for v in t3),
        'bitangent_3d': tuple(float(v) for v in b3),
        'normal': (float(n_lp[0]), float(n_lp[1])),
        'tangent': (float(t_lp[0]), float(t_lp[1])),
        'z_mm': float(apex_world[2]),
        'scalp_target_dist_mm': scalp_to_target_mm,
        'apex_target_dist_mm': apex_to_target_mm,
        'perp_residual_mm': perp_residual_mm,
        'bowl_depth_mm': h,
        'focal_length_mm': float(focal_length_mm),
        'bowl_radius_mm': float(bowl_radius_mm),
        'eeg_seed_lps': (tuple(float(v) for v in eeg_seed)
                         if eeg_seed is not None else None),
        'eeg_to_placement_mm': (float(np.linalg.norm(eeg_seed - scalp_contact))
                                if eeg_seed is not None else None),
        'apex_to_scalp_mm': apex_to_scalp_mm,
        'aim_focus_at_target': bool(aim_focus_at_target),
        'patch_normal_used': bool(use_patch_normal),
        'rim_safety_mm': float(rim_safety_mm),
        'focus_to_target_mm': focus_to_target_mm,
        'focus_to_target_along_beam_mm': focus_to_target_along_beam_mm,
        'target_depth_along_beam_mm': target_depth_along_beam,
        'frame': frame_name,
    }
