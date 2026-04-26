import torch

from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import quaternion_multiply
from utils.rigid_utils import from_homogenous, to_homogenous


def normalize_per_pixel_l1(render_image, gt_image, eps=1e-6):
    """Return a [H, W] normalized per-pixel L1 error map."""
    error = torch.abs(render_image[:3] - gt_image[:3]).mean(dim=0)
    error_min = error.min()
    error_max = error.max()
    denom = error_max - error_min
    if torch.isclose(denom, torch.zeros_like(denom), atol=eps):
        return torch.zeros_like(error)
    return (error - error_min) / (denom + eps)


def build_fastgs_metric_map(normalized_error, loss_thresh):
    return (normalized_error > loss_thresh).to(dtype=torch.int32)


def normalize_metric_scores(scores, eps=1e-6):
    score_min = scores.min()
    score_max = scores.max()
    denom = score_max - score_min
    if torch.isclose(denom, torch.zeros_like(denom), atol=eps):
        return torch.zeros_like(scores)
    return (scores - score_min) / (denom + eps)


def sampling_cameras(cameras, max_cameras=10, generator=None):
    cameras = list(cameras)
    if max_cameras <= 0 or len(cameras) <= max_cameras:
        return cameras

    permutation = torch.randperm(len(cameras), generator=generator)[:max_cameras].tolist()
    return [cameras[idx] for idx in permutation]


def compose_dynamic_gaussians(gaussians, d_xyz, d_rotation, d_scaling, is_6dof=False, scaling_modifier=1.0):
    if is_6dof and torch.is_tensor(d_xyz):
        means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(gaussians.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = gaussians.get_xyz if not torch.is_tensor(d_xyz) else gaussians.get_xyz + d_xyz

    if isinstance(d_scaling, float):
        scales = gaussians.get_scaling
    else:
        scales = gaussians.modify_scaling(d_scaling)

    if isinstance(d_rotation, float):
        rotations = gaussians.get_rotation
    else:
        rotations = quaternion_multiply(d_rotation, gaussians.get_rotation)

    return means3D, rotations, scales


def compute_vipgare_dynamic_state(
    gaussians,
    deform,
    time_input,
    dt,
    is_6dof=False,
    step_mode="train",
):
    xyz = gaussians.get_xyz
    if xyz.shape[0] == 0:
        empty_xyz = torch.empty((0, 3), device=xyz.device, dtype=xyz.dtype)
        empty_rot = torch.empty((0, 4), device=xyz.device, dtype=xyz.dtype)
        return empty_xyz, empty_rot, empty_xyz, empty_xyz, empty_rot, empty_xyz

    deform_code = deform.code_field(xyz.detach())
    if step_mode == "full_rot":
        d_xyz, d_rotation, d_scaling = deform.step_full_rot(xyz.detach(), time_input, deform_code, dt)
    elif step_mode == "default":
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, deform_code, dt)
    else:
        d_xyz, d_rotation, d_scaling = deform.step_no_rot(xyz.detach(), time_input, deform_code, dt)

    means3D, rotations, scales = compose_dynamic_gaussians(
        gaussians,
        d_xyz,
        d_rotation,
        d_scaling,
        is_6dof=is_6dof,
    )
    return d_xyz, d_rotation, d_scaling, means3D, rotations, scales


def render_dynamic_fastgs(
    viewpoint_camera,
    gaussians,
    deform,
    pipe,
    background,
    mult,
    dt,
    is_6dof=False,
    get_flag=False,
    metric_map=None,
    step_mode="train",
):
    fid = viewpoint_camera.fid
    time_input = fid.unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
    d_xyz, d_rotation, d_scaling, means3D, rotations, scales = compute_vipgare_dynamic_state(
        gaussians,
        deform,
        time_input,
        dt,
        is_6dof=is_6dof,
        step_mode=step_mode,
    )
    return render(
        viewpoint_camera,
        gaussians,
        pipe,
        background,
        d_xyz,
        d_rotation,
        d_scaling,
        is_6dof=is_6dof,
        mult=mult,
        get_flag=get_flag,
        metric_map=metric_map,
        means3D_override=means3D,
        rotations_override=rotations,
        scales_override=scales,
    )


@torch.no_grad()
def compute_dynamic_gaussian_score_fastgs(
    cameras,
    gaussians,
    deform,
    pipe,
    background,
    mult,
    dt,
    loss_thresh,
    load2gpu_on_the_fly=False,
    is_6dof=False,
    score_cameras=10,
):
    sampled_cameras = sampling_cameras(cameras, score_cameras)
    device = gaussians.get_xyz.device
    full_metric_counts = torch.zeros(gaussians.get_xyz.shape[0], device=device, dtype=torch.float32)
    full_metric_score = torch.zeros_like(full_metric_counts)

    if len(sampled_cameras) == 0 or gaussians.get_xyz.shape[0] == 0:
        return torch.zeros_like(full_metric_counts), torch.zeros_like(full_metric_counts)

    for camera in sampled_cameras:
        if load2gpu_on_the_fly:
            camera.load2device()

        render_pkg = render_dynamic_fastgs(
            camera,
            gaussians,
            deform,
            pipe,
            background,
            mult,
            dt,
            is_6dof=is_6dof,
            get_flag=False,
            step_mode="default",
        )
        render_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(camera.original_image.to(device), 0.0, 1.0)
        photometric_loss = (1.0 - 0.2) * l1_loss(render_image, gt_image) + 0.2 * (1.0 - ssim(render_image, gt_image))

        normalized_error = normalize_per_pixel_l1(render_image, gt_image)
        metric_map = build_fastgs_metric_map(normalized_error, loss_thresh)

        metric_pkg = render_dynamic_fastgs(
            camera,
            gaussians,
            deform,
            pipe,
            background,
            mult,
            dt,
            is_6dof=is_6dof,
            get_flag=True,
            metric_map=metric_map,
            step_mode="default",
        )
        counts = metric_pkg["accum_metric_counts"].to(device=device, dtype=torch.float32)
        full_metric_counts += counts
        full_metric_score += photometric_loss.detach() * counts

        if load2gpu_on_the_fly:
            camera.load2device("cpu")

    importance_score = torch.floor(full_metric_counts / max(len(sampled_cameras), 1))
    pruning_score = normalize_metric_scores(full_metric_score)
    return importance_score, pruning_score
