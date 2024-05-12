import os
import time

import imageio
from PIL import Image
import numpy as np
import torch
from pytorch_msssim import ms_ssim as MS_SSIM
from tqdm.auto import tqdm

from .util.metric import rgb_lpips, rgb_ssim
from .util.util import visualize_depth_numpy, visualize_tensorial_feature_numpy


def OctreeRender_trilinear_fast(
    rays,
    model,
    chunk=4096,
    n_samples=-1,
    ndc_ray=False,
    white_bg=True,
    progress=0.0,
    is_train=False,
    device="cuda",
):
    """
    Batched rendering function.
    """
    rgbs, alphas, depth_maps, z_vals = [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        
        rgb_map, depth_map, alpha_map, z_val_map = model(
            rays_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            n_samples=n_samples,
            progress=progress
        )
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        alphas.append(alpha_map)
        z_vals.append(z_val_map)
    return (
        torch.cat(rgbs),
        torch.cat(alphas),
        torch.cat(depth_maps),
        torch.cat(z_vals),
        None,
    )

def OctreeRender_trilinear_fast_disentangled(
    rays,
    model,
    chunk=4096,
    n_samples=-1,
    ndc_ray=False,
    white_bg=True,
    progress=0.0,
    is_train=False,
    device="cuda",
):
    """
    Batched rendering function.
    """
    rgbs, alphas, depth_maps, z_vals = [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        
        rgb_map, depth_map, alpha_map, z_val_map = model.forward_no_tensor_feature(
            rays_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            n_samples=n_samples,
            progress=progress,
            is_test_disentangled=True
        )
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        alphas.append(alpha_map)
        z_vals.append(z_val_map)
    return (
        torch.cat(rgbs),
        torch.cat(alphas),
        torch.cat(depth_maps),
        torch.cat(z_vals),
        None,
    )


@torch.no_grad()
def render(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    chunk=4096,
    prefix="",
    n_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    progress=1.0,
    is_val=False,
    device="cuda",
):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    PSNRs, rgb_maps, depth_maps, gt_depth_maps = [], [], [], []
    msssims, ssims, l_alex, l_vgg = [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
    idxs = list(range(0, len(test_dataset), img_eval_interval))

    start_time = time.time()
    for idx in tqdm(idxs):
        data = test_dataset[idx]
        samples, gt_rgb = data["rays"], data["rgbs"]
        depth = None

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        
        rgb_map, _, depth_map, _, _ = OctreeRender_trilinear_fast(
            rays,
            model,
            chunk=chunk,
            n_samples=n_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            progress=progress if is_val else 1.0,
            device=device,
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        if "depth" in data.keys():
            depth = data["depth"]
            gt_depth, _ = visualize_depth_numpy(depth.numpy(), near_far)

        if len(test_dataset):
            gt_rgb = gt_rgb.view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                ms_ssim = MS_SSIM(
                    rgb_map.permute(2, 0, 1).unsqueeze(0),
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    data_range=1,
                    size_average=True,
                )
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", device)
                ssims.append(ssim)
                msssims.append(ms_ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        gt_rgb_map = (gt_rgb.numpy() * 255).astype("uint8")

        if depth is not None:
            gt_depth_maps.append(gt_depth)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            # GT
            imageio.imwrite(f"{savePath}/{idx:03d}_gt.png", gt_rgb_map)
            if depth is not None:
                rgb_map = np.concatenate((gt_rgb_map, gt_depth), axis=1)
                imageio.imwrite(f"{savePath}/rgbd/{idx:03d}_gt.png", rgb_map)
            
            # Render Imgs
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            
            # Render_Imgs + Depth
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)

    end_time = time.time()
    exec_time = end_time - start_time
    
    # Wall Clock Time
    print(f"Execution time: {end_time - start_time} seconds")
    with open(f'{savePath}/{prefix}_render_exec_time.txt', 'w') as file: file.write(f'[Render] Exec time : {exec_time}\n')
    
    # GT videos
    if depth is not None:
        imageio.mimwrite(f"{savePath}/{prefix}_gt_depthvideo.mp4",
                         np.stack(gt_depth_maps), format="FFMPEG", fps=30, quality=10)

    # Rendered video            
    imageio.mimwrite(f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps), fps=30, format="FFMPEG", quality=10)
    imageio.mimwrite(f"{savePath}/{prefix}depthvideo.mp4",
        np.stack(depth_maps), format="FFMPEG", fps=30, quality=10)
    
    # Metrics
    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            msssim = np.mean(np.asarray(msssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n")
                print(f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n")
                
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}, SSIM: {ssims[i]}, MS-SSIM: {msssim}, LPIPS_a: {l_alex[i]}, LPIPS_v: {l_vgg[i]}\n")
        else:
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr} \n")
                print(f"PSNR: {psnr} \n")
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}\n")

    return PSNRs


@torch.no_grad()
def render_disentangled(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    chunk=4096,
    prefix="",
    n_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    progress=1.0,
    is_val=False,
    device="cuda",
):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    PSNRs, rgb_maps, depth_maps, gt_depth_maps = [], [], [], []
    msssims, ssims, l_alex, l_vgg = [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/disentangled", exist_ok=True)
    os.makedirs(savePath + "/disentangled/rgbd", exist_ok=True)
    os.makedirs(savePath + "/disentangled/tensor_feat", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    # model
    model.eval()
    
    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
    idxs = list(range(0, len(test_dataset), img_eval_interval))

    for idx in tqdm(idxs):
        data = test_dataset[idx]
        samples, gt_rgb = data["rays"], data["rgbs"]
        depth = None

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        
        rgb_map, _, depth_map, _, _ = OctreeRender_trilinear_fast_disentangled(
            rays,
            model,
            chunk=chunk,
            n_samples=n_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            progress=progress if is_val else 1.0,
            device=device,
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        if "depth" in data.keys():
            depth = data["depth"]
            gt_depth, _ = visualize_depth_numpy(depth.numpy(), near_far)

        if len(test_dataset):
            gt_rgb = gt_rgb.view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                ms_ssim = MS_SSIM(
                    rgb_map.permute(2, 0, 1).unsqueeze(0),
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    data_range=1,
                    size_average=True,
                )
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", device)
                ssims.append(ssim)
                msssims.append(ms_ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        gt_rgb_map = (gt_rgb.numpy() * 255).astype("uint8")

        if depth is not None:
            gt_depth_maps.append(gt_depth)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
            
        if savePath is not None:
            # GT
            # imageio.imwrite(f"{savePath}/disentangled/{idx:03d}_gt.png", gt_rgb_map)
            # if depth is not None:
            #     rgb_map = np.concatenate((gt_rgb_map, gt_depth), axis=1)
            #     imageio.imwrite(f"{savePath}/disentangled/rgbd/{idx:03d}_gt.png", rgb_map)
            
            # Render Imgs
            imageio.imwrite(f"{savePath}/disentangled/{prefix}{idx:03d}.png", rgb_map)
            
            # Render_Imgs + Depth
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/disentangled/rgbd/{prefix}{idx:03d}.png", rgb_map)

            # Tensorial feature visualization
            plane_feature, line_feature = model.get_tensorial_features()
            os.makedirs(savePath + f"/disentangled/tensor_feat/iter_{prefix}", exist_ok=True)

            for i in range(len(plane_feature)):
                os.makedirs(savePath + f"/disentangled/tensor_feat/iter_{prefix}/comp_{i}/", exist_ok=True)
                for j in range(plane_feature[i].shape[0]):
                    plane_feature_j = plane_feature[i][j]; line_feature_j = line_feature[i][j]
                    plane_feature_j_maps, _ = visualize_tensorial_feature_numpy(plane_feature_j)
                    line_feature_j_maps, _ = visualize_tensorial_feature_numpy(line_feature_j)

                    # resize
                    height = plane_feature_j.shape[-1]; width = plane_feature_j.shape[-2]
                    img_size = 256 # hardcorded
                    principle_axis = height if height >= width else width
                    height /= principle_axis; width /= principle_axis

                    # PIL
                    resized_image = Image.fromarray(plane_feature_j_maps).resize((int(img_size*height), int(img_size*width)), Image.NEAREST)
                    resized_image.save(savePath + f"/disentangled/tensor_feat/iter_{prefix}/comp_{i}/plane_channel_{j}.png")
                    # resized_image_line = Image.fromarray(line_feature_j_maps).resize((int(img_size*height), int(img_size*width)), Image.NEAREST)
                    # resized_image_line.save(savePath + f"/disentangled/tensor_feat/iter_{prefix}/comp_{i}/line_channel_{j}.png")


                    # imageio.imwrite(f"{savePath}/disentangled/tensor_feat/iter_{idx:03d}/comp_{i}/tensor_channel_{j}.png", plane_feature_j_maps)

    # GT videos
    if depth is not None:
        imageio.mimwrite(f"{savePath}/{prefix}_gt_depthvideo.mp4",
                         np.stack(gt_depth_maps), format="FFMPEG", fps=30, quality=10)

    # Rendered video            
    imageio.mimwrite(f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps), fps=30, format="FFMPEG", quality=10)
    imageio.mimwrite(f"{savePath}/{prefix}depthvideo.mp4",
        np.stack(depth_maps), format="FFMPEG", fps=30, quality=10)
    
    # Metrics
    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            msssim = np.mean(np.asarray(msssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n")
                print(f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n")
                
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}, SSIM: {ssims[i]}, MS-SSIM: {msssim}, LPIPS_a: {l_alex[i]}, LPIPS_v: {l_vgg[i]}\n")
        else:
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr} \n")
                print(f"PSNR: {psnr} \n")
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}\n")

    return PSNRs



@torch.no_grad()
def render_trajectory(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    chunk=8192,
    prefix="",
    n_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the valiation rays.
    """
    rgb_maps, depth_maps = [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    val_rays = test_dataset.get_val_rays()

    for idx in tqdm(range(len(val_rays))):
        W, H = test_dataset.img_wh
        rays = val_rays[idx]
        
        rgb_map, _, depth_map, _, _ = OctreeRender_trilinear_fast(
            rays,
            model,
            chunk=chunk,
            n_samples=n_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)

    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4", np.stack(rgb_maps), fps=30, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4", np.stack(depth_maps), fps=30, quality=8
    )

    return 0

@torch.no_grad()
def vis_planes(
    model,
    cfg,
    savePath=None,
    prefix="",
    device="cuda",
):
    os.makedirs(savePath, exist_ok=True)

    ########################
    # PLANES VISUALIZATION #
    ########################
    density_plane_feature, density_line_feature, _, _ = model.get_tensorial_features()
    os.makedirs(savePath + f"/tensor_feat/iter_{prefix}", exist_ok=True)

    for i in range(len(density_plane_feature)):
        os.makedirs(savePath + f"/tensor_feat/iter_{prefix}/comp_{i}/", exist_ok=True)
        os.makedirs(savePath + f"/tensor_feat/iter_{prefix}/comp_{i}/plane", exist_ok=True)
        os.makedirs(savePath + f"/tensor_feat/iter_{prefix}/comp_{i}/line", exist_ok=True)
        for j in range(density_plane_feature[i].shape[0]):
            plane_feature_j = density_plane_feature[i][j]; line_feature_j = density_line_feature[i][j]
            plane_feature_j_maps, _ = visualize_tensorial_feature_numpy(plane_feature_j)
            line_feature_j_maps, _ = visualize_tensorial_feature_numpy(line_feature_j)

            # resize
            height = plane_feature_j.shape[-1]; width = plane_feature_j.shape[-2]
            img_size = 256 # hardcorded
            principle_axis = height if height >= width else width
            height /= principle_axis; width /= principle_axis

            # PIL
            resized_image_plane = Image.fromarray(plane_feature_j_maps).resize((int(img_size*height), int(img_size*width)), Image.NEAREST)
            resized_image_line = Image.fromarray(line_feature_j_maps).resize((int(img_size*height), int(img_size*width)), Image.NEAREST)
            resized_image_plane.save(savePath + f"/tensor_feat/iter_{prefix}/comp_{i}/plane/plane_channel_{j}.png")
            resized_image_line.save(savePath + f"/tensor_feat/iter_{prefix}/comp_{i}/line/line_channel_{j}.png")
