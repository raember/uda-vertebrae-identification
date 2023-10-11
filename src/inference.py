from __future__ import division

import numpy as np
import torch
from matplotlib import pyplot as plt, cm
from scipy.ndimage import label

import wandb
from typing import Tuple, List

from src.measure import DetectionModelWrapper, IdentificationModelWrapper, args
from src.utility_functions import opening_files
from src.utility_functions.labels import LABELS_NO_L6, VERTEBRAE_SIZES
from src.utility_functions.writing_files import create_lml_file

torch.cuda.empty_cache()


# def load_ckpt(path: Path) -> Tuple[Module, Module, Optimizer]:
#     print("=> loading checkpoint '{}'".format(str(path)))
#     if not path.is_file():
#         raise OSError(f"{str(path)} does not exist!")
#
#     checkpoint = torch.load(path, map_location='cuda')
#     # ---------- Replace Args!!! ----------- #
#     args2 = checkpoint['args']
#     # -------------------------------------- #
#     model_g, model_head = get_models(
#         mode=args2.mode,
#         n_class=args2.n_class,
#         is_data_parallel=args2.is_data_parallel
#     )
#
#     optimizer = get_optimizer(list(model_head.parameters()) + list(model_g.parameters()), lr=args.lr, opt=args.opt,
#                               momentum=args.momentum, weight_decay=args.weight_decay)
#
#     model_g.load_state_dict(checkpoint['g_state_dict'])
#     model_head.load_state_dict(checkpoint['f1_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print("=> loaded checkpoint '{}'".format(str(path)))
#
#     for state in optimizer.state.values():
#         for k, v in state.items():
#             if isinstance(v, torch.Tensor):
#                 state[k] = v.cuda()
#
#     return model_g, model_head, optimizer
#
# # Load checkpoints and prepare them
# det_model_g, det_model_head, det_optim = load_ckpt(args.detection_pth)
# id_model_g, id_model_head, id_optim = load_ckpt(args.identification_pth)
#
# if torch.cuda.is_available():
#     det_model_g.cuda()
#     det_model_head.cuda()
#     id_model_g.cuda()
#     id_model_head.cuda()


def load_dicom(path: str, spacing: List[int]=None) -> np.ndarray:
    if spacing is None:
        spacing = [1., 1., 1.]
    volume: np.ndarray
    if path.suffix == '.dcm':
        volume, *_ = opening_files.read_volume_dcm_series(str(path), spacing=spacing, series_prefix="mediastinum")
    elif path.suffix == '.nii.gz':
        volume, *_ = opening_files.read_volume_nii_format(str(path), spacing=spacing)
    else:
        volume, *_ = opening_files.read_volume_dcm_series(str(path), spacing=spacing, series_prefix="mediastinum")
        # raise ValueError(f'DICOM file does not end in .dcm or .nii.gz: {str(path)}')
    return volume


def load_spine_model(mtype):
    if mtype == "detection":
        return DetectionModelWrapper(
            mode=mtype,
            n_class=2,
        )
    elif mtype == "identification":
        return IdentificationModelWrapper(
            mode=mtype,
            n_class=1,
        )

    else:
        raise AttributeError("Unknown model type " + mtype)


def apply_detection_model(volume, model, X_size, y_size, ignore_small_masks_detection, img_name=None):
    # E.g if X_size = 30 x 30 x 30 and y_size is 20 x 20 x 20
    #  Then cropping is ((5, 5), (5, 5), (5, 5)) pad the whole thing by cropping
    # Then pad an additional amount to make it divisible by Y_size + cropping
    # Then iterate through in y_size + cropping steps
    # Then uncrop at the end

    border = ((X_size - y_size) / 2.0).astype(int)
    border_paddings = np.array(list(zip(border, border))).astype(int)
    volume_padded = np.pad(volume, border_paddings, mode="constant")

    # pad to make it divisible to patch size
    divisible_area = volume_padded.shape - X_size
    paddings = np.mod(y_size - np.mod(divisible_area.shape, y_size), y_size)
    paddings = np.array(list(zip(np.zeros(3), paddings))).astype(int)
    volume_padded = np.pad(volume_padded, paddings, mode="constant")

    output = np.zeros(volume_padded.shape)

    print(X_size, y_size, volume.shape, output.shape)
    for x in range(0, volume_padded.shape[0] - X_size[0] + 1, y_size[0]):
        for y in range(0, volume_padded.shape[1] - X_size[1] + 1, y_size[1]):
            for z in range(0, volume_padded.shape[2] - X_size[2] + 1, y_size[2]):
                corner_a = [x, y, z]
                corner_b = corner_a + X_size
                corner_c = corner_a + border
                corner_d = corner_c + y_size
                patch = volume_padded[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
                patch = patch.reshape(1, *X_size, 1)
                result = model.predict(patch)  # patch: [1, 64, 64, 80, 1]    result: [1, 64, 64, 80, 2]
                result = np.squeeze(result, axis=0)
                decat_result = np.argmax(result, axis=3)
                cropped_decat_result = decat_result[border[0]:-border[0], border[1]:-border[1], border[2]:-border[2]]
                output[corner_c[0]:corner_d[0], corner_c[1]:corner_d[1], corner_c[2]:corner_d[2]] = cropped_decat_result
                # output[corner_c[0]:corner_d[0], corner_c[1]:corner_d[1], corner_c[2]:corner_d[2]] = decat_result
                # print(x, y, z, np.bincount(decat_result.reshape(-1).astype(int)))

    output = output[border[0]:border[0] + volume.shape[0],
             border[1]:border[1] + volume.shape[1],
             border[2]:border[2] + volume.shape[2]]

    if ignore_small_masks_detection:
        # only keep the biggest connected component
        structure = np.ones((3, 3, 3), dtype=int)
        labeled, ncomponents = label(output, structure)
        unique, counts = np.unique(labeled, return_counts=True)
        output_without_small_masks = np.zeros(labeled.shape)
        output_without_small_masks[labeled == unique[np.argsort(counts)[-2]]] = 1

        if img_name is not None:
            flatten_output = np.sum(output, axis=0)
            flatten_output[flatten_output > 1] = 1
            flatten = np.sum(output_without_small_masks, axis=0)
            flatten[flatten > 1] = 1

            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 20), dpi=300)

            axes[0, 0].imshow(volume[volume.shape[0] // 2, :, :], cmap='bone')
            axes[0, 0].imshow(output[output.shape[0] // 2, :, :], cmap=cm.winter, alpha=0.3)
            axes[0, 0].set_title("Center Slice")
            axes[0, 1].imshow(volume[volume.shape[0] // 2, :, :], cmap='bone')
            axes[0, 1].imshow(output_without_small_masks[output_without_small_masks.shape[0] // 2, :, :],
                              cmap=cm.winter,
                              alpha=0.3)
            axes[0, 1].set_title("Center Slice")

            axes[1, 0].imshow(volume[volume.shape[0] // 2, :, :], cmap='bone')
            axes[1, 0].imshow(flatten_output, cmap=cm.winter, alpha=0.3)
            axes[1, 0].set_title("All Slices")
            axes[1, 1].imshow(volume[volume.shape[0] // 2, :, :], cmap='bone')
            axes[1, 1].imshow(flatten, cmap=cm.winter, alpha=0.3)
            axes[1, 1].set_title("All Slices")

            flatten_output = np.sum(output, axis=1)
            flatten_output[flatten_output > 1] = 1
            flatten = np.sum(output_without_small_masks, axis=1)
            flatten[flatten > 1] = 1

            axes[2, 0].imshow(np.rot90(volume[:, volume.shape[0] // 2, :]), cmap='bone')
            axes[2, 0].imshow(np.rot90(flatten_output), cmap=cm.winter, alpha=0.3)
            axes[2, 0].set_title("All Slices")
            axes[2, 1].imshow(np.rot90(volume[:, volume.shape[0] // 2, :]), cmap='bone')
            axes[2, 1].imshow(np.rot90(flatten), cmap=cm.winter, alpha=0.3)
            axes[2, 1].set_title("All Slices")

            fig.tight_layout()
            fig.savefig(img_name)
            plt.close(fig)
            plt.close()
            # fig.show()

        output = output_without_small_masks

    return output


def apply_identification_model(volume, i_min, i_max, model):
    paddings = np.mod(16 - np.mod(volume.shape[1:3], 16), 16)
    paddings = np.array(list(zip(np.zeros(3), [0] + list(paddings)))).astype(int)
    volume_padded = np.pad(volume, paddings, mode="constant")
    output = np.zeros(volume_padded.shape)
    i_min = max(i_min, 4)
    i_max = min(i_max, volume_padded.shape[0] - 4)

    for i in range(i_min, i_max, 1):
        volume_slice_padded = volume_padded[i - 4:i + 4, :, :]
        volume_slice_padded = np.transpose(volume_slice_padded, (1, 2, 0))
        patch = volume_slice_padded.reshape(1, *volume_slice_padded.shape)
        result = model.predict(patch)
        result = np.squeeze(result, axis=0)
        result = np.squeeze(result, axis=-1)
        result = np.round(result)
        output[i, :, :] = result

    output = output[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
    return output


def test_scan(detection_model, detection_X_shape, detection_y_shape,
              identification_model, volume, ignore_small_masks_detection, img_name=None):
    # first stage is to put the volume through the detection model to find where vertebrae are
    print("apply detection")
    detections = apply_detection_model(volume, detection_model, detection_X_shape, detection_y_shape,
                                       ignore_small_masks_detection, img_name=img_name)
    print("finished detection")

    # get the largest island
    largest_island_np = np.transpose(np.nonzero(detections))
    i_min = np.min(largest_island_np[:, 0])
    i_max = np.max(largest_island_np[:, 0])

    # second stage is to pass slices of this to the identification network
    print("apply identification")
    identifications = apply_identification_model(volume, i_min, i_max, identification_model)
    print("finished identification")

    # crop parts of slices
    identifications_croped = identifications * detections
    print("finished multiplying")

    fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(20, 15), dpi=300)
    slice_idx = identifications_croped.shape[0] // 2 - 12
    v_min = 0
    v_max = np.max(identifications_croped)
    for i in range(5):
        for j in range(5):
            axes[i, j].imshow(volume[slice_idx, :, :], cmap='bone')
            mask = axes[i, j].imshow(identifications_croped[slice_idx, :, :], cmap='gist_ncar', alpha=0.4, vmin=v_min,
                                     vmax=v_max)
            axes[i, j].set_title(f"Sagital  {slice_idx}")
            slice_idx += 1
            fig.colorbar(mask, ax=axes[i, j])

    plt.tight_layout()
    if img_name is not None:
        fig.savefig(img_name.split(".png")[0] + "_sagital-slices.png")
        plt.close(fig)
        plt.close()

    def fix_sagital_slices(array):
        array = array.copy()
        column_medians = np.ma.median(np.ma.masked_where(array == 0, array), axis=0).filled(0)
        medians = np.repeat(column_medians[np.newaxis, :], array.shape[0], axis=0)
        non_zero_mask = array != 0
        array[non_zero_mask] = medians[non_zero_mask]
        return array

    def fix_sagital_slices_3d(array):
        array = array.copy()
        column_medians = np.ma.median(np.ma.masked_where(array == 0, array), axis=1).filled(0)
        medians = np.repeat(column_medians[:, np.newaxis, :], array.shape[1], axis=1)
        non_zero_mask = array != 0
        array[non_zero_mask] = medians[non_zero_mask]
        return array

    identifications_croped_fixed = fix_sagital_slices_3d(identifications_croped)

    fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(20, 15), dpi=300)
    slice_idx = identifications_croped.shape[0] // 2 - 12
    v_min = 0
    v_max = np.max(identifications_croped)
    for i in range(5):
        for j in range(5):
            fixed_slice = fix_sagital_slices(identifications_croped[slice_idx, :, :])
            assert np.all(fixed_slice == identifications_croped_fixed[slice_idx, :, :])

            axes[i, j].imshow(volume[slice_idx, :, :], cmap='bone')
            mask = axes[i, j].imshow(fixed_slice, cmap='gist_ncar', alpha=0.4, vmin=v_min, vmax=v_max)
            axes[i, j].set_title(f"Sagital  {slice_idx}")
            slice_idx += 1
            fig.colorbar(mask, ax=axes[i, j])

    plt.tight_layout()
    if img_name is not None:
        fig.savefig(img_name.split(".png")[0] + "_sagital-slices-fixed.png")
        plt.close(fig)
        plt.close()

    fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(20, 15), dpi=300)

    def plot_median(array, axis):
        arr_masked = np.ma.masked_where(array == 0, array)
        return np.ma.median(arr_masked, axis=axis).filled(0).T

    def plot_center_slice(array, axis):
        if axis == 0:
            return array[array.shape[0] // 2, :, :].T
        elif axis == 1:
            return array[:, array.shape[1] // 2, :].T
        elif axis == 2:
            return array[:, :, array.shape[2] // 2].T

    axes[0, 0].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_00 = axes[0, 0].imshow(detections[detections.shape[0] // 2, :, :].T, cmap='gist_ncar', alpha=0.4)
    axes[0, 0].set_title("Sagital Center Slice (detections)")

    axes[0, 1].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_01 = axes[0, 1].imshow(plot_median(detections, 0), cmap='gist_ncar', alpha=0.3)
    axes[0, 1].set_title("Sagital Median (detections)")

    axes[0, 2].imshow(plot_center_slice(volume, axis=1), cmap='bone')
    mask_02 = axes[0, 2].imshow(plot_median(detections, 1), cmap='gist_ncar', alpha=0.3)
    axes[0, 2].set_title("Coronal Median (detections)")

    axes[1, 0].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_10 = axes[1, 0].imshow(identifications[identifications.shape[0] // 2, :, :].T, cmap='gist_ncar', alpha=0.3)
    axes[1, 0].set_title("Sagital Center Slice (identifications)")

    axes[1, 1].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_11 = axes[1, 1].imshow(plot_median(identifications, 0), cmap='gist_ncar', alpha=0.3)
    axes[1, 1].set_title("Sagital Median (identifications)")

    axes[1, 2].imshow(plot_center_slice(volume, axis=1), cmap='bone')
    mask_12 = axes[1, 2].imshow(plot_median(identifications, 1), cmap='gist_ncar', alpha=0.3)
    axes[1, 2].set_title("Coronal Median (identifications)")

    axes[2, 0].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_20 = axes[2, 0].imshow(identifications_croped[identifications_croped.shape[0] // 2, :, :].T, cmap='gist_ncar',
                                alpha=0.3)
    axes[2, 0].set_title("Sagital Center Slice (identifications * detections)")

    axes[2, 1].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_21 = axes[2, 1].imshow(plot_median(identifications_croped, 0), cmap='gist_ncar', alpha=0.3)
    axes[2, 1].set_title("Sagital Median (identifications * detections)")

    axes[2, 2].imshow(plot_center_slice(volume, axis=1), cmap='bone')
    mask_22 = axes[2, 2].imshow(plot_median(identifications_croped, 1), cmap='gist_ncar', alpha=0.3)
    axes[2, 2].set_title("Coronal Median (identifications * detections)")

    axes[3, 0].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_30 = axes[3, 0].imshow(identifications_croped_fixed[identifications_croped_fixed.shape[0] // 2, :, :].T,
                                cmap='gist_ncar', alpha=0.3)
    axes[3, 0].set_title("Sagital Center Slice (fixedd)")

    axes[3, 1].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_31 = axes[3, 1].imshow(plot_median(identifications_croped_fixed, 0), cmap='gist_ncar', alpha=0.3)
    axes[3, 1].set_title("Sagital Median (fixed)")

    axes[3, 2].imshow(plot_center_slice(volume, axis=1), cmap='bone')
    mask_32 = axes[3, 2].imshow(plot_median(identifications_croped_fixed, 1), cmap='gist_ncar', alpha=0.3)
    axes[3, 2].set_title("Coronal Median (fixed)")

    fig.colorbar(mask_00, ax=axes[0, 0])
    fig.colorbar(mask_01, ax=axes[0, 1])
    fig.colorbar(mask_02, ax=axes[0, 2])
    fig.colorbar(mask_10, ax=axes[1, 0])
    fig.colorbar(mask_11, ax=axes[1, 1])
    fig.colorbar(mask_12, ax=axes[1, 2])
    fig.colorbar(mask_20, ax=axes[2, 0])
    fig.colorbar(mask_21, ax=axes[2, 1])
    fig.colorbar(mask_22, ax=axes[2, 2])
    fig.colorbar(mask_30, ax=axes[3, 0])
    fig.colorbar(mask_31, ax=axes[3, 1])
    fig.colorbar(mask_32, ax=axes[3, 2])
    fig.tight_layout()
    # fig.show()
    if img_name is not None:
        fig.savefig(img_name)
        plt.close(fig)
        plt.close()

    # aggregate the predictions
    print("start aggregating")
    identifications_croped = np.round(identifications_croped).astype(int)
    histogram = {}
    for key in range(1, len(LABELS_NO_L6)):
        histogram[key] = np.argwhere(identifications_croped == key)
    '''
    for i in range(identifications_croped.shape[0]):
        for j in range(identifications_croped.shape[1]):
            for k in range(identifications_croped.shape[2]):
                key = identifications_croped[i, j, k]
                if key != 0:
                    if key in histogram:
                        histogram[key] = histogram[key] + [[i, j, k]]
                    else:
                        histogram[key] = [[i, j, k]]
    '''
    print("finish aggregating")

    print("start averages")
    # find averages
    labels = []
    centroid_estimates = []
    for key in sorted(histogram.keys()):
        if 0 <= key < len(LABELS_NO_L6):
            arr = histogram[key]
            # print(LABELS_NO_L6[key], arr.shape[0])
            if arr.shape[0] > max(VERTEBRAE_SIZES[LABELS_NO_L6[key]] ** 3 * 0.4, 3000):
                print(LABELS_NO_L6[key], arr.shape[0])
                centroid_estimate = np.median(arr, axis=0)
                # ms = MeanShift(bin_seeding=True, min_bin_freq=300)
                # ms.fit(arr)
                # centroid_estimate = ms.cluster_centers_[0]
                centroid_estimate = np.around(centroid_estimate, decimals=2)
                labels.append(LABELS_NO_L6[key])
                centroid_estimates.append(list(centroid_estimate))
    print("finish averages")

    return labels, centroid_estimates, detections, identifications_croped


def complete_detection_picture(volume, plot_path, ignore_small_masks_detection, detection_input_size, detection_input_shift,
                               save_detections=False):
    fig, axes = plt.subplots(figsize=(20, 10), dpi=300)
    axes.set_title('DICOM', fontsize=10, pad=10)

    img_name = plot_path + f"/effect_postprocessing.png"
    detections = apply_detection_model(volume, load_spine_model("detection"), detection_input_size,
                                       detection_input_shift, ignore_small_masks_detection, img_name)

    # save detections as weak binary label
    if save_detections:
        np.save(plot_path + "/detection", detections)

    # just take the center slice since we don't know where the vertebrae are
    cut = volume.shape[0] // 2

    volume_slice = volume[cut, :, :]
    detections_slice = detections[cut, :, :]

    masked_data = np.ma.masked_where(detections_slice == 0, detections_slice)

    axes.imshow(volume_slice.T, cmap='gray')
    axes.imshow(masked_data.T, cmap=cm.autumn, alpha=0.4)
    fig.tight_layout()
    # fig.savefig(plot_path + f'/detection-complete.png')
    plt.close(fig)
    plt.close()
    return detections


def complete_identification_picture(scan_path, plot_path, ignore_small_masks_detection,
                                    detection_input_size, detection_input_shift, volume_format, label_format,
                                    save_predictions, spacing=(1.0, 1.0, 1.0), weights=np.array([0.1, 0.9])):
    if volume_format == '.nii.gz' and label_format == ".lml":
        volume, *_ = opening_files.read_volume_nii_format(scan_path, spacing=spacing)
    elif volume_format == '.dcm' and label_format == ".nii.gz":
        volume, *_ = opening_files.read_volume_dcm_series(scan_path, spacing=spacing)
    detection_model = load_spine_model("detection")
    identification_model = load_spine_model("identification")

    fig, axes = plt.subplots(figsize=(8, 8), dpi=300)

    # without label -> just take the sagital center
    cut = round(volume.shape[0] / 2)

    axes.set_title('identification', fontsize=10, pad=10)

    pred_labels, pred_centroid_estimates, pred_detections, pred_identifications = test_scan(
        detection_model=detection_model,
        detection_X_shape=detection_input_size,
        detection_y_shape=detection_input_shift,
        identification_model=identification_model,
        volume=volume,
        ignore_small_masks_detection=ignore_small_masks_detection,
        img_name=plot_path + '/centroids_debug.png'
    )

    volume_slice = volume[cut, :, :]
    # detections_slice = pred_detections[cut, :, :]
    identifications_slice = pred_identifications[cut, :, :]
    # identifications_slice = np.max(pred_identifications, axis=0)

    # masked_data = np.ma.masked_where(identifications_slice == 0, identifications_slice)
    # masked_data = np.ma.masked_where(detections_slice == 0, detections_slice)

    axes.imshow(volume_slice.T, cmap='gray', origin='lower')
    # axes[col].imshow(masked_data.T, vmin=1, vmax=27, cmap=cm.jet, alpha=0.4, origin='lower')

    slices = {}
    for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
        u, v = pred_centroid_idx[1:3]
        axes.annotate(pred_label, (u, v), color="red", size=6)
        axes.scatter(u, v, color="red", s=8)
        if pred_label in ['L3', 'T10', 'T12']:
            slices[pred_label] = tuple(pred_centroid_idx), volume[:, :, int(v)].T

    if save_predictions:
        create_lml_file(plot_path + "/prediction.lml", pred_labels, pred_centroid_estimates)

    pred_centroid_estimates = np.array(pred_centroid_estimates).reshape((-1, 3))
    axes.plot(pred_centroid_estimates[:, 1], pred_centroid_estimates[:, 2], color="red")

    fig.tight_layout()
    # plt.show()
    fig.savefig(plot_path + '/centroids.png')
    plt.close(fig)
    plt.close()
    return pred_identifications, slices


if __name__ == '__main__':
    # volume = load_dicom(args.dicom)
    # detections = complete_detection_picture(
    #     volume,
    #     plot_path=str(args.plot_path),
    #     detection_input_size=np.array(args.detection_input_size),
    #     detection_input_shift=np.array(args.detection_input_shift),
    #     ignore_small_masks_detection=args.ignore_small_masks_detection,
    #     save_detections=args.save_detections,
    # )
    identifications, slices = complete_identification_picture(
        str(args.dicom),
        plot_path=str(args.plot_path),
        detection_input_size=np.array(args.detection_input_size),
        detection_input_shift=np.array(args.detection_input_shift),
        spacing=tuple(args.spacing),
        volume_format=str(args.volume_format),
        label_format=str(args.label_format),
        ignore_small_masks_detection=args.ignore_small_masks_detection,
        save_predictions=args.save_predictions,
    )
    fig, axs = plt.subplots(len(slices))
    for (vert_id, ((x, y, z), img)), ax in zip(slices.items(), axs):
        ax.set_title(vert_id)
        ax.imshow(img)
        ax.scatter([x], [y], marker='+', c='r')
    fig.tight_layout()
    fig.show()
