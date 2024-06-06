"""
Copyright (c) 2024 Julien Posso
"""

import torch
import numpy as np
import cv2
from PIL import Image

import src.spe.se3lib as se3lib


class SPEUtils:
    """Spacecraft Pose Estimation Utils"""
    def __init__(
        self,
        camera,
        ori_mode,
        n_ori_bins_per_dim,
        ori_smooth_factor,
        pos_mode):

        self.n_ori_bins_per_dim = n_ori_bins_per_dim
        self.ori_mode = ori_mode
        self.ori_smooth_factor = ori_smooth_factor
        self.pos_mode = pos_mode
        self.camera = camera
        # Pre-compute histogram to save time during execution
        self.H_MAP, self.REDUNDANT_FLAGS = self.build_histogram(
            torch.tensor([-180, -90, -180]), torch.tensor([180, 90, 180])
        )
        self.b = self.pre_compute_ori_decode()

    def rotate_image(self, image, ori, pos, rot_max_magnitude):
        """Data augmentation: rotate image and adapt position/orientation.
        Rotation amplitude is randomly picked from [-rot_max_magnitude/2, +rot_max_magnitude/2]
        """

        image = np.array(image)

        change = (np.random.rand(1) - 0.5) * rot_max_magnitude

        # Convert euler angles (pitch, yaw roll) to rotation matrix
        r_change = se3lib.euler2SO3_left(0, 0, change[0])

        # Construct warping (perspective) matrix
        warp_matrix = self.camera.K * r_change * np.linalg.inv(self.camera.K)

        height, width = np.shape(image)[:2]

        image_warped = cv2.warpPerspective(image, warp_matrix, (width, height), cv2.WARP_INVERSE_MAP)

        # Update pose
        pos_new = np.array(np.matrix(pos) * r_change.T)[0]
        q_change = se3lib.SO32quat(r_change)
        ori_new = np.array(se3lib.quat_mult(q_change, ori))[0]

        image_pil = Image.fromarray(image_warped)
        pos_new = torch.tensor(pos_new, dtype=torch.float32)
        ori_new = torch.tensor(ori_new, dtype=torch.float32)

        return image_pil, ori_new, pos_new


    def process_output_nn(self, ori, pos, out_type="pytorch"):
        """Take the output of the neural network and compute the spacecraft Pose.
        Expect Pytorch inputs"""

        assert out_type in ("pytorch", "numpy")

        if self.ori_mode == 'regression':
            ori = torch.nn.functional.normalize(ori, p=2, dim=1)
        else:
            ori = torch.nn.functional.softmax(ori, dim=1)

        # Save ori for solver computation
        ori_loss = ori

        # Send predictions to CPU
        ori = ori.detach().cpu()
        pos = pos.detach().cpu()

        ori, _ = self.decode_ori_batch(ori)

        if out_type == "numpy":
            ori, pos = ori.numpy(), pos.numpy()

        pred_pose = {'ori': ori, 'pos': pos}

        return pred_pose, ori_loss

    def build_histogram(self, min_lim, max_lim):
        """Building the histogram of all possible orientation bins, given the number of bins per dimension and
        min/max limits on Z, Y and X axis (rotation). See https://arxiv.org/pdf/1906.09868.pdf
        The histogram is built only once to save time during execution
        """

        d = 3
        n_bins = self.n_ori_bins_per_dim ** d

        # Construct histogram structure
        bins_per_dim = torch.linspace(0.0, 1.0, self.n_ori_bins_per_dim)
        bins_all_dims = torch.cartesian_prod(bins_per_dim, bins_per_dim, bins_per_dim)
        euler_bins = bins_all_dims * (max_lim - min_lim) + min_lim
        quaternions_bins = torch.zeros((n_bins, 4), dtype=torch.float32)

        for i in range(n_bins):
            quaternions_bins[i, :] = se3lib.euler2quat(euler_bins[i, 0], euler_bins[i, 1], euler_bins[i, 2])

        # Pruning redundant bins
        # Mark redundant boundary bins
        boundary_flags = torch.logical_or(euler_bins[:, 0] == max_lim[0], euler_bins[:, 2] == max_lim[2])

        # Mark redundant bins due to the two singularities at y = -+ 90 deg
        gymbal_flags = torch.logical_and(np.abs(euler_bins[:, 1]) == max_lim[1], euler_bins[:, 0] != min_lim[0])
        redundant_flags = torch.logical_or(boundary_flags, gymbal_flags)

        return quaternions_bins, redundant_flags

    def encode_ori(self, ori):
        """Encode orientation (true orientation from the dataset)
        This code is optimized compared to Proença code: vectorization"""

        variance = (self.ori_smooth_factor / self.n_ori_bins_per_dim) ** 2 / 12

        # Compute Kernel function (equation 3 in Proença article https://arxiv.org/pdf/1907.04298.pdf)
        kernel_fct = torch.exp(- ((2 * torch.arccos(torch.minimum(torch.tensor(1.0), torch.abs(torch.sum(
            ori * self.H_MAP, dim=1)))) / np.pi) ** 2) / (2 * variance))

        kernel_fct[self.REDUNDANT_FLAGS] = 0

        ori_encoded = kernel_fct / torch.sum(kernel_fct)

        if True in torch.isnan(ori_encoded):
            raise ValueError('ori encoded is nan in encode_ori function')

        return ori_encoded

    def decode_ori(self, ori):
        """
        Decode orientation predicted by the neural network.Expects Pytorch tensor as input.
        This code is optimized compared to Proença code: vectorization and pre-compute

        Compute the average quaternion q of a set of quaternions Q,
        based on a Linear Least Squares Solution of the form: Ax = 0

        The sum of squared dot products between quaternions:
            L(q) = sum_i w_i(Q_i^T*q)^T(Q_i^T*q)^T

        achieves its maximum when its derivative wrt q is zero, i.e.:
            Aq = 0 where A = sum_i (Q_i*Q_i^T)

        Therefore, the optimal q is simply the right null space of A.

        For more solutions check:
        F. Landis Markley et al. "Averaging quaternions." Journal of Guidance, Control, and Dynamics (2007)
        https://ntrs.nasa.gov/api/citations/20070017872/downloads/20070017872.pdf

        Arguments:
            b: The pre-computed decode ori variable based on the histogram
            ori: The weights associated to each orientation bin (prediction of the neural network)
        Returns:
            q_avg: The solution
            h_inv: The uncertainty in the maximum likelihood sense
        """

        n_bins = self.b.size(0)

        # Remark: referenced as small "a" in the code but big "A" in the article
        a = torch.sum(self.b * torch.reshape(ori, (n_bins, 1, 1)), dim=0)

        if True in torch.isnan(a):
            raise ValueError("Error during orientation decoding")

        # One or another depending on the Pytorch version
        try:
            s, v = torch.linalg.eig(a)
            s, v = torch.real(s), torch.real(v)
        except AttributeError:
            s, v = torch.eig(a, eigenvectors=True)
            s = s[:, 0]

        idx = torch.argsort(s)

        q_avg = v[:, idx[-1]]

        # Due to numerical errors, we need to enforce normalization
        q_avg = q_avg / torch.linalg.norm(q_avg)

        h_inv = torch.inverse(a)

        return q_avg, h_inv

    def decode_ori_batch(self, ori):
        """Decode a batch of orientation (ori) using the pre-computed orientation decode variable (b)
        based on the histogram (see pre_compute_ori_decode)
        Expects Pytorch tensor as input.
        """

        ori = ori.cpu()
        batch_size = ori.size(0)

        ori_avg = torch.zeros((batch_size, 4), dtype=torch.float32)
        h_avg = torch.zeros((batch_size, 4, 4), dtype=torch.float32)

        for i in range(batch_size):
            ori_avg[i], h_avg[i] = self.decode_ori(ori[i])

        return ori_avg, h_avg

    def pre_compute_ori_decode(self):
        """Pre-compute Orientation decode to save time during training/inference"""
        n_bins = self.H_MAP.size(0)
        return torch.reshape(self.H_MAP, (n_bins, 4, 1)) * torch.reshape(self.H_MAP, (n_bins, 1, 4))

    def get_score(self, true_pose, pred_pose):
        """Score definition:
        https://kelvins.esa.int/satellite-pose-estimation-challenge/scoring/
        https://arxiv.org/abs/1911.02050
        Expects pytorch inputs.
        Shape of target['pos'] is (batch_size, 3). Same for others
        """

        if self.ori_mode == 'regression':
            ori_target = true_pose['ori'].detach().cpu().numpy()
        else:
            ori_target = true_pose['ori_original'].detach().cpu().numpy()
        pos_target = true_pose['pos'].detach().cpu().numpy()
        ori_pred = pred_pose['ori'].detach().cpu().numpy()
        pos_pred = pred_pose['pos'].detach().cpu().numpy()

        # 1. Position error (e_t):
        pos_error = np.linalg.norm(pos_target - pos_pred, axis=1)
        mean_pos_error = np.mean(pos_error)

        # 2. Normalized position error (e_t/)
        norm_pos_error = pos_error / np.linalg.norm(pos_target, axis=1)
        mean_norm_pos_error = np.mean(norm_pos_error)

        # 3. Orientation error (e_q)
        inter_sum = np.abs(np.sum(ori_pred * ori_target, axis=1, keepdims=True))
        # Scaling down intermediate sum to avoid nan of arccos(x) when x > 1 (seems it is what ESA does for scoring) :
        # Set it to one when the sum is just above 1 : I guess the overflow is due to numerical errors
        # Raise Value Error when it is greater than 1.01 : the overflow is due to errors in model prediction
        if True in inter_sum[inter_sum > 1.01]:
            raise ValueError("Intermediate sum issue due to error in model prediction (orientation)")
            # Remark: it seems that ESA scoring (website) is not Raising error but scaling down to zero.
            # In practice this condition is true only when there are issues with the model or solver function.
            # With the current code this error was never raised
            # inter_sum[inter_sum > 1.01] = 0
        inter_sum[inter_sum > 1] = 1

        mean_ori_error = np.mean(2 * np.arccos(inter_sum))
        mean_ori_error_deg = mean_ori_error * 180/np.pi

        # 4. ESA score
        esa_score = mean_ori_error + mean_norm_pos_error

        eval_metrics = {
            'esa_score': esa_score,
            'ori_score': mean_ori_error,
            'pos_score': mean_norm_pos_error,
            'ori_error': mean_ori_error_deg,
            'pos_error': mean_pos_error,
        }

        return eval_metrics
