"""
Copyright (c) 2024 Julien Posso
"""

import torch
import numpy as np


class SPEFinn:
    """Spacecraft Pose Estimation with FINN and compare with Pytorch/Brevitas"""
    def __init__(self, pynq_remote, brevitas_model, dataloader, spe_utils, dataflow_parent_model):
        self.spe_utils = spe_utils
        self.pynq_remote = pynq_remote
        self.brevitas_model = brevitas_model
        self.brevitas_model.eval()
        self.dataloader = dataloader
        self.scale = self._get_scaling_value(dataflow_parent_model)

    @staticmethod
    def _print_pose(pose, prefix="FINN"):
        ori = pose['ori'].squeeze().tolist()
        pos = pose['pos'].squeeze().tolist()
        print(f"{prefix} POSE:\nORI = {ori}, \nPOS = {pos}\n")

    @staticmethod
    def _print_pose_score(score, prefix="FINN"):
        print(f"{prefix} POSE score:\n"
              f"ori error (deg) = {score['ori_error']:.2f}\n"
              f"pos error (m) = {score['pos_error']:.2f}\n"
              f"esa score = {score['esa_score']:.2f}\n")

    @staticmethod
    def _print_throughput(metrics):
        print("Throughput metrics:")
        for x in metrics:
            print(f"{x} = {metrics[x]:.3f}")

    @staticmethod
    def _get_scaling_value(model):
        """Automatically find the scaling factor by which we need to multiply the output of the FINN accelerator.
        The dataflow parent model is: global_in -> Transpose -> FINN_NODES -> Transpose -> Multiply -> global_out"""
        tensor_list = model.get_all_tensor_names()
        mul_nodes = [i for i in tensor_list if i[:4] == "Mul_"]
        assert len(mul_nodes) == 1, f"Should be only 1 Multiply node but found {mul_nodes}"
        scaling_value = model.get_initializer(mul_nodes[0]).item()
        return scaling_value

    def predict_finn(self, image: torch.Tensor):
        """
        Execute the end-to-end Spacecraft POSE estimation model, including inference on the remote pynq board:
        Image -> Transpose -> inference FINN model on Pynq board -> Transpose -> Multiply -> inference Neural Network
        Head -> post-precessing -> predicted POSE
        :param image: An Pytorch UINT8 NCHW image on [0-255 range]
        :return:
        """

        # Transpose image from NCHW to NHWC
        image = image.permute(0, 2, 3, 1).numpy()

        # Execute the inference on the FPGA board
        out_finn = self.pynq_remote.execute(image, exec_mode="execute", timeout=240)

        # Set back to Pytorch with NHWC layout
        out_finn = torch.tensor(out_finn).permute(0, 3, 1, 2)

        # Multiply by the scale of the output feature tensor of the brevitas model
        out_finn_rescaled = out_finn * self.scale

        # Predict the Head (AvgPooling and FC layers with Brevitas on the CPU)
        ori, pos = self.brevitas_model.head(out_finn_rescaled)

        # Post-processing: retrieve the estimated POSE based on the neural network output
        pose, _ = self.spe_utils.process_output_nn(ori, pos)

        return pose, out_finn_rescaled

    def _predict_torch(self, image: torch.Tensor):
        """Predict the POSE with Pytorch (to be compared with FINN POSE)"""
        with torch.no_grad():
            out_conv_layers = self.brevitas_model.features(image)
            ori, pos = self.brevitas_model.head(out_conv_layers)
        pose, _ = self.spe_utils.process_output_nn(ori, pos)
        return pose, out_conv_layers

    def predict_and_compare(self, image: torch.tensor, true_pose, print_results=False):
        """
        Predict and compare POSE with Pytorch/Brevitas and FINN on remote Pynq board.
        Print the similarity between the FINN and the Pytorch/Brevitas
        :param print_results:
        :param image: A Pytorch Float32 image on [0-1] scale
        :param true_pose: The true POSE from the dataset
        :return:None
        """

        uint8_img = (image * 255).type(torch.uint8)
        finn_pose, out_finn_features = self.predict_finn(uint8_img)
        torch_pose, out_torch_features = self._predict_torch(image)

        finn_score = self.spe_utils.get_score(true_pose, finn_pose)
        torch_score = self.spe_utils.get_score(true_pose, torch_pose)

        if print_results:
            self._print_pose(finn_pose, prefix="FINN")
            self._print_pose(torch_pose, prefix="Pytorch/Brevitas")

            self._print_pose_score(finn_score, prefix="FINN")
            self._print_pose_score(torch_score, prefix="Pytorch/Brevitas")

            n_zeros = torch.count_nonzero(out_finn_features) / torch.numel(out_finn_features)
            print(f'Non zeros on FINN backbone output: {n_zeros*100:.2f}%')
            n_zeros = torch.count_nonzero(out_torch_features) / torch.numel(out_torch_features)
            print(f'Non zeros on Pytorch backbone output: {n_zeros*100:.2f}%\n')

            mse = torch.mean((out_finn_features - out_torch_features) ** 2).item()
            print(f'MSE = {mse}\n')

            # print(f'Mean² out FINN features: {torch.mean(out_finn_features).item():.3f}')
            # print(f'Mean² out Torch features: {torch.mean(out_torch_features).item():.3f}')

            # Do the zeros are at the same place in the tensors out_finn_features and out_torch_features?
            zero_similarities = (out_torch_features == 0) == (out_finn_features == 0)
            zero_similarity_score = zero_similarities.sum() / torch.numel(zero_similarities)
            print(f'Similarity score on zero elements indices: {zero_similarity_score*100:.2f}%\n')

            # zero_elements_finn = out_finn_features[out_finn_features == 0]
            # zero_elements_torch = out_torch_features[out_torch_features == 0]
            # similarities = (zero_elements_torch == zero_elements_finn)
            # n_similarities = similarities.sum().item()
            # similarity_score = torch.count_nonzero(similarities) / torch.numel(similarities) * 100

            # Compare non-zero elements using torch.isclose
            # is_close_mask = torch.isclose(non_zero_elements_finn, non_zero_elements_torch, rtol=0.1, atol=0.1)
            # similarity_score = torch.count_nonzero(is_close_mask) / torch.numel(is_close_mask) * 100
            # print(f"Similarity on non zero elements: {similarity_score:.3f}%\n")

            # print(f'FINN:\n{out_finn_features.numpy()[0,11,:,:]}\n')
            # print(f'Brevitas:\n{out_torch_features.numpy()[0,11,:,:]}\n')

            similarity = torch.isclose(out_finn_features, out_torch_features, atol=0.1, rtol=0.1)
            similarity_score = torch.count_nonzero(similarity) / torch.numel(similarity) * 100
            print(f"Similarity between Pytorch and FINN predicted tensor: {similarity_score:.3f}%\n")

        return finn_score, torch_score

    def throughput_test(self):

        # Get random image
        image, _ = next(iter(self.dataloader))
        image = image['torch']

        # Transpose image from NCHW to NHWC
        image = (image * 255).type(torch.uint8).permute(0, 2, 3, 1).numpy()

        print('Throughput test....')
        # Execute the inference on the FPGA board
        execution_metrics = self.pynq_remote.execute(image, exec_mode="throughput_test")
        self._print_throughput(execution_metrics)
