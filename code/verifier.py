import argparse
import torch
import torch.nn as nn
from networks import get_network
from utils.loading import parse_spec
import numpy as np
from components import *
from skip_block import SkipBlock
import time


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)
debug = False


class Verifier:
    def __init__(self):
        self.backsub_module = BacksubstituteModule(max_alpha_counts=350)
        self.num_trials = 200
    
    def construct_model(self, net: torch.nn.Module, eps: float, true_label: int) -> bool:
        layers = [Input(eps, self.backsub_module)]
        relu_layer_count = 0
        
        for idx, layer in enumerate(net):
            if isinstance(layer, nn.Flatten):
                layers.append(Flatten(backsub_module=self.backsub_module))
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.ReLU6):
                relu_layer_count += 1
                if isinstance(net[idx-1], nn.Linear):
                    prev_layer_name = "fc"
                else:
                    prev_layer_name = "conv2d"  

                if prev_layer_name == "fc" or relu_layer_count > 1:
                    use_alphas = True
                else:
                    use_alphas = False

                if isinstance(layer, nn.ReLU6):
                    layers.append(ReLU6(backsub_module=self.backsub_module, prev_layer_name=prev_layer_name, use_alphas=use_alphas))
                else:
                    layers.append(ReLU(backsub_module=self.backsub_module, prev_layer_name=prev_layer_name, use_alphas=use_alphas))
            elif isinstance(layer, nn.Linear):
                layers.append(FullyConnected(layer, backsub_module=self.backsub_module))
            elif isinstance(layer, nn.Conv2d):
                layers.append(Conv2d(backsub_module=self.backsub_module, net_layer=layer))
            elif isinstance(layer, SkipBlock):
                layers.append(SkipBlockDP(layer, self.backsub_module))

        layers.append(Output(backsub_module=self.backsub_module, true_label=true_label))
        return nn.Sequential(*layers)

    def analyze(self, net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int) -> bool:
        model = self.construct_model(net, eps, true_label)
        
        # pass the input through to init the crossing relu
        with torch.no_grad():
            loss = model(inputs)
            if debug:
                print(f"alpha_counts: {self.backsub_module.alpha_counts}")
            self.backsub_module.reset()
            # input()
        
        num_alphas = 0  
        for name, param in model.named_parameters():
            if "alphas" in name:
                num_alphas += 1
            # print(name, param.requires_grad, param.size())
        if debug:
            print(f"Number of alphas: {num_alphas}")   

        requires_grad = any(p.requires_grad for p in model.parameters())
        if requires_grad:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        else:
            optimizer = None

        if debug:
            with torch.no_grad():
                print("Image output", net(inputs.unsqueeze(0)))

        for i in range(self.num_trials):
            if optimizer is not None:
                optimizer.zero_grad()
            diff = model(inputs)
            if debug:
                print(diff)

            if diff <= 0:
                return True
            if not requires_grad:
                break
            diff.log().backward()
            optimizer.step()
            self.backsub_module.reset()
            
        return False


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    """
    Analyzes the given network with the given input and epsilon.
    :param net: Network to analyze.
    :param inputs: Input to analyze.
    :param eps: Epsilon to analyze.
    :param true_label: True label of the input.
    :return: True if the network is verified, False otherwise.
    """
    verifier = Verifier()
    return verifier.analyze(net, inputs, eps, true_label)


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_linear",
            "fc_base",
            "fc_w",
            "fc_d",
            "fc_dw",
            "fc6_base",
            "fc6_w",
            "fc6_d",
            "fc6_dw",
            "conv_linear",
            "conv_base",
            "conv6_base",
            "conv_d",
            "skip",
            "skip_large",
            "skip6",
            "skip6_large",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    if dataset == "mnist":
        in_ch, in_dim, num_class = 1, 28, 10
    elif dataset == "cifar10":
        in_ch, in_dim, num_class = 3, 32, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    net = get_network(
        args.net,
        in_ch=in_ch,
        in_dim=in_dim,
        num_class=num_class,
        weight_path=f"models/{dataset}_{args.net}.pt",
    ).to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
