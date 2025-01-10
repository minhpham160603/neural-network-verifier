import torch.nn as nn
import torch
import numpy as np
import time

use_res = True


class BacksubstituteModule:
    def __init__(self, max_alpha_counts=500):
        self.abstract_lower_bound_list = []
        self.abstract_upper_bound_list = []
        self.skip_index_start = []
        self.skip_index_end = []
        self.alpha_counts = 0
        self.max_alpha_counts = max_alpha_counts

    def reset(self):
        self.abstract_lower_bound_list = []
        self.abstract_upper_bound_list = []
        self.skip_index_start = []
        self.skip_index_end = []
        self.alpha_counts = 0

    def is_max_num_alpha(self):
        return self.alpha_counts >= self.max_alpha_counts

    def increment_alpha_count(self):
        self.alpha_counts += 1

    def add_output(self, abstract_lower_bounds, abstract_upper_bounds):
        self.abstract_lower_bound_list.append(abstract_lower_bounds)
        self.abstract_upper_bound_list.append(abstract_upper_bounds)

    def back_substitute_one_step(self, in_expression, lower_bound_matrix, upper_bound_matrix, lower=True):
        if lower:
            return lower_bound_matrix @ torch.where(in_expression > 0, in_expression, torch.zeros(in_expression.shape)) \
                + upper_bound_matrix @ torch.where(in_expression < 0, in_expression, torch.zeros(in_expression.shape))
        else:
            return upper_bound_matrix @ torch.where(in_expression > 0, in_expression, torch.zeros(in_expression.shape)) \
                + lower_bound_matrix @ torch.where(in_expression < 0, in_expression, torch.zeros(in_expression.shape))

    def add_skip_index(self, start, end):
        self.skip_index_start.append(start)
        self.skip_index_end.append(end)

    def back_substitute(self, in_expression, steps=-1, lower=True, flag=False):
        if steps == -1:
            steps = len(self.abstract_lower_bound_list) - 1
        new_expression = in_expression.clone()
        prev_weight = None
        for i in range(steps):
            layer_idx = len(self.abstract_lower_bound_list) - 2 - i
            if layer_idx in self.skip_index_end:
                prev_weight = new_expression.clone()
            new_expression = self.back_substitute_one_step(new_expression, self.abstract_lower_bound_list[layer_idx], self.abstract_upper_bound_list[layer_idx], lower=lower)
            if layer_idx in self.skip_index_start and prev_weight is not None:
                new_expression[:-1, :] = new_expression[:-1, :] + prev_weight[:-1, :]
                prev_weight = None
        return new_expression

        
class Input(nn.Module):
    def __init__(self, eps, backsub_module):
        super(Input, self).__init__()
        self.eps = eps
        self.backsub_module = backsub_module

    def forward(self, inputs):
        total_nodes = int(np.prod(inputs.shape)) + 1
        abstract_lower_bounds = torch.zeros((total_nodes, total_nodes), dtype=torch.float64)
        abstract_upper_bounds = torch.zeros((total_nodes, total_nodes), dtype=torch.float64)
        abstract_lower_bounds[-1, -1] = 1.
        abstract_upper_bounds[-1, -1] = 1.
        input_flat = inputs.clone().view(-1)
        abstract_lower_bounds[-1, :-1] = torch.clamp(input_flat - self.eps, min=0., max=1.)
        abstract_upper_bounds[-1, :-1] = torch.clamp(input_flat + self.eps, min=0., max=1.)

        self.backsub_module.add_output(abstract_lower_bounds, abstract_upper_bounds)

        abstract_bounds = abstract_lower_bounds, abstract_upper_bounds
        concrete_bounds = torch.clamp(inputs - self.eps, min=0), torch.clamp(inputs + self.eps, max=1)

        return concrete_bounds, abstract_bounds


class Flatten(nn.Module):
    def __init__(self, backsub_module):
        super(Flatten, self).__init__()
        self.backsub_module = backsub_module
        self.flatten = nn.Flatten()
    
    def forward(self, inputs):
        concrete_bounds, abstract_bounds = inputs
        abstract_lower_bounds = torch.eye(abstract_bounds[0].shape[1], dtype=torch.float64)
        abstract_upper_bounds = torch.eye(abstract_bounds[1].shape[1], dtype=torch.float64)
        self.backsub_module.add_output(abstract_lower_bounds, abstract_upper_bounds)
        return (self.flatten(concrete_bounds[0]), self.flatten(concrete_bounds[1])), abstract_bounds

class FullyConnected(nn.Module):
    def __init__(self, net_layer, backsub_module):
        super(FullyConnected, self).__init__()
        self.backsub_module = backsub_module
        self.weights = net_layer.weight.detach().requires_grad_(False)
        self.biases = net_layer.bias.detach().requires_grad_(False) if net_layer.bias is not None else torch.zeros(net_layer.weight.shape[0], dtype=torch.float64).requires_grad_(True)

    def forward(self, inputs):
        _ = inputs
        bounds = torch.cat((self.weights.T, self.biases.view(1, -1)), dim=0)
        padding = torch.zeros((bounds.shape[0], 1), dtype=torch.float64)
        padding[-1, 0] = 1.
        abstract_lower_bounds = abstract_upper_bounds = torch.cat((bounds, padding), dim=1)
        self.backsub_module.add_output(abstract_lower_bounds, abstract_upper_bounds)

        backsub_abstract_lower_bounds = self.backsub_module.back_substitute(abstract_lower_bounds, lower=True)
        backsub_abstract_upper_bounds = self.backsub_module.back_substitute(abstract_upper_bounds, lower=False)
        
        out_concrete_bounds = backsub_abstract_lower_bounds[-1, :-1], backsub_abstract_upper_bounds[-1, :-1]
        out_abstract_bounds = abstract_lower_bounds, abstract_upper_bounds
            
        return out_concrete_bounds, out_abstract_bounds

class ReLU(nn.Module):
    def __init__(self, backsub_module, prev_layer_name, use_alphas=True):
        super(ReLU, self).__init__()
        self.backsub_module = backsub_module
        self.prev_layer_name = prev_layer_name
        self.use_alphas = use_alphas
        self.alphas = nn.ParameterDict()

    def forward(self, inputs):
        time_start = time.time()
        concrete_bounds, abstract_bounds = inputs
        output_size_flatten = abstract_bounds[0].shape[1]
        abstract_upper_bounds = torch.zeros((output_size_flatten, output_size_flatten), dtype=torch.float64)
        abstract_upper_bounds[-1, -1] = 1.
        abstract_lower_bounds = abstract_upper_bounds.clone()
        out_abstract_bounds = (abstract_lower_bounds, abstract_upper_bounds)
        out_concrete_bounds = torch.zeros(concrete_bounds[0].shape, dtype=torch.float64), torch.zeros(concrete_bounds[1].shape, dtype=torch.float64)
        if self.prev_layer_name == 'conv2d':
            c, h, w = concrete_bounds[0].shape
            for i in range(c):
                for j in range(h):
                    for k in range(w):
                        self.create_relu_node(concrete_bounds, (i, j, k), out_concrete_bounds, out_abstract_bounds)

        else:
            out_concrete_bounds = torch.zeros(concrete_bounds[0].shape, dtype=torch.float64), torch.zeros(concrete_bounds[1].shape, dtype=torch.float64)
            for i in range(output_size_flatten-1):
                self.create_relu_node(concrete_bounds, i, out_concrete_bounds, out_abstract_bounds)
            
        self.backsub_module.add_output(abstract_lower_bounds, abstract_upper_bounds)

        return out_concrete_bounds, out_abstract_bounds

    def create_relu_node(self, in_concrete_bound, node_idx, out_concrete_bounds, out_abstract_bounds):
        abstract_lower_bounds, abstract_upper_bounds = out_abstract_bounds
        concrete_lower_bounds, concrete_upper_bounds = out_concrete_bounds
        in_node_l = in_concrete_bound[0][node_idx]
        in_node_u = in_concrete_bound[1][node_idx]
        if self.prev_layer_name == 'conv2d':
            c, h, w = in_concrete_bound[0].shape
            flatten_idx = node_idx[0]*h*w + node_idx[1]*w + node_idx[2]
        else:
            flatten_idx = node_idx
        if in_node_u <= 0:
            return
        elif in_node_l >= 0:
            weights = torch.zeros(abstract_lower_bounds.shape[0], dtype=torch.float64)
            weights[flatten_idx] = 1.0
            concrete_lower_bounds[node_idx] = in_node_l
            concrete_upper_bounds[node_idx] = in_node_u
            abstract_lower_bounds[:, flatten_idx] = weights
            abstract_upper_bounds[:, flatten_idx] = weights.clone()
        else:
            if self.alphas.get(str(flatten_idx)) is None:
                if in_node_u < -in_node_l:
                    alpha = torch.tensor(1e-5, dtype=torch.float64)
                else:
                    alpha = torch.tensor(1., dtype=torch.float64)
                
                if self.use_alphas and not self.backsub_module.is_max_num_alpha():
                    self.alphas.update({str(flatten_idx): alpha})
                    self.backsub_module.increment_alpha_count()
            else:
                alpha = self.alphas.get(str(flatten_idx))

            alpha = alpha.clamp(min=0, max=1)

            lambda_ = in_node_u / (in_node_u - in_node_l)
            abstract_lower_bounds[flatten_idx, flatten_idx] = alpha
            abstract_upper_bounds[flatten_idx, flatten_idx] = lambda_
            abstract_upper_bounds[-1, flatten_idx] = - lambda_ * in_node_l

            concrete_lower_bounds[node_idx] = alpha * in_node_l
            concrete_upper_bounds[node_idx] = in_node_u

class ReLU6(ReLU):
    def __init__(self, backsub_module, prev_layer_name, use_alphas=True):
        super(ReLU6, self).__init__(backsub_module, prev_layer_name, use_alphas)
        self.alpha_upper = nn.ParameterDict()
        self.alpha_lower = nn.ParameterDict()

    def forward(self, inputs):
        concrete_bounds, abstract_bounds = inputs
        output_size_flatten = abstract_bounds[0].shape[1]
        abstract_upper_bounds = torch.zeros((output_size_flatten, output_size_flatten), dtype=torch.float64)
        abstract_upper_bounds[-1, -1] = 1.
        abstract_lower_bounds = abstract_upper_bounds.clone()
        out_abstract_bounds = (abstract_lower_bounds, abstract_upper_bounds)
        out_concrete_bounds = torch.zeros(concrete_bounds[0].shape, dtype=torch.float64), torch.zeros(concrete_bounds[1].shape, dtype=torch.float64)
        if self.prev_layer_name == 'conv2d':
            # print(concrete_bounds[0].shape)
            c, h, w = concrete_bounds[0].shape
            for i in range(c):
                for j in range(h):
                    for k in range(w):
                        self.create_relu6_node(concrete_bounds, (i, j, k), out_concrete_bounds, out_abstract_bounds)

        else:
            out_concrete_bounds = torch.zeros(concrete_bounds[0].shape, dtype=torch.float64), torch.zeros(concrete_bounds[1].shape, dtype=torch.float64)
            for i in range(output_size_flatten-1):
                self.create_relu6_node(concrete_bounds, i, out_concrete_bounds, out_abstract_bounds)
            
        self.backsub_module.add_output(abstract_lower_bounds, abstract_upper_bounds)
        
        return out_concrete_bounds, out_abstract_bounds
    
    def create_relu6_node(self, in_concrete_bound, node_idx, out_concrete_bounds,out_abstract_bounds):
        abstract_lower_bounds, abstract_upper_bounds = out_abstract_bounds
        concrete_lower_bounds, concrete_upper_bounds = out_concrete_bounds
        in_node_u = in_concrete_bound[1][node_idx]  
        in_node_l = in_concrete_bound[0][node_idx]  
        if self.prev_layer_name == 'conv2d':
            c, h, w = in_concrete_bound[0].shape
            flatten_idx = node_idx[0]*h*w + node_idx[1]*w + node_idx[2]
        else:
            flatten_idx = node_idx

        if in_node_u <= 0:
            return 
        else:
            if in_node_u <= 6:
                self.create_relu_node(in_concrete_bound, node_idx, out_concrete_bounds, out_abstract_bounds)
            else: 
                # l < 0 < 6 < u
                if in_node_l <= 0:
                    # upper bound
                    if self.alpha_upper.get(str(node_idx)) is None:
                        if in_node_u - 6. < 6. - in_node_l:
                            alpha_upper = torch.tensor(6. / (6. - float(in_node_l)), dtype=torch.float64)
                        else:
                            alpha_upper = torch.tensor(1e-5, dtype=torch.float64)
                        if self.use_alphas and not self.backsub_module.is_max_num_alpha():
                            self.alpha_upper.update({str(node_idx): alpha_upper})
                            self.backsub_module.increment_alpha_count()
                    else:
                        alpha_upper = self.alpha_upper.get(str(node_idx)).clamp(min=0., max=6. / (6. - float(in_node_l)))

                    abstract_upper_bounds[flatten_idx, flatten_idx] = alpha_upper
                    abstract_upper_bounds[-1, flatten_idx] = 6 * (1 - alpha_upper)
                    concrete_upper_bounds[node_idx] = 6 + alpha_upper * (in_node_u - 6)
                    # lower bound
                    if self.alpha_lower.get(str(node_idx)) is None:
                        if in_node_u < - in_node_l:
                            alpha_lower = torch.tensor(1e-5, dtype=torch.float64)
                        else:
                            alpha_lower = torch.tensor(6. / float(in_node_u), dtype=torch.float64)
                        if self.use_alphas and not self.backsub_module.is_max_num_alpha():
                            self.alpha_lower.update({str(node_idx): alpha_lower})
                            self.backsub_module.increment_alpha_count()
                    else:
                        alpha_lower = self.alpha_lower.get(str(node_idx)).clamp(min=0, max=6. / float(in_node_u))

                    abstract_lower_bounds[flatten_idx, flatten_idx] = alpha_lower
                    concrete_lower_bounds[node_idx] = alpha_lower * in_node_l
                
                elif in_node_l <= 6: 
                    # upper bound
                    if self.alpha_upper.get(str(node_idx)) is None: 
                        if in_node_u - 6. < 6. - in_node_l:
                            alpha_upper = torch.tensor(1., dtype=torch.float64)
                        else:
                            alpha_upper = torch.tensor(1e-5, dtype=torch.float64)
                        if self.use_alphas and not self.backsub_module.is_max_num_alpha():
                            self.alpha_upper.update({str(node_idx): alpha_upper})
                            self.backsub_module.increment_alpha_count()
                    else:
                        alpha_upper = self.alpha_upper.get(str(node_idx)).clamp(min=0, max=1)

                    abstract_upper_bounds[flatten_idx, flatten_idx] = alpha_upper
                    abstract_upper_bounds[-1, flatten_idx] = 6. * (1. - alpha_upper)
                    concrete_upper_bounds[node_idx] = 6. + alpha_upper * (in_node_u - 6.)
                    
                    # lower bound
                    alpha_lower = (6. - in_node_l) / (in_node_u - in_node_l)
                    abstract_lower_bounds[flatten_idx, flatten_idx] = alpha_lower
                    abstract_lower_bounds[-1, flatten_idx] = in_node_l * (1 - alpha_lower)
                    concrete_lower_bounds[node_idx] = in_node_l
                else:
                    # 6 < l < u
                    concrete_lower_bounds[node_idx] = 6.
                    concrete_upper_bounds[node_idx] = 6.
                    abstract_lower_bounds[-1, flatten_idx] = 6.
                    abstract_upper_bounds[-1, flatten_idx] = 6.

class Conv2d(nn.Module):
    def __init__(self, backsub_module, net_layer):
        super(Conv2d, self).__init__()
        self.backsub_module = backsub_module
        self.weights = net_layer.weight.detach()
        self.bias = net_layer.bias.detach() if net_layer.bias is not None else torch.zeros(net_layer.weight.shape[0], dtype=torch.float64)
        self.padding = int(net_layer.padding[0]), int(net_layer.padding[1])
        self.stride = int(net_layer.stride[0]), int(net_layer.stride[1])

    def construct_conv_matrix(self, kernel, input_shape, stride=(1, 1), padding=(0, 0)):
        # Kernel and input dimensions
        c, kernel_h, kernel_w = kernel.shape

        indices = torch.arange(np.prod(input_shape)).reshape(input_shape).to(torch.float32)

        unfold = nn.Unfold(kernel_size=(kernel_h, kernel_w), stride=stride, padding=padding)
        indices_unfold = unfold(indices).to(torch.long)

        weight_matrix = torch.zeros((np.prod(input_shape), indices_unfold.shape[1]), dtype=torch.float32)
        for i in range(weight_matrix.shape[1]):   
            weight_matrix[:, i][indices_unfold[:, i]] = kernel.flatten()
        
        return weight_matrix


    def _create_conv2d_channel(self, prev_dim, out_dim, channel_idx, kernel_weights, kernel_bias, abstract_bounds):
        k = int(kernel_weights.shape[1]), int(kernel_weights.shape[2])
        
        out_h = out_dim[1]
        out_w = out_dim[2]

        abstract_lower_bounds, abstract_upper_bounds = abstract_bounds
        abstract_bounds_channel = self.construct_conv_matrix(kernel_weights, prev_dim, stride=self.stride, padding=self.padding)
        start_idx = channel_idx * out_h * out_w
        end_idx =  channel_idx * out_h * out_w + out_w * out_h
        weights = torch.concat([abstract_bounds_channel, kernel_bias * torch.ones(1, out_w * out_h)], dim=0)
        abstract_lower_bounds[:, start_idx : end_idx] = weights
        abstract_upper_bounds[:, start_idx : end_idx] = weights

    def _create_conv2d_channel_v0(self, prev_dim, out_dim, channel_idx, kernel_weights, kernel_bias, abstract_bounds):
        prev_channel = prev_dim[0]
        k = int(kernel_weights.shape[1]), int(kernel_weights.shape[2])
        channel_nodes = []
        
        out_h = out_dim[1]
        out_w = out_dim[2]

        start_idx = channel_idx * out_h * out_w
        end_idx =  channel_idx * out_h * out_w + out_w * out_h

        abstract_lower_bounds, abstract_upper_bounds = abstract_bounds
        for i in range(out_h):
            for j in range(out_w):
                weights = torch.zeros(np.prod(prev_dim) + 1, dtype=torch.float64, requires_grad=False)
                weights[-1] = kernel_bias
                
                start_h = i * self.stride[0] - self.padding[0]
                start_w = j * self.stride[1] - self.padding[1]
                
                for c in range(prev_channel):
                    for y in range(k[0]):
                        for x in range(k[1]):
                            idy, idx = start_h + y, start_w + x
                            if 0 <= idy < prev_dim[1] and 0 <= idx < prev_dim[2]:
                                prev_node_idx = c * prev_dim[1] * prev_dim[2] + idy * prev_dim[2] + idx
                                weights[prev_node_idx] = kernel_weights[c, y, x]

                
                abstract_lower_bounds[:, start_idx + i * out_w + j] = weights
                abstract_upper_bounds[:, start_idx + i * out_w + j] = weights

    def forward(self, inputs):
        time_start = time.time()
        concrete_bounds, abstract_bounds = inputs
        prev_dim = concrete_bounds[0].shape
        out_channel = self.weights.shape[0]     
        k = int(self.weights.shape[2]), int(self.weights.shape[3])

        out_h = ((prev_dim[1] + 2 * self.padding[0] - k[0]) // self.stride[0]) + 1
        out_w = ((prev_dim[2] + 2 * self.padding[1] - k[1]) // self.stride[1]) + 1
        out_dim = out_channel, out_h, out_w
        
        abstract_upper_bounds = torch.zeros((np.prod(prev_dim) + 1, np.prod(out_dim) + 1), dtype=torch.float64)
        abstract_upper_bounds[-1, -1] = 1.
        abstract_lower_bounds = abstract_upper_bounds.clone()


        out_abstract_bounds = (abstract_lower_bounds, abstract_upper_bounds)
        self.backsub_module.add_output(abstract_lower_bounds, abstract_upper_bounds)
        
        for c in range(out_channel):
            self._create_conv2d_channel_v0(prev_dim, out_dim, c, self.weights[c], self.bias[c], out_abstract_bounds)
        lower_backsub = self.backsub_module.back_substitute(abstract_lower_bounds)
        upper_backsub = self.backsub_module.back_substitute(abstract_upper_bounds, lower=False)
        
        concrete_lower = lower_backsub[-1, :-1].reshape(out_dim)
        concrete_upper = upper_backsub[-1, :-1].reshape(out_dim)
        out_concrete_bounds = (concrete_lower, concrete_upper)

        time_end = time.time()
        # print(f"conv2d time: {time_end - time_start} layer_idx: {len(self.backsub_module.abstract_lower_bound_list)}")
        return out_concrete_bounds, out_abstract_bounds


class SkipBlockDP(nn.Module):
    def __init__(self, net_layer, backsub_module):
        super(SkipBlockDP, self).__init__()
        self.path = []
        self.input_size = net_layer.path[0].in_features
        self.backsub_module = backsub_module
        self.num_skip_blocks = len(net_layer.path)
        
        for layer in net_layer.path:
            if isinstance(layer, nn.Linear):
                self.path.append(FullyConnected(layer, backsub_module=self.backsub_module))
            elif isinstance(layer, nn.ReLU):
                self.path.append(ReLU(backsub_module=self.backsub_module, prev_layer_name="fc"))
            elif isinstance(layer, nn.ReLU6):
                self.path.append(ReLU6(backsub_module=self.backsub_module, prev_layer_name="fc"))
        self.block = nn.Sequential(*self.path)

    def forward(self, inputs):
        # self.own_backsub.reset()
        _, in_abstract_bounds = inputs
        skip_index_start = len(self.backsub_module.abstract_lower_bound_list)
        skip_index_end = skip_index_start + self.num_skip_blocks - 1

        self.backsub_module.add_skip_index(skip_index_start, skip_index_end)
        _, abstract_bounds = self.block(inputs)
        abstract_lower_bounds = self.backsub_module.back_substitute(abstract_bounds[0], steps=self.num_skip_blocks-1, lower=True) + torch.eye(self.input_size+1, dtype=torch.float64)
        abstract_lower_bounds[-1, -1] = 1.
        abstract_upper_bounds = self.backsub_module.back_substitute(abstract_bounds[1], steps=self.num_skip_blocks-1, lower=False) + torch.eye(self.input_size+1, dtype=torch.float64)
        abstract_upper_bounds[-1, -1] = 1.  

        tmp1 = []   
        tmp2 = []
        for i in range(self.num_skip_blocks-1):
            tmp1.append(self.backsub_module.abstract_lower_bound_list.pop())
            tmp2.append(self.backsub_module.abstract_upper_bound_list.pop())
        
        e_l = self.backsub_module.back_substitute(abstract_lower_bounds, lower=True)
        e_u = self.backsub_module.back_substitute(abstract_upper_bounds, lower=False)

        for i in range(self.num_skip_blocks-1):
            self.backsub_module.abstract_lower_bound_list.append(tmp1[-i-1])
            self.backsub_module.abstract_upper_bound_list.append(tmp2[-i-1])

        concrete_bounds = (e_l[-1, :-1], e_u[-1, :-1])
        return concrete_bounds, (abstract_lower_bounds, abstract_upper_bounds)


class Output(nn.Module):
    def __init__(self, backsub_module, true_label):
        super(Output, self).__init__()
        self.backsub_module = backsub_module
        self.true_label = true_label

    def forward(self, inputs):
        concrete_bounds, abstract_bounds = inputs 
        true_label_lower_bound = abstract_bounds[0][:, self.true_label].view(-1, 1)
        compare_upper_bounds = torch.concat((abstract_bounds[1][:, :self.true_label], abstract_bounds[1][:, self.true_label+1:]), dim=1)
        diff = (true_label_lower_bound.unsqueeze(0) - compare_upper_bounds).squeeze(0)
        diff = self.backsub_module.back_substitute(diff, lower=True, flag=True)[-1, :-1]
        loss = torch.where(diff < 0, -diff, torch.zeros(diff.shape)).sum()
        return loss
        