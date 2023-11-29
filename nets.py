# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
import logging

class PENetwork(nn.Module):
    """
    This is a simple network loosely based on the one used in ALOHA: Auxiliary Loss Optimization for Hypothesis Augmentation (https://arxiv.org/abs/1903.05700)

    Note that it uses fewer (and smaller) layers, as well as a single layer for all tag predictions, performance will suffer accordingly.
    """
    def __init__(self,use_malware=True,use_counts=True,use_tags=True,n_tags=None,feature_dimension=1024, layer_sizes = None):
        self.use_malware=use_malware
        self.use_counts=use_counts
        self.use_tags=use_tags
        self.n_tags = n_tags
        if self.use_tags and self.n_tags == None:
            raise ValueError("n_tags was None but we're trying to predict tags. Please include n_tags")
        super(PENetwork,self).__init__()
        p = 0.05
        layers = []
        if layer_sizes is None:layer_sizes=[512,512,128]
        for i,ls in enumerate(layer_sizes):
            if i == 0:
                layers.append(nn.Linear(feature_dimension,ls))
            else:
                layers.append(nn.Linear(layer_sizes[i-1],ls))
            layers.append(nn.LayerNorm(ls))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(p))
        self.model_base = nn.Sequential(*tuple(layers))
        self.malware_head = nn.Sequential(nn.Linear(layer_sizes[-1], 1),
                                          nn.Sigmoid())
        self.count_head = nn.Linear(layer_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.tag_head = nn.Sequential(nn.Linear(layer_sizes[-1],64),
                                        nn.ELU(), 
                                        nn.Linear(64,64),
                                        nn.ELU(),
                                        nn.Linear(64,n_tags),
                                        nn.Sigmoid())

    def forward(self,data):
        rv = {}
        base_result = self.model_base.forward(data)
        if self.use_malware:
            rv['malware'] = self.malware_head(base_result)
        if self.use_counts:
            rv['count'] = self.count_head(base_result)
        if self.use_tags:
            rv['tags'] = self.tag_head(base_result)
        return rv

#--------------------- FFNN ---------------------#

class FFNN(nn.Module):
    def __init__(self, feature_dimension=1024, use_malware=True,use_counts=True,use_tags=True,n_tags=None):
        super(FFNN, self).__init__()
        self.use_malware=use_malware
        self.use_counts=use_counts
        self.use_tags=use_tags
        self.n_tags = n_tags
        if self.use_tags and self.n_tags == None:
            raise ValueError("n_tags was None but we're trying to predict tags. Please include n_tags")
        self.model = nn.Sequential(
                     nn.Linear(feature_dimension, 512),
                     nn.ReLU(),
                     nn.Dropout(0.2),
                     nn.Linear(512, 256),
                     nn.ReLU(),
                     nn.Dropout(0.2),
                     nn.Linear(256, 128),
                     nn.ReLU(),
                     nn.Dropout(0.2),
                     nn.Linear(128, 64),
                     nn.ReLU(),
                     nn.Dropout(0.2),
            ).cuda()
        self.malware_head = nn.Sequential(nn.Linear(64, 1),
                                          nn.Sigmoid())
        self.count_head = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.tag_head = nn.Sequential(nn.Linear(64, 64),
                                        nn.ReLU(), 
                                        nn.Linear(64, n_tags),
                                        nn.Sigmoid())
    def forward(self, data):
        rv = {}
        base_result = self.model.forward(data)
        if self.use_malware:
            rv['malware'] = self.malware_head(base_result)
        if self.use_counts:
            rv['count'] = self.count_head(base_result)
        if self.use_tags:
            rv['tags'] = self.tag_head(base_result)
        return rv


#--------------------- ELBO ---------------------#

class LinearReparameterization(nn.Module):
    def __init__(self,
                 feature_dimension,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Linear layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            feature_dimension: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(LinearReparameterization, self).__init__()

        self._dnn_to_bnn_flag = False
        self.feature_dimension = feature_dimension
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias

        self.mu_weight = nn.Parameter(torch.Tensor(out_features, feature_dimension))
        self.rho_weight = nn.Parameter(torch.Tensor(out_features, feature_dimension))
        self.sigma_weight = torch.log(1 + torch.exp(self.rho_weight))
        
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, feature_dimension),
                             persistent=False)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, feature_dimension),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, feature_dimension),
                             persistent=False)
        
        self.weight = self.mu_weight + \
            (self.sigma_weight * self.eps_weight.data.normal_())
            
        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_features))
            self.rho_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer(
                'eps_bias',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer(
                'prior_bias_mu',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)
        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()
        # self.get_weight_particles()


    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value
        
    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl.mean()

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma)
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias,
                              self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def get_weight_particles(self, num_particles=20):
        
        eps_samples = []
        for _ in range(num_particles):
            eps_samples.append(self.eps_weight.data.normal_())
        return eps_samples
        
            
    def mi_forward(self, input, eps):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        y_scores = torch.tensor([])
        bias = None
        weight = self.mu_weight + \
            (sigma_weight * eps)
        out = F.linear(input, weight, bias)
        return out
        # out = out.view(1, -1)
        # if y_scores.size(0) == 0:
        #     y_scores = out
        # else:
        #     y_scores = torch.cat((y_scores, out), dim=0)
            
        # self.weight = weight
        # return y_scores
    
    def forward(self, input, return_kl=False):
        if self.dnn_to_bnn_flag:
            return_kl = False
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        weight = self.mu_weight + \
            (sigma_weight * self.eps_weight.data.normal_())
        self.weight = weight
        if return_kl:
            kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            bias = bias.cuda()
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # input = input.reshape(input.shape[1], input.shape[0])
        input = input.cuda()
        self.weight = self.weight.cuda()
        out = F.linear(input, self.weight, bias)
        # print(self.weight.data.sum())
        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out

class ELBO(nn.Module):
    def __init__(self, 
                 feature_dimension,
                 use_malware=True,
                 use_counts=True,
                 use_tags=True,
                 n_tags=None,
                 prior_mu = 0.0, 
                 prior_sigma = 1.0,
                 posterior_mu_init = 0.0, 
                 posterior_rho_init = -3.0):
        super(ELBO, self).__init__()
        # self.input_size = input_size
        # self.linear = nn.Linear(input_size, 1)
        # self.log_var = nn.nn.Parameter(torch.Tensor(1, input_size).fill_(1e-6))
        print("prior variance:  %d" % prior_sigma)
        self.use_malware=use_malware
        self.use_counts=use_counts
        self.use_tags=use_tags
        self.n_tags = n_tags
        self.fc = LinearReparameterization(
            feature_dimension=feature_dimension,
            out_features=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.malware_head = nn.Sigmoid()


    def forward(self, x):
        # out, kl = self.fc(x, return_kl=True)
        # return out, kl
        rv = {}
        base_result, kl = self.fc(x, return_kl=True)
        if self.use_malware:
            rv['malware'] = self.malware_head(base_result)
        # if self.use_counts:
        #     rv['count'] = base_result
        # if self.use_tags:
        #     rv['tags'] = self.tag_head(base_result)
        return rv, kl
    
    def multi_forward(self, x):
        logits, entropies = [], []
        dict_heads = []
        for particle in self.particles:
            l = particle(x)
            dict_heads.append(l)
        dict_stack = torch.stack(dict_heads)
        return dict_stack
    

class ELBO2(nn.Module):
    def __init__(self, 
                 feature_dimension,
                 use_malware=True,
                 use_counts=True,
                 use_tags=True,
                 n_tags=None,
                 prior_mu = 0.0, 
                 prior_sigma = 1.0,
                 posterior_mu_init = 0.0, 
                 posterior_rho_init = -3.0):
        super(ELBO2, self).__init__()
        # self.input_size = input_size
        # self.linear = nn.Linear(input_size, 1)
        # self.log_var = nn.nn.Parameter(torch.Tensor(1, input_size).fill_(1e-6))
        print("prior variance:  %d" % prior_sigma)
        self.use_malware=use_malware
        self.use_counts=use_counts
        self.use_tags=use_tags
        self.n_tags = n_tags
        self.fc1 = LinearReparameterization(
            feature_dimension=feature_dimension,
            out_features=256,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.fc2 = LinearReparameterization(
            feature_dimension=256,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.fc3 = LinearReparameterization(
            feature_dimension=128,
            out_features=64,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )
            
        self.malware_head = nn.Sequential(LinearReparameterization(
            feature_dimension=64,
            out_features=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        ),
        nn.Sigmoid())
        self.count_head = LinearReparameterization(
            feature_dimension=64,
            out_features=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )
        self.sigmoid = nn.Sigmoid()
        self.tag_head = nn.Sequential(LinearReparameterization(
                                        feature_dimension=64,
                                        out_features=64,
                                        prior_mean=prior_mu,
                                        prior_variance=prior_sigma,
                                        posterior_mu_init=posterior_mu_init,
                                        posterior_rho_init=posterior_rho_init),
                                        nn.Linear(64, n_tags),
                                        nn.Sigmoid())


    def forward(self, x):
        # out, kl = self.fc(x, return_kl=True)
        # return out, kl
        rv = {}
        base_result, kl = self.fc(x, return_kl=True)
        if self.use_malware:
            rv['malware'] = self.malware_head(base_result)
        # if self.use_counts:
        #     rv['count'] = base_result
        # if self.use_tags:
        #     rv['tags'] = self.tag_head(base_result)
        return rv, kl
    
    def multi_forward(self, x):
        logits, entropies = [], []
        dict_heads = []
        for particle in self.particles:
            l = particle(x)
            dict_heads.append(l)
        dict_stack = torch.stack(dict_heads)
        return dict_stack
    

#--------------------- BNN ---------------------#
ALPHA = 0.01 

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


class BayesWrap(nn.Module):
    def __init__(self, NET, num_particles):
        super().__init__()

        self.h_kernel = 0
        self.particles = []

        for i in range(num_particles):
            self.particles.append(copy.deepcopy(NET))
            # set init weights for different particle
            self.particles[i].apply(init_weights)
        #             if num_particles > 1:
        #                 setattr(opt, shared_feat_name, self.particles[-1])
        #                 self.particles.append(Model(opt))
        #         delattr(opt, shared_feat_name)

        for i, particle in enumerate(self.particles):
            self.add_module(str(i), particle)

        logging.info("num particles: %d" % len(self.particles))

    def sample_particle(self):
        return self.particles[np.random.randint(0, len(self.particles))]

    def get_particle(self, index):
        return self.particles[index]

    def forward(self, x, **kwargs):
        logits, entropies = [], []
        dict_heads = []
        #  return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        return_max = "return_max" in kwargs and kwargs["return_max"]
        for particle in self.particles:
            l = particle(x)
            dict_heads.append(l)
            #  if return_entropy:
            #  l = torch.softmax(l, 1)
            #  entropies.append((-l * torch.log(l + 1e-8)).sum(1))
        if return_max:
            avg_dict = dict_heads
        else:
            avg_dict = dict_mean(dict_heads)

        #  logits = torch.stack(logits).mean(0)
        #  if return_entropy:
        #  entropies = torch.stack(entropies).mean(0)
        #  return logits, entropies
        return avg_dict

    def update_grads(self):
        if np.random.rand() < 0.6:
            return
        all_pgs = self.particles
        if self.h_kernel <= 0:
            self.h_kernel = 0.1  # 1
        dists = []
        alpha = ALPHA  # if t < 100 else 0.0
        new_parameters = [None] * len(all_pgs)

        for i in range(len(all_pgs)):
            new_parameters[i] = {}
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is None:
                    new_parameters[i][l] = None
                else:
                    new_parameters[i][l] = p.grad.data.new(
                        p.grad.data.size()).zero_()
            for j in range(len(all_pgs)):
                # if i == j:
                #     continue
                for l, params in enumerate(
                        zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                    p, p2 = params
                    if p.grad is None or p2.grad is None:
                        continue
                    if p is p2:
                        dists.append(0)
                        new_parameters[i][
                            l] = new_parameters[i][l] + p.grad.data
                    else:
                        d = (p.data - p2.data).norm(2)
                        # if p is not p2:
                        dists.append(d.cpu().item())
                        kij = torch.exp(-(d**2) / self.h_kernel**2 / 2)
                        new_parameters[i][l] = (
                            ((new_parameters[i][l] + p2.grad.data) -
                             (d / self.h_kernel**2) * alpha) /
                            float(len(all_pgs))) * kij
        self.h_kernel = np.median(dists)
        self.h_kernel = np.sqrt(0.5 * self.h_kernel / np.log(len(all_pgs)) + 1)
        for i in range(len(all_pgs)):
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is not None:
                    p.grad.data = new_parameters[i][l]
