import torch


class HyperNetDynamicLinear(torch.nn.Module):
    def __init__(self, max_in_features, max_out_features, low_rank, arch_embeds, hypernet_hidden_size):
        super().__init__()
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.active_in_features = max_in_features
        self.active_out_features = max_out_features
        self.low_rank = low_rank
        self.base_Linear = torch.nn.Linear(self.max_in_features, self.max_out_features)
        #self.low_rank_weight_factor = torch.nn.Parameter(
        #    torch.Tensor(self.low_rank, self.max_in_features)
        #).data.normal_(
        #    mean=0.0, std=0.01
        #)  # small std to make low rank part ~ 0
        self.low_rank_weight_factor = torch.nn.Linear(self.max_in_features, self.low_rank, bias=False)
        self.low_rank_weight_factor.weight.data.normal_(mean=0.0, std=0.01)
        hypernet_hidden_size = hypernet_hidden_size
        hypernet_output_size = self.max_out_features * self.low_rank
        self.low_rank_hypernet = torch.nn.Sequential(
            torch.nn.Linear(len(arch_embeds), hypernet_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hypernet_hidden_size, hypernet_output_size),
        )

        self.register_buffer("active_arch_embed", torch.zeros(len(arch_embeds)))
        # self.register_parameter("low_rank_weight_factor", torch.nn.Parameter(torch.Tensor(self.low_rank, self.max_in_features)).data.normal_(mean=0.0, std=0.01))

    def set_sample_config(self, active_in_features, active_out_features, active_arch_embeds):
        self.active_in_features = active_in_features
        self.active_out_features = active_out_features
        for i in range(len(active_arch_embeds)):
            self.active_arch_embed[i] = active_arch_embeds[i]

    def forward(self, x):
        self.active_in_features = x.shape[-1]
        #weight = self.base_Linear.weight + self.low_rank_hypernet(self.active_arch_embed).view(
        #    self.max_out_features, self.low_rank
        #).mm(self.low_rank_weight_factor)
        weight = self.base_Linear.weight + self.low_rank_hypernet(self.active_arch_embed).view(self.max_out_features, self.low_rank).mm(self.low_rank_weight_factor.weight)
        weight = weight[: self.active_out_features, : self.active_in_features]
        return torch.nn.functional.linear(x, weight, bias=self.base_Linear.bias[0:  self.active_out_features])

