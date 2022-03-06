import torch
from torch.distributions import Normal, Independent, kl


mu_pr = torch.tensor([0.2])
mu_pos = torch.tensor([0.8])

log_var_pr = torch.tensor([- 2])
log_var_pos = torch.tensor([- 0.5])

std_pr = torch.exp(0.5 * log_var_pr)
std_pos = torch.exp(0.5 * log_var_pos)

kld = log_var_pr - log_var_pos + (torch.exp(log_var_pos) + torch.square(mu_pos - mu_pr)) / (2 * torch.exp(log_var_pr)) - 0.5

dist_pr = Normal(loc=mu_pr, scale=std_pr)
dist_pos = Normal(loc=mu_pos, scale=std_pos)

print(kld)
print(kl.kl_divergence(dist_pos, dist_pr))
