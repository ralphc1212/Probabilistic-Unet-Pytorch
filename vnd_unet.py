#This code is based on: https://github.com/SimonKohl/probabilistic_unet

from unet_blocks import *
from unet import Unet
from utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 3 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_var = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_var = torch.squeeze(mu_log_var, dim=2)
        mu_log_var = torch.squeeze(mu_log_var, dim=2)

        mu, log_var, p_vnd = mu_log_var.chunk(chunks = 3, dim = 1)

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        # dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return mu, log_var, p_vnd

class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)

TAU = 1.
PI = 0.8
RSV_DIM = 2
EPS = 1e-8

class VNDUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=4, beta=10.0):
        super(VNDUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers, apply_last_layer=False, padding=True).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers,).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True).to(device)

    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_params = self.posterior.forward(patch, segm)

        self.prior_params = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch, False)

    @staticmethod
    def clip_beta(tensor, to=5.):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    def sample(self, testing=False, fix_len=None):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            # z_prior = self.prior_latent_space.rsample()
            # self.z_prior_sample = z_prior
            pass
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            #z_prior = self.prior_latent_space.base_dist.loc 
            mu_pr, log_var_pr, p_vnd_pr = self.prior_params
            # mu_pr, log_var_pr, p_vnd_pr = self.posterior_params

            std = torch.exp(0.5 * log_var_pr)
            eps = torch.randn_like(std)

            if fix_len:
                z = eps * std + mu_pr
                z_prior = torch.cat([z[:, :int(RSV_DIM + fix_len)], torch.zeros_like(z[:, int(RSV_DIM + fix_len):])], dim = -1)
            else:
                beta = torch.sigmoid(self.clip_beta(p_vnd_pr[:,RSV_DIM:]))
                ONES = torch.ones_like(beta[:,0:1])
                qv = torch.cat([ONES, torch.cumprod(beta, dim=1)], dim = -1) * torch.cat([1 - beta, ONES], dim = -1)
                s_vnd = F.gumbel_softmax(qv, tau=TAU, hard=False)

                cumsum = torch.cumsum(s_vnd, dim=1)
                dif = cumsum - s_vnd
                mask0 = dif[:, 1:]
                mask1 = 1. - mask0
                s_vnd = torch.cat([torch.ones_like(p_vnd_pr[:,:RSV_DIM]), mask1], dim = -1)

                z_prior = (eps * std + mu_pr) * s_vnd

        return self.fcomb.forward(self.unet_features, z_prior)


    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None, hard=False):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_params[0]
        else:
            if calculate_posterior:
                # z_posterior = self.posterior_latent_space.rsample()
                mu, log_var, p_vnd = self.posterior_params

                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)

                beta = torch.sigmoid(self.clip_beta(p_vnd[:,RSV_DIM:]))
                ONES = torch.ones_like(beta[:,0:1])
                qv = torch.cat([ONES, torch.cumprod(beta, dim=1)], dim = -1) * torch.cat([1 - beta, ONES], dim = -1)
                s_vnd = F.gumbel_softmax(qv, tau=TAU, hard=hard)

                cumsum = torch.cumsum(s_vnd, dim=1)
                dif = cumsum - s_vnd
                mask0 = dif[:, 1:]
                mask1 = 1. - mask0
                s_vnd = torch.cat([torch.ones_like(p_vnd[:,:RSV_DIM]), mask1], dim = -1)

                z_posterior = (eps * std + mu) * s_vnd

        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            # kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
            mu_pr, log_var_pr, p_vnd_pr = self.prior_params
            mu_pos, log_var_pos, p_vnd_pos = self.posterior_params
            kld_gaussian = (log_var_pr - log_var_pos) / 2 + (torch.exp(log_var_pos) + torch.square(mu_pos - mu_pr)) / (2 * torch.exp(log_var_pr)) - 0.5

            beta = torch.sigmoid(self.clip_beta(p_vnd_pos[:, RSV_DIM:]))
            ONES = torch.ones_like(beta[:,0:1])
            qv = torch.cat([ONES, torch.cumprod(beta, dim=1)], dim = -1) * torch.cat([1 - beta, ONES], dim = -1)

            ZEROS = torch.zeros_like(beta[:, 0:1])
            cum_sum = torch.cat([ZEROS, torch.cumsum(qv[:, 1:], dim = 1)], dim = -1)[:, :-1]
            coef1 = torch.sum(qv, dim=1, keepdim=True) - cum_sum
            coef1 = torch.cat([torch.ones_like(p_vnd_pos[:,:RSV_DIM]), coef1], dim = -1)
            kld_weighted_gaussian = torch.diagonal(kld_gaussian.mm(coef1.t()), 0).mean()

            beta_pr = torch.sigmoid(self.clip_beta(p_vnd_pr[:, RSV_DIM:]))
            ONES = torch.ones_like(beta_pr[:,0:1])
            pv = torch.cat([ONES, torch.cumprod(beta_pr, dim=1)], dim = -1) * torch.cat([1 - beta_pr, ONES], dim = -1)

            log_frac = torch.log(qv / pv + EPS)
            kld_vnd = torch.diagonal(qv.mm(log_frac.t()), 0).mean()
            kl_div = kld_vnd + kld_weighted_gaussian

        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False, hard=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        criterion = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)

        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=True, z_posterior=self.posterior_params, hard=hard)

        # add this later
        self.kl = self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=self.posterior_params)

        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        # return -(self.reconstruction_loss + self.beta * self.kl)
        return -(self.mean_reconstruction_loss + self.beta * self.kl)
