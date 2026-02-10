import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.utils import cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule, Losses, extract, Silent


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action, max_state, device, beta_schedule='linear',
                 n_timesteps=100, loss_type='l2', clip_denoised=True, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.max_state = max_state
        self.model = model
        self.device = device

        min_action = np.tile(np.array([-0.01, 0.01, 0.01]), len(max_action)//3)
        self.max_values = torch.from_numpy(np.maximum(min_action, self.max_action)).float().to(device)
        self.min_values = torch.from_numpy(np.minimum(min_action, self.max_action)).float().to(device)
        # next_action = next_action.clip(min_values, max_values)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # self.loss_fn = Losses[loss_type]()
        self.loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

    # sampling
    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is noise;
        otherwise, model predicts x0 directly
        :param x_t:
        :param noise:
        :return:
        """
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def predict_start_from_noise_state(self, s_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, s_t.shape) * s_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, s_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_state_posterior(self, s_start, s_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, s_t.shape) * s_start +
            extract(self.posterior_mean_coef2, t, s_t.shape) * s_t
        )
        posterior_variance = extract(self.posterior_variance, t, s_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, s_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        action_noise, _ = self.model(x, t, s, x, s)
        x_recon = self.predict_start_from_noise(x, t=t, noise=action_noise)

        if self.clip_denoised:
            # x_recon.clamp_(-self.max_action, self.max_action)
            x_recon.clamp_(self.min_values, self.max_values)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_state_mean_variance(self, s, t, state, action):
        _, state_noise = self.model(action, t, state, action, s)
        s_recon = self.predict_start_from_noise_state(s, t=t, noise=state_noise)

        if self.clip_denoised:
            s_recon.clamp_(-self.max_state, self.max_state)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_state_posterior(s_start=s_recon, s_t=s, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_state(self, s, t, state, action):
        b, *_, device = *s.shape, s.device
        model_mean, _, model_log_variance = self.p_state_mean_variance(s=s, t=t, state=state, action=action)
        noise = torch.randn_like(s)
        # No nosie when t== 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(s.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        # progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
            # progress.update({'t': i})

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def p_sample_state_loop(self, state, action, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        s = torch.randn(shape, device=device)
        if return_diffusion:
            diffusion_state = [s]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            s = self.p_sample_state(s, timesteps, state, action)

            if return_diffusion:
                diffusion_state.append(s)

        if return_diffusion:
            return s, torch.stack(diffusion_state, dim=1)
        else:
            return s

    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)

        # return action.clamp_(-self.max_action, self.max_action)
        return action.clamp_(self.min_values, self.max_values)

    def sample_state(self, state, action, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.state_dim)

        state = self.p_sample_state_loop(state, action, shape, *args, **kwargs)
        return state.clamp_(-self.max_state, self.max_state)

    # Train
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # print("~~~~~~~~~~~~~~~~~~~~~ device: ", self.device)
        # print("~~~~~~~~~~~~~~~~~~~~~ x_start device: ", x_start.device)
        #
        # print("~~~~~~~~~~~~~~~~~~~~~ noise device: ", noise.device)
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def q_state_sample(self, s_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(s_start)

        sample_state = (
                extract(self.sqrt_alphas_cumprod, t, s_start.shape) * s_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, s_start.shape) * noise
        )
        return sample_state

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon, _ = self.model(x_noisy, t, state, x_noisy, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            # loss = self.loss_fn(x_recon, noise, weights)
            loss = self.loss_fn(x_recon.float(), noise.float()).mean()
        else:
            # loss = self.loss_fn(x_recon, x_start, weights)
            loss = self.loss_fn(x_recon.float(), x_start.float()).mean()
        return loss

    def p_state_losses(self, s_start, state, action, t, weights=1.0):
        noise = torch.randn_like(s_start)

        s_noisy = self.q_state_sample(s_start=s_start, t=t, noise=noise)

        _, s_recon = self.model(action, t, state, action, s_noisy)

        assert noise.shape == s_recon.shape

        if self.predict_epsilon:
            # loss = self.loss_fn(s_recon, noise, weights)
            loss = self.loss_fn(s_recon.float(), noise.float()).mean()
        else:
            # loss = self.loss_fn(s_recon, s_start, weights)
            loss = self.loss_fn(s_recon.float(), s_start.float()).mean()
        return loss

    def loss_action(self, x, state, weights=1.0):
        batch_size = len(x)
        # batch_size = x.size(0)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
        return self.p_losses(x, state, t, weights)

    def loss_state(self, s, state, action, weights=1.0):
        batch_size = len(s)
        t = torch.randint(0, self.n_timesteps, (batch_size, ), device=s.device).long()
        return self.p_state_losses(s, state, action, t, weights)

    def loss(self, state, action, next_state, weights=1.0):
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        # print("@@@@@@@@@@@self.loss_action(action, state, weights)", self.loss_action(action, state, weights).shape)
        # print("@@@@@@@@@@@self.loss_state(next_state, state, action, weights)", self.loss_state(next_state, state, action, weights).shape)
        return self.loss_action(action, state, weights) + self.loss_state(next_state, state, action, weights)

    def forward(self, state, *args, **kwargs):
        gen_action = self.sample(state, *args, **kwargs)
        gen_next_state = self.sample_state(state, gen_action, *args, **kwargs)
        return gen_action, gen_next_state
