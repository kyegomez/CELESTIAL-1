import copy
from inspect import isfunction
from typing import List
import tqdm
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from contextlib import contextmanager

from research.BindDiffusion.ldm.modules.diffusionmodules.util import noise_like


#HELPERS
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_updates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

def preprocess_model_args(args):
    # If args has layer_units, get the corresponding
    #     units.
    # If args get backbone, get the backbone model.
    args = copy.deepcopy(args)
    if 'layer_units' in args:
        layer_units = [
            get_unit()(i) for i in args.layer_units
        ]
        args.layer_units = layer_units
    if 'backbone' in args:
        args.backbone = get_model()(args.backbone)
    return args

@singleton
class get_model(object):
    def __init__(self):
        self.model = {}
        self.version = {}

    def register(self, model, name, version='x'):
        self.model[name] = model
        self.version[name] = version

    def __call__(self, cfg, verbose=True):
        """
        Construct model based on the config. 
        """
        t = cfg.type

        # the register is in each file
        if t.find('audioldm')==0:
            pass
        elif t.find('autoencoderkl')==0:
            pass
        elif t.find('optimus')==0:
            pass
            
        elif t.find('clip')==0:
            pass
        elif t.find('clap')==0:
            pass   
            
        elif t.find('sd')==0:
            pass
        elif t.find('codi')==0:
            pass
        elif t.find('openai_unet')==0:
            pass
        
        args = preprocess_model_args(cfg.args)
        net = self.model[t](**args)

        return net

    def get_version(self, name):
        return self.version[name]

def register(name, version='x'):
    def wrapper(class_):
        get_model().register(class_, name, version)
        return class_
    return wrapper

version = '0'
symbol = 'sd'

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

    
class DDPM(nn.Module):
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 use_ema=True,

                 beta_schedule="linear",
                 beta_linear_start=1e-4,
                 beta_linear_end=2e-2,
                 loss_type="l2",

                 clip_denoised=True,
                 cosine_s=8e-3,
                 given_betas=None,

                 l_simple_weight=1.,
                 original_elbo_weight=0.,
                 
                 v_posterior=0., # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 parameterization="eps",
                 use_positional_encodings=False,
                 learn_logvar=False, 
                 logvar_init=0, ):

        super().__init__()
        assert parameterization in ["eps", "x0"], \
            'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        highlight_print("Running in {} mode".format(self.parameterization))

        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.use_positional_encodings = use_positional_encodings

        from collections import OrderedDict
        self.model = nn.Sequential(OrderedDict([('diffusion_model', get_model()(unet_config))]))

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.v_posterior = v_posterior
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight

        self.register_schedule(
            given_betas=given_betas, 
            beta_schedule=beta_schedule, 
            timesteps=timesteps,
            linear_start=beta_linear_start, 
            linear_end=beta_linear_end, 
            cosine_s=cosine_s)

        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(
            fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, 
                          given_betas=None, 
                          beta_schedule="linear", 
                          timesteps=1000,
                          linear_start=1e-4, 
                          linear_end=2e-2, 
                          cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, \
            'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        value1 = extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        value2 = extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return value1*x_t -value2*noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

version = '0'
symbol = 'sd'

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

    
class DDPM(nn.Module):
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 use_ema=True,

                 beta_schedule="linear",
                 beta_linear_start=1e-4,
                 beta_linear_end=2e-2,
                 loss_type="l2",

                 clip_denoised=True,
                 cosine_s=8e-3,
                 given_betas=None,

                 l_simple_weight=1.,
                 original_elbo_weight=0.,
                 
                 v_posterior=0., # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 parameterization="eps",
                 use_positional_encodings=False,
                 learn_logvar=False, 
                 logvar_init=0, ):

        super().__init__()
        assert parameterization in ["eps", "x0"], \
            'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        highlight_print("Running in {} mode".format(self.parameterization))

        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.use_positional_encodings = use_positional_encodings

        from collections import OrderedDict
        self.model = nn.Sequential(OrderedDict([('diffusion_model', get_model()(unet_config))]))

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.v_posterior = v_posterior
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight

        self.register_schedule(
            given_betas=given_betas, 
            beta_schedule=beta_schedule, 
            timesteps=timesteps,
            linear_start=beta_linear_start, 
            linear_end=beta_linear_end, 
            cosine_s=cosine_s)

        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(
            fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, 
                          given_betas=None, 
                          beta_schedule="linear", 
                          timesteps=1000,
                          linear_start=1e-4, 
                          linear_end=2e-2, 
                          cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, \
            'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        value1 = extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        value2 = extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return value1*x_t -value2*noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

version = '0'
symbol = 'codi'
    
    
@register('codi', version)
class Celestial(DDPM):
    def __init__(self,
                 audioldm_cfg,
                 autokl_cfg,
                 optimus_cfg,
                 clip_cfg,
                 clap_cfg,
                 vision_scale_factor=0.1812,
                 text_scale_factor=4.3108,
                 audio_scale_factor=0.9228,
                 scale_by_std=False,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.audioldm = get_model()(audioldm_cfg)
        
        self.autokl = get_model()(autokl_cfg)
            
        self.optimus = get_model()(optimus_cfg)
            
        self.clip = get_model()(clip_cfg)
            
        self.clap = get_model()(clap_cfg)
        
        if not scale_by_std:
            self.vision_scale_factor = vision_scale_factor
            self.text_scale_factor = text_scale_factor
            self.audio_scale_factor = audio_scale_factor
        else:
            self.register_buffer("text_scale_factor", torch.tensor(text_scale_factor))
            self.register_buffer("audio_scale_factor", torch.tensor(audio_scale_factor))
            self.register_buffer('vision_scale_factor', torch.tensor(vision_scale_factor))

        self.freeze()
        
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    @property
    def device(self):
        return next(self.parameters()).device
            
    @torch.no_grad()
    def autokl_encode(self, image):
        encoder_posterior = self.autokl.encode(image)
        z = encoder_posterior.sample().to(image.dtype)
        return self.vision_scale_factor * z

    @torch.no_grad()
    def autokl_decode(self, z):
        z = 1. / self.vision_scale_factor * z
        return self.autokl.decode(z)

    @torch.no_grad()
    def optimus_encode(self, text):
        if isinstance(text, List):
            tokenizer = self.optimus.tokenizer_encoder
            token = [tokenizer.tokenize(sentence.lower()) for sentence in text]
            token_id = []
            for tokeni in token:
                token_sentence = [tokenizer._convert_token_to_id(i) for i in tokeni]
                token_sentence = tokenizer.add_special_tokens_single_sentence(token_sentence)
                token_id.append(torch.LongTensor(token_sentence))
            token_id = torch._C._nn.pad_sequence(token_id, batch_first=True, padding_value=0.0)[:, :512]
        else:
            token_id = text
        z = self.optimus.encoder(token_id, attention_mask=(token_id > 0))[1]
        z_mu, z_logvar = self.optimus.encoder.linear(z).chunk(2, -1)
        return z_mu.squeeze(1) * self.text_scale_factor

    @torch.no_grad()
    def optimus_decode(self, z, temperature=1.0):
        z = 1.0 / self.text_scale_factor * z
        return self.optimus.decode(z, temperature)
    
    @torch.no_grad()
    def audioldm_encode(self, audio, time=2.0):
        encoder_posterior = self.audioldm.encode(audio, time=time)
        z = encoder_posterior.sample().to(audio.dtype)
        return z * self.audio_scale_factor

    @torch.no_grad()
    def audioldm_decode(self, z):
        if (torch.max(torch.abs(z)) > 1e2):
            z = torch.clip(z, min=-10, max=10)
        z = 1.0 / self.audio_scale_factor * z
        return self.audioldm.decode(z)
    
    @torch.no_grad()
    def mel_spectrogram_to_waveform(self, mel):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.audioldm.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        return waveform
    
    @torch.no_grad()
    def clip_encode_text(self, text, encode_type='encode_text'):
        swap_type = self.clip.encode_type
        self.clip.encode_type = encode_type
        embedding = self.clip.encode(text)
        self.clip.encode_type = swap_type
        return embedding

    @torch.no_grad()
    def clip_encode_vision(self, vision, encode_type='encode_vision'):
        swap_type = self.clip.encode_type
        self.clip.encode_type = encode_type
        embedding = self.clip.encode(vision)
        self.clip.encode_type = swap_type
        return embedding
    
    @torch.no_grad()
    def clap_encode_audio(self, audio):
        embedding = self.clap(audio)
        return embedding

    def forward(self, x=None, c=None, noise=None, xtype='image', ctype='prompt', u=None, return_algined_latents=False):
        if isinstance(x, list):
            t = torch.randint(0, self.num_timesteps, (x[0].shape[0],), device=x[0].device).long()
        else:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, c, t, noise, xtype, ctype, u, return_algined_latents)

    def apply_model(self, x_noisy, t, cond, xtype='image', ctype='prompt', u=None, return_algined_latents=False):
        return self.model.diffusion_model(x_noisy, t, cond, xtype, ctype, u, return_algined_latents)

    def get_pixel_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=-0.0)
        return loss

    def get_text_loss(self, pred, target):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)    
        return loss

    def p_losses(self, x_start, cond, t, noise=None, xtype='image', ctype='prompt', u=None, return_algined_latents=False):
        if isinstance(x_start, list):
            noise = [torch.randn_like(x_start_i) for x_start_i in x_start] if noise is None else noise
            x_noisy = [self.q_sample(x_start=x_start_i, t=t, noise=noise_i) for x_start_i, noise_i in zip(x_start, noise)]
            model_output = self.apply_model(x_noisy, t, cond, xtype, ctype, u, return_algined_latents)
            if return_algined_latents:
                return model_output
            

            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = noise
            else:
                raise NotImplementedError()
            
            loss = 0.0
            for model_output_i, target_i, xtype_i in zip(model_output, target, xtype):
                if xtype_i == 'image':
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3])
                elif xtype_i == 'video':
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3, 4])
                elif xtype_i == 'text':
                    loss_simple = self.get_text_loss(model_output_i, target_i).mean([1])
                elif xtype_i == 'audio':
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3])
                loss += loss_simple.mean()
            return loss / len(xtype)
        
        else:    
            noise = torch.randn_like(x_start) if noise is None else noise
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            model_output = self.apply_model(x_noisy, t, cond, xtype, ctype)


            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = noise
            else:
                raise NotImplementedError()

            if xtype == 'image':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3])
            elif xtype == 'video':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
            elif xtype == 'text':
                loss_simple = self.get_text_loss(model_output, target).mean([1])
            elif xtype == 'audio':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3])
            loss = loss_simple.mean()
            return loss