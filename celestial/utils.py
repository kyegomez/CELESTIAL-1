from argparse import Namespace
from typing import Union

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import random
import torch
from collections import OrderedDict

import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision import transforms as tvtrans

from decord import VideoReader, cpu, gpu




REGISTRIES = {}


def setup_registry(registry_name: str,
                   base_class=None,
                   default=None,
                   required=False):
    assert registry_name.startswith('--')
    registry_name = registry_name[2:].replace('-', '_')

    REGISTRY = {}
    REGISTRY_CLASS_NAMES = set()
    DATACLASS_REGISTRY = {}

    # maintain a registry of all registries
    if registry_name in REGISTRIES:
        return  # registry already exists
    REGISTRIES[registry_name] = {
        'registry': REGISTRY,
        'default': default,
        'dataclass_registry': DATACLASS_REGISTRY,
    }

    def build_x(cfg: Union[DictConfig, str, Namespace], *extra_args,
                **extra_kwargs):

        assert isinstance(cfg, str)
        choice = cfg
        if choice in DATACLASS_REGISTRY:
            cfg = DATACLASS_REGISTRY[choice]()

        if choice is None:
            if required:
                raise ValueError('{} is required!'.format(registry_name))
            return None

        cls = REGISTRY[choice]
        if hasattr(cls, 'build_' + registry_name):
            builder = getattr(cls, 'build_' + registry_name)
        else:
            builder = cls
        return builder(cfg, *extra_args, **extra_kwargs)

    def register_x(name, dataclass=None):
        def register_x_cls(cls):
            if name in REGISTRY:
                raise ValueError('Cannot register duplicate {} ({})'.format(
                    registry_name, name))
            if cls.__name__ in REGISTRY_CLASS_NAMES:
                raise ValueError(
                    'Cannot register {} with duplicate class name ({})'.format(
                        registry_name, cls.__name__))
            if base_class is not None and not issubclass(cls, base_class):
                raise ValueError('{} must extend {}'.format(
                    cls.__name__, base_class.__name__))

            cls.__dataclass = dataclass
            if cls.__dataclass is not None:
                DATACLASS_REGISTRY[name] = cls.__dataclass

                cs = ConfigStore.instance()
                node = dataclass()
                node._name = name
                cs.store(name=name,
                         group=registry_name,
                         node=node,
                         provider='layoutlmft')

            REGISTRY[name] = cls

            return cls

        return register_x_cls

    return build_x, register_x, REGISTRY, DATACLASS_REGISTRY


###############
# text helper #
###############

def remove_duplicate_word(tx):
    def combine_words(input, length):
        combined_inputs = []
        if len(splitted_input)>1:
            for i in range(len(input)-1):
                combined_inputs.append(input[i]+" "+last_word_of(splitted_input[i+1],length)) #add the last word of the right-neighbour (overlapping) sequence (before it has expanded), which is the next word in the original sentence
        return combined_inputs, length+1

    def remove_duplicates(input, length):
        bool_broke=False #this means we didn't find any duplicates here
        for i in range(len(input) - length):
            if input[i]==input[i + length]: #found a duplicate piece of sentence!
                for j in range(0, length): #remove the overlapping sequences in reverse order
                    del input[i + length - j]
                bool_broke = True
                break #break the for loop as the loop length does not matches the length of splitted_input anymore as we removed elements
        if bool_broke:
            return remove_duplicates(input, length) #if we found a duplicate, look for another duplicate of the same length
        return input

    def last_word_of(input, length):
        splitted = input.split(" ")
        if len(splitted)==0:
            return input
        else:
            return splitted[length-1]

    def split_and_puncsplit(text):
        tx = text.split(" ")
        txnew = []
        for txi in tx:
            txqueue=[]
            while True:
                if txi[0] in '([{':
                    txqueue.extend([txi[:1], '<puncnext>'])
                    txi = txi[1:]
                    if len(txi) == 0:
                        break
                else:
                    break
            txnew += txqueue
            txstack=[]
            if len(txi) == 0:
                continue
            while True:
                if txi[-1] in '?!.,:;}])':
                    txstack = ['<puncnext>', txi[-1:]] + txstack
                    txi = txi[:-1]
                    if len(txi) == 0:
                        break
                else:
                    break
            if len(txi) != 0:
                txnew += [txi]
            txnew += txstack
        return txnew

    if tx == '':
        return tx

    splitted_input = split_and_puncsplit(tx)
    word_length = 1
    intermediate_output = False
    while len(splitted_input)>1:
        splitted_input = remove_duplicates(splitted_input, word_length)
        if len(splitted_input)>1:
            splitted_input, word_length = combine_words(splitted_input, word_length)
        if intermediate_output:
            print(splitted_input)
            print(word_length)
    output = splitted_input[0]
    output = output.replace(' <puncnext> ', '')
    return output

#################
# vision helper #
#################

def regularize_image(x, image_size=512):
    BICUBIC = T.InterpolationMode.BICUBIC
    if isinstance(x, str):
        x = Image.open(x)
    elif isinstance(x, Image.Image):
        x = x.convert('RGB')
    elif isinstance(x, np.ndarray):
        x = Image.fromarray(x).convert('RGB')
    elif isinstance(x, torch.Tensor):
        pass
    else:
        assert False, 'Unknown image type'
    
    transforms = T.Compose([
                T.RandomCrop(min(x.size)),
                T.Resize(
                    (image_size, image_size),
                    interpolation=BICUBIC,
                ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
    x = transforms(x)
    assert (x.shape[1]==image_size) & (x.shape[2]==image_size), \
        'Wrong image size'

    x = x * 2 - 1
    return x


def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[2]
    height = img.shape[1]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))
    if len(img.shape) == 3:
        center_cropped_img = img[:, top:bottom, left:right]
    else:
        center_cropped_img = img[:, top:bottom, left:right, ...]

    return center_cropped_img

def _transform(n_px):
    return Compose([
        Resize([n_px, n_px], interpolation=T.InterpolationMode.BICUBIC),])
    
def regularize_video(video, image_size=256):
    min_shape = min(video.shape[1:3])
    video = center_crop(video, min_shape, min_shape)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)
    video = _transform(image_size)(video)       
    video = video/255.0 * 2.0 - 1.0
    return video.permute(1, 0, 2, 3)

def time_to_indices(video_reader, time):
    times = video_reader.get_frame_timestamp(range(len(video_reader))).mean(-1)
    indices = np.searchsorted(times, time)
    # Use `np.bitwise_or` so it works both with scalars and numpy arrays.
    return np.where(np.bitwise_or(indices == 0, times[indices] - time <= time - times[indices - 1]), indices,
                    indices - 1)

def load_video(video_path, sample_duration=8.0, num_frames=8):
    sample_duration = 4.0
    num_frames = 4

    vr = VideoReader(video_path, ctx=cpu(0))
    framerate = vr.get_avg_fps()
    video_frame_len = len(vr)
    video_len = video_frame_len/framerate
    sample_duration = min(sample_duration, video_len)

    if video_len > sample_duration:
        s = random.random() * (video_len - sample_duration)
        t = s + sample_duration
        start, end = time_to_indices(vr, [s, t])
        end = min(video_frame_len-1, end)
        start = min(start, end-1)
        downsamlp_indices = np.linspace(start, end, num_frames, endpoint=True).astype(int).tolist()
    else:            
        downsamlp_indices = np.linspace(0, video_frame_len-1, num_frames, endpoint=True).astype(int).tolist()

    video = vr.get_batch(downsamlp_indices).asnumpy()
    return video


###############
# some helper #
###############

def atomic_save(cfg, net, opt, step, path):
    if isinstance(net, (torch.nn.DataParallel,
                        torch.nn.parallel.DistributedDataParallel)):
        netm = net.module
    else:
        netm = net
    sd = netm.state_dict()
    slimmed_sd = [(ki, vi) for ki, vi in sd.items()
        if ki.find('first_stage_model')!=0 and ki.find('cond_stage_model')!=0]

    checkpoint = {
        "config" : cfg,
        "state_dict" : OrderedDict(slimmed_sd),
        "step" : step}
    if opt is not None:
        checkpoint['optimizer_states'] = opt.state_dict()
    import io
    import fsspec
    bytesbuffer = io.BytesIO()
    torch.save(checkpoint, bytesbuffer)
    with fsspec.open(path, "wb") as f:
        f.write(bytesbuffer.getvalue())

def load_state_dict(net, cfg):
    pretrained_pth_full  = cfg.get('pretrained_pth_full' , None)
    pretrained_ckpt_full = cfg.get('pretrained_ckpt_full', None)
    pretrained_pth       = cfg.get('pretrained_pth'      , None)
    pretrained_ckpt      = cfg.get('pretrained_ckpt'     , None)
    pretrained_pth_dm    = cfg.get('pretrained_pth_dm'   , None)
    pretrained_pth_ema   = cfg.get('pretrained_pth_ema'  , None)
    strict_sd = cfg.get('strict_sd', False)
    errmsg = "Overlapped model state_dict! This is undesired behavior!"

    if pretrained_pth_full is not None or pretrained_ckpt_full is not None:
        assert (pretrained_pth is None) and \
               (pretrained_ckpt is None) and \
               (pretrained_pth_dm is None) and \
               (pretrained_pth_ema is None), errmsg            
        if pretrained_pth_full is not None:
            target_file = pretrained_pth_full
            sd = torch.load(target_file, map_location='cpu')
            assert pretrained_ckpt is None, errmsg
        else:
            target_file = pretrained_ckpt_full
            sd = torch.load(target_file, map_location='cpu')['state_dict']
        print('Load full model from [{}] strict [{}].'.format(
            target_file, strict_sd))
        net.load_state_dict(sd, strict=strict_sd)

    if pretrained_pth is not None or pretrained_ckpt is not None:
        assert (pretrained_ckpt_full is None) and \
               (pretrained_pth_full is None) and \
               (pretrained_pth_dm is None) and \
               (pretrained_pth_ema is None), errmsg
        if pretrained_pth is not None:
            target_file = pretrained_pth
            sd = torch.load(target_file, map_location='cpu')
            assert pretrained_ckpt is None, errmsg
        else:
            target_file = pretrained_ckpt
            sd = torch.load(target_file, map_location='cpu')['state_dict']
        print('Load model from [{}] strict [{}].'.format(
            target_file, strict_sd))
        sd_extra = [(ki, vi) for ki, vi in net.state_dict().items() \
            if ki.find('first_stage_model')==0 or ki.find('cond_stage_model')==0]
        sd.update(OrderedDict(sd_extra))
        net.load_state_dict(sd, strict=strict_sd)

    if pretrained_pth_dm is not None:
        assert (pretrained_ckpt_full is None) and \
               (pretrained_pth_full is None) and \
               (pretrained_pth is None) and \
               (pretrained_ckpt is None), errmsg
        print('Load diffusion model from [{}] strict [{}].'.format(
            pretrained_pth_dm, strict_sd))
        sd = torch.load(pretrained_pth_dm, map_location='cpu')
        net.model.diffusion_model.load_state_dict(sd, strict=strict_sd)

    if pretrained_pth_ema is not None:
        assert (pretrained_ckpt_full is None) and \
               (pretrained_pth_full is None) and \
               (pretrained_pth is None) and \
               (pretrained_ckpt is None), errmsg
        print('Load unet ema model from [{}] strict [{}].'.format(
            pretrained_pth_ema, strict_sd))
        sd = torch.load(pretrained_pth_ema, map_location='cpu')
        net.model_ema.load_state_dict(sd, strict=strict_sd)

def auto_merge_imlist(imlist, max=64):
    imlist = imlist[0:max]
    h, w = imlist[0].shape[0:2]
    num_images = len(imlist)
    num_row = int(np.sqrt(num_images))
    num_col = num_images//num_row + 1 if num_images%num_row!=0 else num_images//num_row
    canvas = np.zeros([num_row*h, num_col*w, 3], dtype=np.uint8)
    for idx, im in enumerate(imlist):
        hi = (idx // num_col) * h
        wi = (idx % num_col) * w
        canvas[hi:hi+h, wi:wi+w, :] = im
    return canvas

def latent2im(net, latent):
    single_input = len(latent.shape) == 3
    if single_input:
        latent = latent[None]
    im = net.decode_image(latent.to(net.device))
    im = torch.clamp((im+1.0)/2.0, min=0.0, max=1.0)
    im = [tvtrans.ToPILImage()(i) for i in im]
    if single_input:
        im = im[0]
    return im

def im2latent(net, im):
    single_input = not isinstance(im, list)
    if single_input:
        im = [im]
    im = torch.stack([tvtrans.ToTensor()(i) for i in im], dim=0)
    im = (im*2-1).to(net.device)
    z = net.encode_image(im)
    if single_input:
        z = z[0]
    return z

class color_adjust(object):
    def __init__(self, ref_from, ref_to):
        x0, m0, std0 = self.get_data_and_stat(ref_from)
        x1, m1, std1 = self.get_data_and_stat(ref_to)
        self.ref_from_stat = (m0, std0)
        self.ref_to_stat   = (m1, std1)
        self.ref_from = self.preprocess(x0).reshape(-1, 3)
        self.ref_to = x1.reshape(-1, 3)

    def get_data_and_stat(self, x):
        if isinstance(x, str):
            x = np.array(PIL.Image.open(x))
        elif isinstance(x, PIL.Image.Image):
            x = np.array(x)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, min=0.0, max=1.0)
            x = np.array(tvtrans.ToPILImage()(x))
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError
        x = x.astype(float)
        m = np.reshape(x, (-1, 3)).mean(0)
        s = np.reshape(x, (-1, 3)).std(0)
        return x, m, s

    def preprocess(self, x):
        m0, s0 = self.ref_from_stat
        m1, s1 = self.ref_to_stat
        y = ((x-m0)/s0)*s1 + m1
        return y

    def __call__(self, xin, keep=0, simple=False):
        xin, _, _ = self.get_data_and_stat(xin)
        x = self.preprocess(xin)
        if simple: 
            y = (x*(1-keep) + xin*keep)
            y = np.clip(y, 0, 255).astype(np.uint8)
            return y

        h, w = x.shape[:2]
        x = x.reshape(-1, 3)
        y = []
        for chi in range(3):
            yi = self.pdf_transfer_1d(self.ref_from[:, chi], self.ref_to[:, chi], x[:, chi])
            y.append(yi)

        y = np.stack(y, axis=1)
        y = y.reshape(h, w, 3)
        y = (y.astype(float)*(1-keep) + xin.astype(float)*keep)
        y = np.clip(y, 0, 255).astype(np.uint8)
        return y

    def pdf_transfer_1d(self, arr_fo, arr_to, arr_in, n=600):
        arr = np.concatenate((arr_fo, arr_to))
        min_v = arr.min() - 1e-6
        max_v = arr.max() + 1e-6
        min_vto = arr_to.min() - 1e-6
        max_vto = arr_to.max() + 1e-6
        xs = np.array(
            [min_v + (max_v - min_v) * i / n for i in range(n + 1)])
        hist_fo, _ = np.histogram(arr_fo, xs)
        hist_to, _ = np.histogram(arr_to, xs)
        xs = xs[:-1]
        # compute probability distribution
        cum_fo = np.cumsum(hist_fo)
        cum_to = np.cumsum(hist_to)
        d_fo = cum_fo / cum_fo[-1]
        d_to = cum_to / cum_to[-1]
        # transfer
        t_d = np.interp(d_fo, d_to, xs)
        t_d[d_fo <= d_to[ 0]] = min_vto
        t_d[d_fo >= d_to[-1]] = max_vto
        arr_out = np.interp(arr_in, xs, t_d)
        return arr_out
    


from email.policy import strict
import torch
import torchvision.models
import os.path as osp
import copy
# from core.common.logger import print_log 
from .utils import \
    get_total_param, get_total_param_sum, \
    get_unit

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
            from ..latent_diffusion.vae import audioldm
        elif t.find('autoencoderkl')==0:
            from ..latent_diffusion.vae import autokl
        elif t.find('optimus')==0:
            from ..latent_diffusion.vae import optimus
            
        elif t.find('clip')==0:
            from ..encoders import clip
        elif t.find('clap')==0:
            from ..encoders import clap   
            
        elif t.find('sd')==0:
            from .. import sd
        elif t.find('codi')==0:
            from .. import codi
        elif t.find('openai_unet')==0:
            from ..latent_diffusion import diffusion_unet
        
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


import torch
import torch.optim as optim
import numpy as np
import itertools

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

class get_optimizer(object):
    def __init__(self):
        self.optimizer = {}
        self.register(optim.SGD, 'sgd')
        self.register(optim.Adam, 'adam')
        self.register(optim.AdamW, 'adamw')

    def register(self, optim, name):
        self.optimizer[name] = optim

    def __call__(self, net, cfg):
        if cfg is None:
            return None
        t = cfg.type
        if isinstance(net, (torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel)):
            netm = net.module
        else:
            netm = net
        pg = getattr(netm, 'parameter_group', None)

        if pg is not None:
            params = []
            for group_name, module_or_para in pg.items():
                if not isinstance(module_or_para, list):
                    module_or_para = [module_or_para]

                grouped_params = [mi.parameters() if isinstance(mi, torch.nn.Module) else [mi] for mi in module_or_para]
                grouped_params = itertools.chain(*grouped_params)
                pg_dict = {'params':grouped_params, 'name':group_name}
                params.append(pg_dict)
        else:
            params = net.parameters()
        return self.optimizer[t](params, lr=0, **cfg.args)
    

import torch
import torch.optim as optim
import numpy as np
import copy
from ... import sync
from ...cfg_holder import cfg_unique_holder as cfguh

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_scheduler(object):
    def __init__(self):
        self.lr_scheduler = {}

    def register(self, lrsf, name):
        self.lr_scheduler[name] = lrsf

    def __call__(self, cfg):
        if cfg is None:
            return None
        if isinstance(cfg, list):
            schedulers = []
            for ci in cfg:
                t = ci.type
                schedulers.append(
                    self.lr_scheduler[t](**ci.args))
            if len(schedulers) == 0:
                raise ValueError
            else:
                return compose_scheduler(schedulers)
        t = cfg.type
        return self.lr_scheduler[t](**cfg.args)
        

def register(name):
    def wrapper(class_):
        get_scheduler().register(class_, name)
        return class_
    return wrapper

class template_scheduler(object):
    def __init__(self, step):
        self.step = step

    def __getitem__(self, idx):
        raise ValueError

    def set_lr(self, optim, new_lr, pg_lrscale=None):
        """
        Set Each parameter_groups in optim with new_lr
        New_lr can be find according to the idx.
        pg_lrscale tells how to scale each pg.
        """
        # new_lr = self.__getitem__(idx)
        pg_lrscale = copy.deepcopy(pg_lrscale)
        for pg in optim.param_groups:
            if pg_lrscale is None:
                pg['lr'] = new_lr
            else:
                pg['lr'] = new_lr * pg_lrscale.pop(pg['name'])
        assert (pg_lrscale is None) or (len(pg_lrscale)==0), \
            "pg_lrscale doesn't match pg"

@register('constant')
class constant_scheduler(template_scheduler):
    def __init__(self, lr, step):
        super().__init__(step)
        self.lr = lr

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        return self.lr

@register('poly')
class poly_scheduler(template_scheduler):
    def __init__(self, start_lr, end_lr, power, step):
        super().__init__(step)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.power = power

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        a, b = self.start_lr, self.end_lr
        p, n = self.power, self.step
        return b + (a-b)*((1-idx/n)**p)

@register('linear')
class linear_scheduler(template_scheduler):
    def __init__(self, start_lr, end_lr, step):
        super().__init__(step)
        self.start_lr = start_lr
        self.end_lr = end_lr

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        a, b, n = self.start_lr, self.end_lr, self.step
        return b + (a-b)*(1-idx/n)

@register('multistage')
class constant_scheduler(template_scheduler):
    def __init__(self, start_lr, milestones, gamma, step):
        super().__init__(step)
        self.start_lr = start_lr
        m = [0] + milestones + [step]
        lr_iter = start_lr
        self.lr = []
        for ms, me in zip(m[0:-1], m[1:]):
            for _ in range(ms, me):
                self.lr.append(lr_iter)
            lr_iter *= gamma

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        return self.lr[idx]

class compose_scheduler(template_scheduler):
    def __init__(self, schedulers):
        self.schedulers = schedulers
        self.step = [si.step for si in schedulers]
        self.step_milestone = []
        acc = 0
        for i in self.step:
            acc += i
            self.step_milestone.append(acc)
        self.step = sum(self.step)

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        ms = self.step_milestone
        for idx, (mi, mj) in enumerate(zip(ms[:-1], ms[1:])):
            if mi <= idx < mj:
                return self.schedulers[idx-mi]
        raise ValueError

####################
# lambda schedular #
####################

class LambdaWarmUpCosineScheduler(template_scheduler):
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, 
                 base_lr,
                 warm_up_steps, 
                 lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        cfgt = cfguh().cfg.train
        bs = cfgt.batch_size
        if 'gradacc_every' not in cfgt:
            print('Warning, gradacc_every is not found in xml, use 1 as default.')
        acc = cfgt.get('gradacc_every', 1)
        self.lr_multi = base_lr * bs * acc
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: 
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __getitem__(self, idx):
        return self.schedule(idx) * self.lr_multi

class LambdaWarmUpCosineScheduler2(template_scheduler):
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """
    def __init__(self, 
                 base_lr,
                 warm_up_steps, 
                 f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        cfgt = cfguh().cfg.train
        # bs = cfgt.batch_size
        # if 'gradacc_every' not in cfgt:
        #     print('Warning, gradacc_every is not found in xml, use 1 as default.')
        # acc = cfgt.get('gradacc_every', 1)
        # self.lr_multi = base_lr * bs * acc
        self.lr_multi = base_lr
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                    1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __getitem__(self, idx):
        return self.schedule(idx) * self.lr_multi

@register('stable_diffusion_linear')
class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    def schedule(self, n):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: 
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                      f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f