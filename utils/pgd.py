import torch
import numpy as np
from tqdm import tqdm
import torchvision
from ldm.models.diffusion.ddim import DDIMSampler




ATTACK_TYPE = [
    'PGD',
    'PGD_SDS',
    'Diff_PGD',
    'Diff_PGD_SDS'
]

class SDEdit():
    def __init__(self, net):
        self.dm = net.model # SD model
        self.c = net.condition
    
    @torch.no_grad()
    def edit(self, x, guidance, restep=None):
        # create DDIM Sampler
        # guidance from 0-1
        assert guidance >= 0  and guidance < 1
        
        
        if restep is None:
            ##
            # directly use one step denoise
            ##
            t_guidance = int(self.dm.num_timesteps * guidance)
            # latent z
            z = self.dm.get_first_stage_encoding(self.dm.encode_first_stage(x)).to(x.device)
            t = torch.full((z.shape[0],), t_guidance, device=z.device, dtype=torch.long)
                            
            # sample noise
            noise = torch.randn_like(z)
            z_noisy = self.dm.q_sample(x_start=z, t=t, noise=noise)
            
            # get z_t
            cnd = self.dm.get_learned_conditioning(self.c)
            eps_pred = self.dm.apply_model(z_noisy, t, cond=cnd) # \hat{eps}
            
            # get \hat{x_0}
            
            z_0_pred = self.dm.predict_start_from_noise(z_noisy, t, eps_pred)
            x_0_pred = self.dm.decode_first_stage(z_0_pred)
            return x_0_pred   
                     
        else:
            # ddim
            ddim_steps = int(restep[4:])
            sampler = DDIMSampler(self.dm, schedule="linear")
            sampler.make_schedule(ddim_steps, 'uniform')
            
            t_guidance = int(ddim_steps * guidance)
            
            t = torch.full((z.shape[0],), t_guidance, device=z.device, dtype=torch.long)
            
            
            
            pass    
    
    



class Linf_PGD():
    def __init__(self, net, fn, epsilon, steps, eps_iter, clip_min, clip_max=1, targeted=True, attack_type='PGD'):
        self.net = net
        self.fn = fn
        self.eps = epsilon
        self.step_size = eps_iter
        self.iters = steps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        
        self.attack_type = attack_type
        
        if self.attack_type not in ATTACK_TYPE:
            raise AttributeError(f"self.attack_type should be in {ATTACK_TYPE}, \
                                 {self.attack_type} is an undefined")
        
    # interface to call all the attacks
    def perturbe(self, X, y):
        if self.attack_type == 'PGD':
            return self.pgd(X, y)
        elif self.attack_type == 'PGD_SDS':
            return self.pgd_sds(X, y)
    
    # traditional pgd
    def pgd(self, X, y):
        
        # add uniform random start [-eps, eps]
        X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*self.eps-self.eps).cuda()
        
        pbar = tqdm(range(self.iters))
        
        # modified from photoguard by Salman et al.
        for i in pbar:
            actual_step_size = self.step_size - (self.step_size - self.step_size / 100) / self.iters * i  

            X_adv.requires_grad_(True)

            #loss = (model(X_adv).latent_dist.mean).norm()
            
            loss = self.fn(self.net(X_adv), y)

            pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

            grad, = torch.autograd.grad(loss, [X_adv])
            
            # update
            X_adv = X_adv - grad.detach().sign() * actual_step_size
            # clip
            X_adv = torch.minimum(torch.maximum(X_adv, X - self.eps), X + self.eps)
            X_adv.data = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max)
            X_adv.grad = None    
            
                
        return X_adv
    
    def pgd_sds(self, X, net, c, label=None, random_start=False):
        
        
        editor = SDEdit(net)
        
        torchvision.utils.save_image(
            (1+torch.cat([X,
                       editor.edit(X, 0.01),
                       editor.edit(X, 0.05),
                       editor.edit(X, 0.1),
                       editor.edit(X, 0.2),
                       editor.edit(X, 0.3)], -1))/2,
            'test/test_sdedit.png'
        )
        
        # gradient required from SDS, torch.no_grad() here
        if random_start:
            print("using random_start")
            X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*self.eps-self.eps).cuda()
        else:
            X_adv = X.clone().detach()
        
        pbar = tqdm(range(self.iters))
        # z_raw = None
        
        dm = net.model # SD model to call
        
        for i in pbar:
            print(i)
            

            actual_step_size = self.step_size - (self.step_size - self.step_size / 100) / self.iters * i 
            
            # SDS update, with only forward function
            with torch.no_grad():
                # to latent
                z = dm.get_first_stage_encoding(dm.encode_first_stage(X_adv)).to(X.device)
                
                # sample noise
                t = torch.randint(0, dm.num_timesteps, (z.shape[0],), device=z.device).long()
                noise = torch.randn_like(z)
                
                # get z_t
                z_noisy = dm.q_sample(x_start=z, t=t, noise=noise)
                cnd = dm.get_learned_conditioning(c)
                eps_pred = dm.apply_model(z_noisy, t, cond=cnd) # \hat{eps}
                
                # update z
                grad = (eps_pred - noise)
            
            torch.cuda.empty_cache()
            # get gradient wrt VAE (with gradient)
            X_adv = X_adv.clone().detach()
            X_adv.requires_grad_(True)
            z = dm.get_first_stage_encoding(dm.encode_first_stage(X_adv)).to(X.device)
            z.backward(gradient=grad)
            g_x = X_adv.grad.detach()
            
            # update x_adv
            X_adv = X_adv - g_x.detach().sign() * actual_step_size
            X_adv = torch.minimum(torch.maximum(X_adv, X - self.eps), X + self.eps)
            X_adv.data = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max)
            X_adv.grad=None
        
        torchvision.utils.save_image(
            (1+torch.cat([X_adv,
                       editor.edit(X_adv, 0.01),
                       editor.edit(X_adv, 0.05),
                       editor.edit(X_adv, 0.1),
                       editor.edit(X_adv, 0.2),
                       editor.edit(X_adv, 0.3)], -1))/2,
            'test/test_sdedit_adv.png'
        )
   
        return X_adv
                
                
                
                
        
        
    
    
    def fast_diff_pgd(self, x, label=None, ):
        pass
        