# -*- coding: utf-8 -*-
import torch
def loss(model,x,y,mean,factor):
    z=torch.randn_like(y);initial=mean+z@factor.T;t=torch.rand(len(y),device=y.device);u=(1-t[:,None])*initial+t[:,None]*y
    return ((model(u,x,t)-(y-initial))**2).mean()
@torch.no_grad()
def generate(model,x,mean,factor,seed,members=200,steps=20):
    model.eval();g=torch.Generator(device=x.device).manual_seed(seed);expanded=x[:,None].expand(-1,members,-1).reshape(-1,130)
    z=torch.randn((len(expanded),168),device=x.device,generator=g);u=mean+z@factor.T
    for k in range(steps):u=u+model(u,expanded,torch.full((len(u),),k/steps,device=x.device))/steps
    return ((u.reshape(len(x),members,168)+1)/2).clamp(0,1).cpu().numpy()
