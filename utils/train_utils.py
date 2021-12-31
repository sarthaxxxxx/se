import torch.optim as optim

def create_optimizer(cfg, gen, disc):
    """ Return optimizer based on user config. 

    Parameters
    ----------
    cfg: main.Configuration
        config file
    gen: 
        generator model 

    disc:
        discriminator model

    Returns
    -------
    g_optim: 
        Generator optimizer
    d_optim:
        Discriminator optimizer
    """

    gen_params = [params for params in gen.parameters() if params.requires_grad]
    disc_params = [params for params in disc.parameters() if params.requires_grad]

    if cfg.optim == 'RMSProp':
        g_optim = optim.RMSprop(gen_params, 
                                lr = cfg.gen_lr)
        d_optim = optim.RMSprop(disc_params, 
                                lr = cfg.disc_lr)
    elif cfg.optim == 'Adam':
        g_optim = optim.Adam(gen_params,
                            lr = cfg.gen_lr, 
                            betas = (cfg.beta, cfg.beta))
        d_optim = optim.Adam(disc_params,
                            lr = cfg.disc_lr, 
                            betas = (cfg.beta, cfg.beta))
    else: raise ValueError("INVALID OPTIMIZER NAME !!!")    
    return g_optim, d_optim



def reset_grad(g_optim, d_optim):
    """ Reset gradients of optimizers. """
    
    g_optim.zero_grad()
    d_optim.zero_grad()
