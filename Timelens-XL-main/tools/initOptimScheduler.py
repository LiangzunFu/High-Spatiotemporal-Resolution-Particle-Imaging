import torch


def init_optimizer(model, epoch, params, scheduler_dict=None):
    if params.training_config.optim.name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.training_config.optim.optim_params.lr)
    elif params.training_config.optim.name == 'AdamW':
        optim_params = params.training_config.optim.optim_params
        optimizer = torch.optim.AdamW(
            model.parameters(),
            **optim_params
        )
    else:
        print('Optim only support Adam, use Adam instead now~')
        optimizer = torch.optim.Adam(model.parameters(), lr=params.training_config.lr)
    scheduler_type = None
    if 'scheduler' in params.training_config.optim.keys():
        if params.training_config.optim.scheduler.lower() == 'multilr':
            scheduler_type = 'epoch'
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=params.training_config.optim.scheduler_params.milestones,
                                                             gamma=params.training_config.optim.scheduler_params.gamma)
        elif params.training_config.optim.scheduler.lower() == 'cosineannealinglr':
            scheduler_type = 'step'
            scheduler_params = params.training_config.optim.scheduler_params
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **scheduler_params
            )
        else:
            print('Scheduler only support MultiLR and CosineAnneal now! Using it instead~')
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=params.training_config.optim.scheduler_lr_milestone,
                                                             gamma=params.training_config.optim.scheduler_lr_gamma)

        epoch = int(epoch)
        print(optimizer.param_groups[0]["lr"])
        if scheduler_dict is not None:
            scheduler.load_state_dict(scheduler_dict)
            optimizer.param_groups[0]["lr"] = scheduler_dict["_last_lr"][0]
            print("after load:", optimizer.param_groups[0]["lr"])

    else:
        scheduler = None
        epoch = epoch
    return optimizer, scheduler, scheduler_type, epoch
