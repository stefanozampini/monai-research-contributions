import torch

def filter_load(model_pth):
    model_dict = torch.load(model_pth)
    state_dict = model_dict["state_dict"]
    # fix potential differences in state dict keys from pre-training to
    # fine-tuning
    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
    return model_dict

def save_ckp(task, model, optimizer, scheduler, global_step, model_pth):
    checkpoint = {
        "task" : task,
        "global_step": global_step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(checkpoint, model_pth)

def load_ckp(task, model, optimizer, scheduler, model_pth, model_only=False):
    ckp_dict = filter_load(model_pth)
    ckp_task = ckp_dict.get("task", "pretrain")
    global_step = ckp_dict["global_step"]
    if ckp_task != task:
       model_only = True
       global_step = 0
    model_dict = ckp_dict["state_dict"]
    model.load_state_dict(model_dict, strict=False)
    if model_only:
      return global_step
    if "optimizer" in ckp_dict:
      optimizer_dict = ckp_dict["optimizer"]
      optimizer.load_state_dict(optimizer_dict)
    if "scheduler" in ckp_dict and scheduler is not None:
      scheduler_dict = ckp_dict["scheduler"]
      if scheduler_dict is not None:
         scheduler.load_state_dict(scheduler_dict)
    return global_step

