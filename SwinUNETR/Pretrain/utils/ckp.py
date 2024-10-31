import torch

def filter_load(model_pth):
    model_dict = torch.load(model_pth)
    state_dict = model_dict["state_dict"]
    # fix potential differences in state dict keys from pre-training to
    # fine-tuning
    if "module." in list(state_dict.keys())[0]:
        # print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    if "swin_vit" in list(state_dict.keys())[0]:
        # print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
    return model_dict
    ## TODO extend to handle different patch embedding (input channels)
    ## Access the PatchEmbed module within SwinViT
    #patch_embed_layer = model.swinViT.patch_embed

    ## Create a new convolutional layer with 4 input channels for 3D data
    #new_proj = nn.Conv3d(4, patch_embed_layer.embed_dim, kernel_size=patch_embed_layer.patch_size,
    #                     stride=patch_embed_layer.patch_size)

    ## Initialize the weights for the new channels
    #with torch.no_grad():
    #    # Get the original weights
    #    original_weights = patch_embed_layer.proj.weight.clone()

    #    # Modify only the weights for the additional channels as needed
    #    # For example, re-initialize weights for channels 3 and 4
    #    nn.init.kaiming_normal_(original_weights[:, 2:4, :, :, :], mode='fan_out', nonlinearity='relu')

    #    # Assign the modified weights back to the layer
    #    patch_embed_layer.proj.weight = nn.Parameter(original_weights)

    ## Replace the original proj layer with the new layer
    #patch_embed_layer.proj = new_proj

    ## Load the pre-trained model weights
    #checkpoint = torch.load(pretrained_pth)
    #pretrained_state_dict = checkpoint['state_dict']
    #
    ## Prepare a new state dictionary for SwinUNETR's SwinViT part
    #new_state_dict = {}
    #for k, v in pretrained_state_dict.items():
    #    if k.startswith('module.swinViT.'):
    #        new_key = k.replace('module.swinViT.', '')  # Remove the prefix
    #        # Skip loading weights for the PatchEmbed proj layer
    #        if new_key != 'patch_embed.proj.weight' and new_key != 'patch_embed.proj.bias':
    #            new_state_dict[new_key] = v

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
