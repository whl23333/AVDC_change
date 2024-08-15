from goal_diffusion import GoalGaussianDiffusion, Trainer, ActionDecoder, ConditionModel, DiffusionActionModel, SimpleActionDecoder
from unet import UnetMW as Unet
from unet import NewUnetMW as NewUnet
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import SequentialDatasetv2
from torch.utils.data import Subset
import argparse
from ImgTextPerceiver import ImgTextPerceiverModel, ConvImgTextPerceiverModel, TwoStagePerceiverModel
from torchvision import utils
def main(args):
    valid_n = 1
    sample_per_seq = 8
    target_size = (128, 128)

    if args.mode == 'inference':
        train_set = valid_set = [None] # dummy
    else:
        train_set = SequentialDatasetv2(
            sample_per_seq=sample_per_seq, 
            path="/media/disk3/WHL/flowdiffusion/datasets/metaworld", 
            target_size=target_size,
            randomcrop=True
        )
        valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
        valid_set = Subset(train_set, valid_inds)

    unet = Unet()
    newunet = NewUnet()

    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusion(
        channels=3*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    diffusion3 = GoalGaussianDiffusion(
        channels=3*(sample_per_seq-1),
        model=newunet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    diffusion5 = GoalGaussianDiffusion(
        channels=3*(sample_per_seq-1),
        model=newunet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    implicit_model = ImgTextPerceiverModel(
        num_freq_bands = 6,
        depth = 6,
        max_freq = 10.,
        img_input_channels = 3,
        img_input_axis = 2,
        text_input_channels = 512,
        text_input_axis = 1,
        num_latents = 7,
        latent_dim = 16,
    )
    
    implicit_model3 = ConvImgTextPerceiverModel(
        num_freq_bands = 6,
        depth = 6,
        max_freq = 10.,
        first_img_channels = 3,
        img_input_channels = 64,
        img_input_axis = 2,
        text_input_channels = 512,
        text_input_axis = 1,
        num_latents = 7,
        latent_dim = 16,
    )

    implicit_model7 = TwoStagePerceiverModel(
        num_freq_bands = 6,
        depth = 6,
        max_freq = 10.,
        first_img_channels = 3,
        img_input_channels = 64,
        img_input_axis = 2,
        text_input_channels = 512,
        text_input_axis = 1,
        num_latents = 7,
        latent_dim = 16,
    )



    action_decoder = ActionDecoder(action_dim = 4, hidden_dim = 16)

    action_decoder3 = SimpleActionDecoder(action_dim = 4, hidden_dim = 16)


    condition_model = ConditionModel()

    diffusion_action_model = DiffusionActionModel(
        diffusion,
        implicit_model,
        action_decoder,
        condition_model,
        0.5,
    )

    diffusion_action_model3 = DiffusionActionModel(
        diffusion3,
        implicit_model3,
        action_decoder3,
        condition_model,
        0.5,
    )
    
    diffusion_action_model5 = DiffusionActionModel(
        diffusion,
        implicit_model3,
        action_decoder3,
        condition_model,
        0.0,
    )

    # 冻结部分参数

    # implicit_model3.requires_grad_(False)
    # implicit_model3.eval()

    # action_decoder.requires_grad_(False)
    # action_decoder.eval()

    # action_decoder3.requires_grad_(False)
    # action_decoder3.eval()

    diffusion.requires_grad_(False)
    diffusion.eval()

    # implicit_model7.requires_grad_(False)
    # implicit_model7.eval()

    diffusion_action_model6 = DiffusionActionModel(
        diffusion,
        implicit_model3,
        action_decoder,
        condition_model,
        0.0,
    )

    diffusion_action_model7 = DiffusionActionModel(
        diffusion,
        implicit_model7,
        action_decoder,
        condition_model,
        1.0,
    )


    # trainer = Trainer(
    #     diffusion_action_model=diffusion_action_model6,
    #     tokenizer=tokenizer, 
    #     text_encoder=text_encoder,
    #     train_set=train_set,
    #     valid_set=valid_set,
    #     train_lr=1e-4,
    #     train_num_steps = 240000,
    #     save_and_sample_every = 2000,
    #     ema_update_every = 10,
    #     ema_decay = 0.999,
    #     train_batch_size = 3,
    #     valid_batch_size =32,
    #     gradient_accumulate_every = 1,
    #     num_samples=valid_n, 
    #     results_folder ='../results6/mw',
    #     fp16 =True,
    #     amp=True,
    # )

    trainer = Trainer(
        diffusion_action_model=diffusion_action_model7,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=1e-4,
        train_num_steps = 240000,
        save_and_sample_every = 10000,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size = 4,
        valid_batch_size =32,
        gradient_accumulate_every = 1,
        num_samples=valid_n, 
        results_folder ='../result10/mw',
        fp16 =True,
        amp=True,
    )

    # trainer = Trainer(
    #     diffusion_action_model=diffusion_action_model7,
    #     tokenizer=tokenizer, 
    #     text_encoder=text_encoder,
    #     train_set=train_set,
    #     valid_set=valid_set,
    #     train_lr=1e-4,
    #     train_num_steps = 240000,
    #     save_and_sample_every = 10000,
    #     ema_update_every = 10,
    #     ema_decay = 0.999,
    #     train_batch_size = 3,
    #     valid_batch_size =32,
    #     gradient_accumulate_every = 1,
    #     num_samples=valid_n, 
    #     results_folder ='../results8/mw',
    #     fp16 =True,
    #     amp=True,
    # )

    if args.checkpoint_num is not None:
        trainer.load_resume(args.checkpoint_num) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    if args.mode == 'train':
        trainer.train()
    else:
        from PIL import Image
        from torchvision import transforms
        import imageio
        import torch
        from os.path import splitext
        text = args.text
        guidance_weight = args.guidance_weight
        image = Image.open(args.inference_path)
        batch_size = 1
        ### 231130 fixed center crop issue 
        transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
        image = transform(image)
        output = trainer.sample(image.unsqueeze(0), [text], batch_size, guidance_weight).cpu()
        output = output[0].reshape(-1, 3, *target_size)
        output = torch.cat([image.unsqueeze(0), output], dim=0)
        root, ext = splitext(args.inference_path)
        output_gif = root + '_out.gif'
        output_png = root + '_out.png'
        utils.save_image(output, output_png, nrow=8)
        output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
        imageio.mimsave(output_gif, output, duration=200, loop=1000)
        print(f'Generated {output_gif}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'inference']) # set to 'inference' to generate samples
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None) # set to checkpoint number to resume training or generate samples
    parser.add_argument('-p', '--inference_path', type=str, default=None) # set to path to generate samples
    parser.add_argument('-t', '--text', type=str, default=None) # set to text to generate samples
    parser.add_argument('-n', '--sample_steps', type=int, default=100) # set to number of steps to sample
    parser.add_argument('-g', '--guidance_weight', type=int, default=0) # set to positive to use guidance
    args = parser.parse_args()
    if args.mode == 'inference':
        assert args.checkpoint_num is not None
        assert args.inference_path is not None
        assert args.text is not None
        assert args.sample_steps <= 100
    main(args)