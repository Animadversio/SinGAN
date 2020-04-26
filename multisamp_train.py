import torch
import torch.optim as optim
from SinGAN.functions import *
import SinGAN.models as models

from skimage.io import imread, imsave
from skimage.transform import rescale,resize,pyramid_gaussian,pyramid_reduce
from SinGAN.imresize import imresize
from SinGAN.imresize import denorm,imresize_in,torch2uint8_batch
import matplotlib.pylab as plt
from os.path import join
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import json
# Simulate getting option configuration
import argparse
from config import get_arguments
parser = get_arguments()
opt = parser.parse_args(["--scale_factor", "0.75",'--min_size','20'])  # argparse.Namespace(device="")
#%%
opt.input_name = "face_182200.jpg"
opt.mode = "train"
opt = post_config(opt)
# , "--input_name", "face_182200.jpg",
#%%  load the multiple training samples, assume they can be fit into memory
celeba_dir = r"E:\Datasets\celeba-dataset\img_align_celeba\img_align_celeba"
idxs = [8, 20, 21, 131349, 131460, 131509, 131511, 131985]  # b = 8
imgs = []
trc_imgs = []
for imgi in idxs:
    img = imread(join(celeba_dir, r"%06d.jpg" % imgi))
    imgs.append(img)
    x = np2torch(img, opt)  # will move to gpu if cuda specified.
    x = x[:, 0:3, :, :]  # only use 3 channel even more is given.
    trc_imgs.append(x)
real_batch = torch.cat(tuple(trc_imgs))
real_num = len(imgs)
#%% Examine the images
im = torch2uint8_batch(real_batch)
im = imresize_in(im, scale_factor=[0.5, 0.5])
# im = np2torch(im,opt)
plt.imshow(im[:,:,:,0]/255)
plt.show()
#%% Create the pyramid
# real = imresize(real_,opt.scale1,opt)
real_batch = adjust_scales2image(real_batch, opt)
reals_pyr = creat_reals_pyramid(real_batch, [], opt)
# Embed or Encode the images into hidden space.
# Most simply, embed the images into random position (random noise vector)
# advanced version, train an auto encoder! VAE-GAN
#%% Get a initial model
def init_models(opt):
    # opt.min_nfc is the main parameter that controls the width (filter number) for each layer
    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)  # apply weight initialize function of models.
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
#%%
outpath = r"E:\DL_Projects\Vision\SinGAN\TrainedModels\mounts\0"
opt.alpha = 100
opt.Dsteps = 3
opt.Gsteps = 4
opt.niter = 3000
logdir = "Log\\mounts"
writer = SummaryWriter(log_dir=logdir, flush_secs=180)
json.dump(opt.__repr__(), open(join(logdir, "opt.json"),"w"), sort_keys=True, indent=4)
# training iteration
# for epoch in range(opt.niter):
#% Preset training constants for this level
lvl = 5
real = reals_pyr[lvl]
opt.nfc = min(opt.min_nfc_init * pow(2, math.floor(lvl / 4)), 128) # channels !
opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(lvl / 4)), 128) # channels !
opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
m_noise = nn.ZeroPad2d(pad_noise)
m_image = nn.ZeroPad2d(pad_image)
reconloss = nn.MSELoss()

errD2plot = []  # collect the error in D training
errG2plot = []  # collect the error in G training
D_real2plot = []  # collect D_x error for real image
D_fake2plot = []  # Discriminator error for fake image
grad_pen2plot = []
recon2plot = []  # collect reconstruction error

fixed_noise_ = generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device, num_samp=real_num)

netD, netG = init_models(opt)
# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)
#%%
for step in range(opt.niter):
    # 1. Train the Discriminator to output right result
    for j in range(opt.Dsteps):
        netD.zero_grad()
        ## 1.1 Pick some (all) real images feed the discriminator
        output = netD(m_image(real))  # Output of netD, the mean of it is the score!
        # D_real_map = output.detach()
        errD_real = -output.mean()  # Invert the loss to maximize D output for patches in real img.
        errD_real.backward(retain_graph=True)
        #%%# Generate some syn images (number matched) and feed the discriminator
        noise_ = generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device, num_samp=real_num)
        if lvl == 0:
            prev = torch.full([real_num, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
            noise = noise_
        else:
            prev = reals_pyr[lvl - 1]
            prev = imresize(prev, 1 / opt.scale_factor, opt)
            prev = prev[:, :, 0:opt.nzx, 0:opt.nzy]
            noise = opt.noise_amp * noise_ + prev
        fakeout = netG(m_noise(noise), m_image(prev))
        output = netD(fakeout.detach())  # desociate G from the graph, or G will be punished and generate crappy image.
        errD_fake = output.mean()  # decrease the D output for fake.
        errD_fake.backward(retain_graph=True)
        gradient_penalty = calc_gradient_penalty(netD, real, fakeout, opt.lambda_grad, opt.device)
        gradient_penalty.backward()
        errD = errD_real + errD_fake + gradient_penalty
        # Collect loss and back prop
        optimizerD.step()
    errD2plot.append(errD.detach().item())
    D_real2plot.append(errD_real.detach().item())
    D_fake2plot.append(errD_fake.detach().item())
    grad_pen2plot.append(gradient_penalty.detach().item())
    #%% 2. Train the Generator to cheat
    for j in range(opt.Gsteps):
        netG.zero_grad()
        ## 2.1 Generate synthsized samples
        noise_ = generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device, num_samp=real_num)  # can be more samples
        noise = opt.noise_amp * noise_ + prev
        fakeout = netG(m_noise(noise), m_image(prev))
        output = netD(fakeout)  # let error flow to G
        errG = - output.mean()  # increase the D output for fake. Cheat the D
        errG.backward(retain_graph=True)
        ## 2.2 Reconstruct real image from embeded fixed noise vectors
        reconnoise = opt.noise_amp * fixed_noise_ + prev
        reconout = netG(m_noise(reconnoise), m_image(prev))
        err_recon = opt.alpha * reconloss(reconout, real)
        err_recon.backward(retain_graph=True)
        ## back prop
        optimizerG.step()
    errG2plot.append(errG.detach().item())
    recon2plot.append(err_recon.detach().item())
    # 3. meta process, like learning rate scheduling, tensorboard recording
    schedulerD.step()
    schedulerG.step()
    writer.add_scalar('LossD/errD', errD.detach().item(), global_step=step)
    writer.add_scalar('LossD/errG_real', errD_real.detach().item(), global_step=step)
    writer.add_scalar('LossD/errG_fake', errD_fake.detach().item(), global_step=step)
    writer.add_scalar('LossD/grad_pen', gradient_penalty.detach().item(), global_step=step)
    writer.add_scalar('LossG/errG', errG.detach().item(), global_step=step)
    writer.add_scalar('LossG/err_recon', err_recon.detach().item(), global_step=step)
    if step % 25 == 0 or step == (opt.niter - 1):
        print('scale %d:[%d/%d] D: %.2f, real %.2f, fake %.2f; G: %.2f recon %.2f' % (lvl, step, opt.niter, errD, errD_real, errD_fake, errG, err_recon))
    if step % 100 == 0 or step == (opt.niter-1):
        fakeimgs = torch2uint8_batch(fakeout.detach())
        for imgi in range(real_num):
            imsave(join(outpath, "fake%d.jpg"%(imgi)),fakeimgs[:,:,:,imgi])
        reconimgs = torch2uint8_batch(reconout.detach())
        for imgi in range(real_num):
            imsave(join(outpath, "recon%d.jpg" % (imgi)), reconimgs[:, :, :, imgi])
        torch.save(
            {"errD": errD2plot, "errG": errG2plot, "D_real": D_real2plot, "D_fake": D_fake2plot, "recons": recon2plot, "grad_pen": grad_pen2plot}, join(outpath, 'loss_trace.pth'))
        figh = plt.figure()
        plt.plot(errD2plot, label="errD", alpha=0.5)
        plt.plot(errG2plot, label="errG", alpha=0.5)
        plt.plot(D_real2plot, label="Dreal", alpha=0.5)
        plt.plot(D_fake2plot, label="Dfake", alpha=0.5)
        plt.plot(grad_pen2plot, label="grad_pen", alpha=0.5)
        plt.plot(recon2plot, label="recons", alpha=0.5)
        plt.legend()
        plt.savefig(join(outpath, "loss_trace.png"))
        plt.close()
        writer.add_figure('loss/curve', figh, global_step=step)
        writer.add_scalar('optim/lr_D', schedulerD.get_lr()[0], global_step=step)
        writer.add_scalar('optim/lr_G', schedulerG.get_lr()[0], global_step=step)
        # fakemontage = make_grid(denorm(fakeout), nrow=2)
        # reconmontage = make_grid(denorm(reconout), nrow=2)
        writer.add_image('synth/fake', make_grid(denorm(fakeout), nrow=2), global_step=step)
        writer.add_image('synth/reconstruct', make_grid(denorm(reconout), nrow=2), global_step=step)
        writer.add_image('noise', make_grid(denorm(noise), nrow=2), global_step=step)
        writer.add_image('noise_fixed', make_grid(denorm(fixed_noise_), nrow=2), global_step=step)
        # plt.imsave('%s/fake_sample.png' % (opt.outf), convert_image_np(fake.detach()))
        # plt.imsave('%s/G(z_opt).png' % (opt.outf), convert_image_np(netG(Z_opt.detach(), z_prev).detach()))
# Record loss and visualize training progress.


