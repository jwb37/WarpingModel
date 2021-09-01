import torch
from ..BaseModel import BaseModel
from . import networks

from Params import Params

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, visualizer=None):
        super().__init__(visualizer)
        self.lambda_L1 = 100.0

        if hasattr(Params, 'OutputNC'):
            output_nc = Params.OutputNC
        else:
            output_nc = 3

        gpu_ids = [0]

        self.netG = networks.define_G(Params.InputNC, output_nc, 64, 'unet_256', use_dropout=True, gpu_ids=gpu_ids)
        self.netD = networks.define_D(Params.InputNC + output_nc, 64, 'basic', gpu_ids=gpu_ids)
        self.trained_nets.update({
            'G': self.netG,
            'D': self.netD
        })

        # define loss functions
        self.criterionGAN = networks.GANLoss('lsgan').to(Params.Device)
        self.criterionL1 = torch.nn.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = Params.create_optimizer(self.netG.parameters())
        self.optimizer_D = Params.create_optimizer(self.netD.parameters())
        self.optimizers.update({
            'optimG': self.optimizer_G,
            'optimD': self.optimizer_D
        })

        self.losses = dict()


    def __call__(self, realA):
        return self.netG(realA)


    def backward_D(self, realA, realB, fakeB):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((realA, fakeB), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.losses['D_fake'] = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((realA, realB), 1)
        pred_real = self.netD(real_AB)
        self.losses['D_real'] = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.losses['D'] = (self.losses['D_fake'] + self.losses['D_real']) * 0.5
        self.losses['D'].backward()


    def backward_G(self, realA, realB, fakeB):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((realA, fakeB), 1)
        pred_fake = self.netD(fake_AB)
        self.losses['G_GAN'] = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.losses['G_L1']  = self.criterionL1(fakeB, realB) * self.lambda_L1
        # combine loss and calculate gradients
        self.losses['G'] = self.losses['G_GAN'] + self.losses['G_L1']
        self.losses['G'].backward()


    def training_step(self, realA, realB):
        fakeB = self.netG(realA)                # compute fake images: G(A)
        self.losses.clear()
        # update D
        for param in self.netD.parameters():    # enable backprop for D
            param.requires_grad = True
        self.optimizer_D.zero_grad()            # set D's gradients to zero
        self.backward_D(realA, realB, fakeB)    # calculate gradients for D
        self.optimizer_D.step()                 # update D's weights
        # update G
        for param in self.netD.parameters():    # D requires no gradients when optimizing G
            param.requires_grad = False
        self.optimizer_G.zero_grad()            # set G's gradients to zero
        self.backward_G(realA, realB, fakeB)    # calculate gradients for G
        self.optimizer_G.step()                 # update G's weights

        if self.visualizer.save_this_iter:
            self.visualizer.add_tensor('realA', realA)
            self.visualizer.add_tensor('realB', realB)
            self.visualizer.add_tensor('fakeB', fakeB)

        return self.losses
