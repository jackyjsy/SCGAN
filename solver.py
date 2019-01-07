import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import Generator
from model import Generator_SC
# from model import Generator_SC_2
from model import Generator_SC_3
from model import Discriminator
from model import Generator_CNN
from model import Discriminator_CNN
from model import Segmentor
from PIL import Image
from util.visualizer import Visualizer
import util.util as util
from collections import OrderedDict

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    # print(y)
    # print(y.size())
    y=np.asarray(y)
    # print(type(y))
    y=np.eye(num_classes, dtype='uint8')[y]
    return y

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        # print(targets.size())
        return self.nll_loss(F.log_softmax(inputs), torch.squeeze(targets))
    
class Solver(object):
    def __init__(self, celebA_loader, config):
        # Data loader
        self.celebA_loader = celebA_loader
        self.visualizer = Visualizer()
        # Model hyper-parameters
        self.z_dim = config.z_dim
        self.c_dim = config.c_dim
        self.s_dim = config.s_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_gp = config.lambda_gp
        self.lambda_s = config.lambda_s
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.a_lr = config.a_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Criterion
        self.criterion_s = CrossEntropyLoss2d(size_average=True).cuda()
        
        # Training settings
        self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model

        # Test settings
        self.test_model = config.test_model
        self.config = config

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.result_path = config.result_path

        # Step size
        self.log_step = config.log_step
        self.visual_step = self.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define a generator and a discriminator
        
        if self.config.mode == "train":
            self.D = Discriminator_CNN(self.c_dim) 
            self.A = Segmentor()
            self.G = Generator_SC_3(self.z_dim, self.c_dim, self.s_dim)
            # Optimizers
            self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
            # self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])
            self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
            self.a_optimizer = torch.optim.Adam(self.A.parameters(), self.a_lr, [self.beta1, self.beta2])
        elif self.config.mode == "seg":
            self.A = Segmentor()
            # Print networks
        else:
            self.G = Generator_SC_3(self.z_dim, self.c_dim, self.s_dim)
        # self.print_network(self.G, 'G')
        if self.config.mode == "train":
            self.print_network(self.G, 'G')
            self.print_network(self.D, 'D')
            self.print_network(self.A, 'A')
        if torch.cuda.is_available() and self.config.cuda:
            
            if self.config.mode == "train":
                self.G.cuda()
                self.D.cuda()
                self.A.cuda()
            elif self.config.mode == "seg":
                self.A.cuda()
            else:
                self.G.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        
        if self.config.mode == "train":
            self.G.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
            self.D.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
            self.A.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_A.pth'.format(self.pretrained_model))))
        elif self.config.mode == "seg":
            self.A.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_A.pth'.format(self.pretrained_model))))
        else:
            self.G.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr, a_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.a_optimizer.param_groups:
            param_group['lr'] = a_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.a_optimizer.zero_grad()
        
    def to_var(self, x, volatile=False):
        if torch.cuda.is_available() and self.config.cuda:
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x

    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out
        
    def make_celeb_labels_test(self):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []
        # fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,1,1]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,1,1]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,1,1]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,1,0]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,1,0]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,1,0]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,0,1]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,0,1]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,0,1]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,0,0]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,0,0]).unsqueeze(0), volatile=True))
        # fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,0,0]).unsqueeze(0), volatile=True))

        fixed_c_list.append(torch.FloatTensor([1,0,0,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,0,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,0,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,0,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,0,0]).unsqueeze(0))

        return fixed_c_list
    
    def make_celeb_labels_all(self):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []

        fixed_c_list.append(torch.FloatTensor([1,0,0,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,1,0,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,1,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,1,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,0,1,1]).unsqueeze(0))
        
        fixed_c_list.append(torch.FloatTensor([1,0,0,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,1,0,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,1,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,1,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,0,1,0]).unsqueeze(0))
        
        fixed_c_list.append(torch.FloatTensor([1,0,0,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,1,0,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,1,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,1,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,0,0,1]).unsqueeze(0))
        
        
        
        fixed_c_list.append(torch.FloatTensor([1,0,0,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,1,0,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,1,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,1,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,0,0,0]).unsqueeze(0))

        return fixed_c_list

    def train(self):
        """Train StarGAN within a single dataset."""

        # Set dataloader
        if self.dataset == 'CelebA':
            self.data_loader = self.celebA_loader
        else:
            self.data_loader = self.rafd_loader

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        # Fixed latent vector and label for output samples
        fixed_size = 5
        fixed_s_size = 2
        fixed_z = torch.randn(fixed_size, self.z_dim)
        fixed_z = self.to_var(fixed_z, volatile=True)

        fixed_c_list = self.make_celeb_labels_test()

        fixed_z_repeat = fixed_z.repeat(len(fixed_c_list) * fixed_s_size,1)
        fixed_c_repeat_list = []
        for fixed_c in fixed_c_list:
            fixed_c_repeat_list.append(fixed_c.expand(fixed_size,fixed_c.size(1)))
        
        fixed_c_repeat = torch.cat(fixed_c_repeat_list, dim=0)
        fixed_c_repeat = torch.cat([fixed_c_repeat,fixed_c_repeat], dim=0)
        fixed_c_repeat = self.to_var(fixed_c_repeat, volatile=True)
        
        fixed_s = []
        for i, (images, seg_i, seg, labels) in enumerate(self.data_loader):
            print(seg_i.size())
            print(seg.size())
            fixed_s.append(seg[0].unsqueeze(0).expand(fixed_size * len(fixed_c_list), seg.size(1), 
                                                      seg.size(2), seg.size(3)))
            fixed_s.append(seg[1].unsqueeze(0).expand(fixed_size * len(fixed_c_list), seg.size(1), 
                                                      seg.size(2), seg.size(3)))
            break
        fixed_s = torch.cat(fixed_s, dim=0) 
        fixed_s = self.to_var(fixed_s, volatile=True)
        fixed_c_list = []
        fixed_c_repeat_list = []
        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr
        a_lr = self.a_lr
        
        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])-1
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            epoch_iter = 0
            for i, (real_x, real_s_i, real_s, real_label) in enumerate(self.data_loader):
                epoch_iter = epoch_iter + 1

                real_c = real_label.clone()
                rand_idx = torch.randperm(real_s.size(0))
                fake_s = real_s[rand_idx]
                fake_s_i = real_s_i[rand_idx]
                # Latent vector z
                z = torch.randn(real_x.size(0), self.z_dim)
                z = self.to_var(z)
                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_c = self.to_var(real_c)  
                real_s = self.to_var(real_s) 
                real_s_i = self.to_var(real_s_i)  
                fake_s = self.to_var(fake_s) 
                fake_s_i = self.to_var(fake_s_i)  

                # ================== Train D ================== #

                # Compute loss with real images
                out_src, out_cls = self.D(real_x)
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = F.binary_cross_entropy_with_logits(
                    out_cls, real_c, size_average=False) / real_x.size(0)

                # # Compute classification accuracy of the discriminator
                # if (i+1) % self.log_step == 0:
                #     accuracies = self.compute_accuracy(out_cls, real_c, self.dataset)
                #     log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                #     if self.dataset == 'CelebA':
                #         print('Classification Acc (Black/Blond/Brown/Gender/Aged): ', end='')
                #     else:
                #         print('Classification Acc (8 emotional expressions): ', end='')
                #     print(log)

                # Compute loss with fake images
                fake_x = self.G(z, real_c, fake_s)
                fake_x = Variable(fake_x.data)
                out_src, out_cls = self.D(fake_x)
                d_loss_fake = torch.mean(out_src)

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out, out_cls = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
                # ================== Train A ================== #
                self.a_optimizer.zero_grad()
                out_real_s = self.A(real_x)
                a_loss_real = self.criterion_s(out_real_s, real_s_i) * self.lambda_s
                # out_fake_s = self.A(fake_x)
                # a_loss_fake = self.criterion_s(out_fake_s, fake_s_i) * self.lambda_s

                a_loss = a_loss_real# + a_loss_fake
                a_loss.backward()
                self.a_optimizer.step()
                
                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.data[0]
                loss['D/loss_fake'] = d_loss_fake.data[0]
                loss['D/loss_cls'] = d_loss_cls.data[0]
                loss['D/loss_gp'] = d_loss_gp.data[0]

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(z, real_c, fake_s)
                    # fake_x2 = self.G(z, fake_c)
                    # Compute losses
                    out_src, out_cls = self.D(fake_x)
                    g_loss_fake = - torch.mean(out_src)
                    
                    g_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls, real_c, size_average=False) / fake_x.size(0)
                    
                    # segmentation loss
                    out_fake_s = self.A(fake_x)
                    g_loss_s = self.lambda_s * self.criterion_s(out_fake_s, fake_s_i)
                    
                    # Backward + Optimize
                    g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + g_loss_s
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake.data[0]
                    loss['G/loss_cls'] = g_loss_cls.data[0]
                
                if (i+1) % self.visual_step == 0:
                    # save visuals
                    self.real_x = real_x
                    self.fake_x = fake_x
                    self.real_s = real_s
                    self.fake_s = fake_s
                    # self.fake_x2 = fake_x2
                    # self.out_real_s = out_real_s
                    # self.out_fake_s = out_fake_s
                    # save losses
                    self.d_real = - d_loss_real
                    self.d_fake = d_loss_fake
                    self.d_loss = d_loss
                    self.g_loss = g_loss
                    self.g_loss_fake = g_loss_fake
                    self.g_loss_cls = self.lambda_cls * g_loss_cls
                    self.g_loss_s = g_loss_s
                    errors_D = self.get_current_errors('D')
                    errors_G = self.get_current_errors('G')
                    self.visualizer.display_current_results(self.get_current_visuals(), e)
                    self.visualizer.plot_current_errors_D(e, float(epoch_iter)/float(iters_per_epoch), errors_D)
                    self.visualizer.plot_current_errors_G(e, float(epoch_iter)/float(iters_per_epoch), errors_G)
                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

#                 # Translate fixed images for debugging
#                 if (i+1) % self.sample_step == 0:
# #                     fake_image_list = []
# #                     for fixed_c in fixed_c_list:
# #                         fixed_c = fixed_c.expand(fixed_z.size(0), fixed_c.size(1))
# #                         fake_image_list.append(self.G(fixed_z, fixed_c))
                    
                        
# #                     fake_images = torch.cat(fake_image_list, dim=3)
# #                     save_image(self.denorm(fake_images.data),
# #                         os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
# #                     print('Translated images and saved into {}..!'.format(self.sample_path))

#                     fake_images_repeat = self.G(fixed_z_repeat, fixed_c_repeat, fixed_s).data.cpu()
#                     fake_image_list = []
#                     for idx in range(24):
#                         fake_image_list.append(fake_images_repeat[fixed_size*(idx):fixed_size*(idx+1)])
#                     fake_images = torch.cat(fake_image_list, dim=3)
#                     save_image(self.denorm(fake_images),
#                         os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
#                     print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))
                    torch.save(self.A.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_A.pth'.format(e+1, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                a_lr -= (self.a_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr, a_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}, a_lr: {}.'.format(g_lr, d_lr, a_lr))
      
    # def test(self):
    #     test_size = 10
    #     test_c_list = self.make_celeb_labels_test()
    #     test_z = self.to_var(torch.randn(test_size, self.z_dim))
    #     fake_image_list = []
    #     for test_c in test_c_list:
    #         test_c = test_c.expand(test_z.size(0), test_c.size(1))
    #         fake_image_list.append(self.G(test_z, test_c))
    #         print(test_c)
    #     fake_images = torch.cat(fake_image_list, dim=3)
    #     save_image(self.denorm(fake_images.data),
    #                     os.path.join(self.result_path, 'fake.png'),nrow=1, padding=0)
    def test(self):
        # Load trained parameters
        # G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        # self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])
        fixed_c_list = self.make_celeb_labels_test()
        test_size = 10
        # for idx in range(test_size):
        for idx in range(0,100):
            test_z = self.to_var(torch.randn(1, self.z_dim), volatile=True)
            fake_image_mat = []
            for fixed_c in fixed_c_list:
                fake_image_list = []
                for i in range(11):
                    seg = Image.open(os.path.join(self.config.test_seg_path, '{}.png'.format(i+1)))
                    seg = transform_seg1(seg)
                    num_s = 7
                    seg_onehot = to_categorical(seg, num_s)
                    seg_onehot = transform_seg2(seg_onehot)*255.0
                    seg_onehot = seg_onehot.unsqueeze(0)

                    s = self.to_var(seg_onehot, volatile=True)

                    fake_x = self.G(test_z,self.to_var(fixed_c, volatile=True),s)
                    fake_image_list.append(fake_x)
                    # save_path = os.path.join(self.result_path, 'fake_x_{}.png'.format(i+1))
                    # save_image(self.denorm(fake_x.data), save_path, nrow=1, padding=0)
                fake_images = torch.cat(fake_image_list, dim=3)
                fake_image_mat.append(fake_images)

            fake_images_save = torch.cat(fake_image_mat, dim=2)
            
            save_path = os.path.join(self.result_path, 'fake_x_sum_{}.png'.format(idx))
            print('Translated test images and saved into "{}"..!'.format(save_path))
            save_image(self.denorm(fake_images_save.data), save_path, nrow=1, padding=0) 
    def test_celeba_single(self):
        image_index = 0
        import math
        test_size = math.ceil(50000/28)
        c_dim = 28
        test_c = self.make_celeb_labels_all()
        test_c = self.to_var(torch.cat(test_c,dim=0), volatile=True)
        for i, (real_x, real_s_i, real_s, real_label) in enumerate(self.celebA_loader):
            real_s = self.to_var(real_s, volatile=True)

            test_z = self.to_var(torch.randn(c_dim, self.z_dim), volatile=True)
            fake_image_list = self.G(test_z, test_c, real_s)
            for ind in range(fake_image_list.size(0)):
                save_image(self.denorm(fake_image_list[ind].data),
                        os.path.join(self.result_path, 'single/fake_{0:05d}.png'.format(image_index)),nrow=1, padding=0)
                image_index = image_index + 1
            if i > test_size-1:
                break
    def test_celeba_epoch(self):
        # Load trained parameters
        test_size=20
        c_dim = 17
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])
        real_c = self.to_var(torch.FloatTensor([0,0,1,0,1]).unsqueeze(0).expand(test_size,5), volatile=True)

        test_z = self.to_var(torch.randn(test_size, self.z_dim), volatile=True)
        
        seg = Image.open(os.path.join(self.config.test_seg_path, '11.png'))
        seg = transform_seg1(seg)
        num_s = 7
        seg_onehot = to_categorical(seg, num_s)
        seg_onehot = transform_seg2(seg_onehot)*255.0
        seg_onehot = seg_onehot.unsqueeze(0)
        real_s = self.to_var(seg_onehot.expand(test_size,seg_onehot.size(1),seg_onehot.size(2),seg_onehot.size(3)), volatile=True)
            
        fake_x_mat = [] 
        for epoch in range(40):
            self.G.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_9000_G.pth'.format(epoch+1))))
            print('Load model {}.'.format(epoch+1))
            self.G.eval()     

             
            fake_x_array = self.G(test_z,real_c,real_s)
            fake_x_mat.append(fake_x_array)
        fake_x_mat = torch.cat(fake_x_mat, dim=3)
        save_path = os.path.join(self.result_path, 'fake_x_epoch.png')

        save_image(self.denorm(fake_x_mat.data), save_path, nrow=1, padding=0)  
        print('Translated test images and saved into "{}"..!'.format(save_path))

    def test_doseg(self):
        self.A.eval()
        num_s = 7
        transform = transforms.Compose([
            # transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])
        accuracy = []
        num_element = 128*128
        for root, _, fnames in sorted(os.walk(self.config.test_seg_path)):
            fnames = sorted(fnames)
            # rand_idx = torch.randperm(len(fnames))
            # fnames_shuffle = fnames[rand_idx]
            import random
            fnames_shuffle = fnames[:]
            # random.shuffle(fnames_shuffle)
            # print(fnames)
            # print(fnames_shuffle)
            for i in range(len(fnames)):
            # for fname in sorted(fnames):
                fname = fnames[i]
                fname_shuffle = fnames_shuffle[i]
                path = os.path.join(self.config.test_img_path, fname[:-4]+'_.png')
                img = Image.open(path)
                img = transform(img).unsqueeze(0)
                img = self.to_var(img, volatile=True)  
                path = os.path.join(self.config.test_seg_path, fname_shuffle[:-3]+'png')
                seg = Image.open(path)
                seg = transform_seg1(seg)
                seg=np.asarray(seg,dtype=np.long)
                seg = torch.LongTensor(seg).unsqueeze(0).unsqueeze(1)
                seg = self.to_var(seg, volatile=True)  
                seg_est = self.A(img)
                # print(seg)
                # print(img)
                # print(seg_est)
                seg_max_num, seg_est_index = torch.max(seg_est, dim=1)
                # print(seg.size())
                # print(seg_est_index.size())
                accuracy_one = torch.sum(torch.eq(seg_est_index.squeeze(), seg.squeeze()).long()).float()/num_element
                accuracy_one = accuracy_one.squeeze()
                print(accuracy_one)
                accuracy.append(accuracy_one.data[0])
                # save_path = os.path.join(self.config.test_seg_path, fname[:-4]+'_est.jpg')
                # seg_est_index = seg_est_index.float().unsqueeze(0)
                # save_image((seg_est_index/torch.max(seg_est_index)).data, save_path, nrow=1, padding=0)  
        print(accuracy)
        print(sum(accuracy) / len(accuracy))
    def test_seg(self):
        # Load trained parameters
        self.G.eval()
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])
        fixed_c_list = self.make_celeb_labels_test()

        test_z = self.to_var(torch.randn(1, self.z_dim), volatile=True)  
        fake_image_mat = []
        fixed_c = fixed_c_list[7]

        for root, _, fnames in sorted(os.walk(self.config.test_seg_path)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                seg = Image.open(path)
                seg = transform_seg1(seg)
                num_s = 7
                seg_onehot = to_categorical(seg, num_s)
                seg_onehot = transform_seg2(seg_onehot)*255.0
                seg_onehot = seg_onehot.unsqueeze(0)
                s = self.to_var(seg_onehot, volatile=True)
                fake_x = self.G(test_z,self.to_var(fixed_c, volatile=True),s)
                save_path = os.path.join(self.result_path, fname[:-3]+'jpg')
                print('Translated test images and saved into "{}"..!'.format(save_path))
                save_image(self.denorm(fake_x.data), save_path, nrow=1, padding=0)

    # def test_seg(self):
    #     # Load trained parameters
    #     self.G.eval()
    #     transform_seg1 = transforms.Compose([
    #         transforms.CenterCrop(self.config.celebA_crop_size),
    #         transforms.Scale(self.config.image_size)])
    #     transform_seg2 = transforms.Compose([
    #         transforms.ToTensor()])
    #     fixed_c_list = self.make_celeb_labels_test()

    #     for idx in range(0,30):
    #         test_z = self.to_var(torch.randn(1, self.z_dim), volatile=True)  
    #         fake_image_mat = []
    #         for fixed_c in fixed_c_list:
    #             fake_image_list = []
    #             # for root, _, fnaend(fake_images)

    #         fake_images_save = torch.cat(fake_image_mat, dim=2)
            
    #         save_path = os.path.join(self.result_path, 'fake_x_sum_{}.png'.format(idx))
    #         print('Translated test images and saved into "{}"..!'.format(save_path))
    #         save_image(self.denorm(fake_images_save.data), save_path, nrow=1, padding=0)hot = seg_onehot.unsqueeze(0)
    #                     s = self.to_var(seg_onehot, volatile=True)
    #                     fake_x = self.G(test_z,self.to_var(fixed_c, volatile=True),s)
    #                     fake_image_list.append(fake_x)
    #             fake_images = torch.cat(fake_image_list, dim=3)
    #             fake_image_mat.append(fake_images)

    #         fake_images_save = torch.cat(fake_image_mat, dim=2)
            
    #         save_path = os.path.join(self.result_path, 'fake_x_sum_{}.png'.format(idx))
    #         print('Translated test images and saved into "{}"..!'.format(save_path))
    #         save_image(self.denorm(fake_images_save.data), save_path, nrow=1, padding=0)

    def test_interp(self):
        # Load trained parameters
        # G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        # self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])
        fixed_c_list = self.make_celeb_labels_test()
        interp_size = 10
        for idx_z in range(100):
            if not os.path.exists(os.path.join(self.result_path, '{}'.format(idx_z))):
                os.makedirs(os.path.join(self.result_path, '{}'.format(idx_z)))
            test_z1 = torch.randn(1, self.z_dim)
            test_z2 = torch.randn(1, self.z_dim)
            test_z_step = (test_z2 - test_z1)/(interp_size - 1)
            test_z_array = []
            for idx in range(interp_size):
                test_z_array.append(self.to_var((test_z1 + test_z_step * idx), volatile=True))
            # test_z = torch.cat(test_z_array, dim=0)
            for idx, fixed_c in enumerate(fixed_c_list):
                fake_image_mat = []
                for z_idx in range(0,10):
                    test_z = test_z_array[z_idx]
                    fake_image_list = []
                    for i in range(11):
                        seg = Image.open(os.path.join(self.config.test_seg_path, '{}.png'.format(i+1)))
                        seg = transform_seg1(seg)
                        num_s = 7
                        seg_onehot = to_categorical(seg, num_s)
                        seg_onehot = transform_seg2(seg_onehot)*255.0
                        seg_onehot = seg_onehot.unsqueeze(0)

                        s = self.to_var(seg_onehot, volatile=True)

                        fake_x = self.G(test_z,self.to_var(fixed_c, volatile=True),s)
                        fake_image_list.append(fake_x)
                    fake_images = torch.cat(fake_image_list, dim=3)
                    fake_image_mat.append(fake_images)

                fake_images_save = torch.cat(fake_image_mat, dim=2)
                # os.makedirs(os.path.join(self.result_path, '{}'.format(idx_z)))
                save_path = os.path.join(self.result_path, '{}'.format(idx_z), 'fake_x_sum_{}.png'.format(idx))
                print('Translated test images and saved into "{}"..!'.format(save_path))
                save_image(self.denorm(fake_images_save.data), save_path, nrow=1, padding=0)  

    def test_interp_all(self):
        self.G.eval()
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])
        fixed_c_list = self.make_celeb_labels_test()
        
        num_s = 7
        interp_size = 10
        for idx_z in range(100):
            test_z1 = torch.randn(1, self.z_dim)
            test_z2 = torch.randn(1, self.z_dim)
            test_z_step = (test_z2 - test_z1)/(interp_size - 1)
            test_z_array = []
            rand_idx = torch.randperm(len(fixed_c_list))
            test_c1 = fixed_c_list[rand_idx[0]]
            test_c2 = fixed_c_list[rand_idx[1]]
            test_c_step = (test_c2 - test_c1)/(interp_size - 1)
            test_c_array = []

            test_s_array = []

            for idx in range(interp_size):
                test_z_array.append(test_z1 + test_z_step * idx)
                test_c_array.append(test_c1 + test_c_step * idx)
                seg = Image.open(os.path.join(self.config.test_seg_path, '{}.png'.format(idx+1)))
                seg = transform_seg1(seg)
                
                seg_onehot = to_categorical(seg, num_s)
                seg_onehot = transform_seg2(seg_onehot)*255.0
                seg_onehot = seg_onehot.unsqueeze(0)
                test_s_array.append(seg_onehot)
            test_z = self.to_var(torch.cat(test_z_array, dim=0), volatile=True)
            test_c = self.to_var(torch.cat(test_c_array, dim=0), volatile=True)
            test_s = self.to_var(torch.cat(test_s_array, dim=0), volatile=True)
            fake_x = self.G(test_z,test_c,test_s)
            save_path = os.path.join(self.result_path, 'fake_x_sum_{}.png'.format(idx_z))
            save_image(self.denorm(fake_x.data), save_path, nrow=interp_size, padding=0)  
            print('Translated test images and saved into "{}"..!'.format(save_path))


    def get_current_errors(self, label='all'):
        D_fake = self.d_fake.data[0]
        D_real = self.d_real.data[0]
        D_loss = self.d_loss.data[0]
        G_loss = self.g_loss.data[0]
        G_loss_s = self.g_loss_s.data[0]
        G_loss_cls = self.g_loss_cls.data[0]
        G_loss_fake = self.g_loss_fake.data[0]
        if label == 'all':
            return OrderedDict([('D_fake', D_fake), 
                                ('D_real', D_real), 
                                ('D_loss', D_loss),
                                ('G_loss', G_loss), 
                                ('G_loss_fake', G_loss_fake)])
        if label == 'D':
            return OrderedDict([('D_fake', D_fake), 
                                ('D_real', D_real), 
                                ('D_loss', D_loss)])
        if label == 'G':
            return OrderedDict([('G_loss', G_loss), 
                                ('G_loss_cls', G_loss_cls), 
                                ('G_loss_s', G_loss_s), 
                                ('G_loss_fake', G_loss_fake)])

    def get_current_visuals(self):
        real_x = util.tensor2im(self.real_x.data)
        fake_x = util.tensor2im(self.fake_x.data)
        real_s = util.tensor2im_seg(self.real_s.data)
        fake_s = util.tensor2im_seg(self.fake_s.data)
        # fake_x2 = util.tensor2im(self.fake_x2.data)
        return OrderedDict([('real_x', real_x), 
                            ('fake_x', fake_x), 
                            ('real_s', self.cat2class(real_s)),
                            ('fake_s', self.cat2class(fake_s)), 
                            # ('fake_x2', fake_x2), 
                            ])
    
    # def get_current_visuals(self):
    #     real_x = util.tensor2im(self.real_x.data)
    #     fake_x = util.tensor2im(self.fake_x.data)
    #     rec_x = util.tensor2im(self.rec_x.data)
    #     real_s = util.tensor2im_seg(self.real_s.data)
    #     fake_s = util.tensor2im_seg(self.fake_s.data)
    #     out_real_s = util.tensor2im_seg(self.out_real_s.data)
    #     out_fake_s = util.tensor2im_seg(self.out_fake_s.data)
    #     return OrderedDict([('real_x', real_x), 
    #                         ('fake_x', fake_x), 
    #                         ('rec_x', rec_x), 
    #                         ('real_s', self.cat2class(real_s)),
    #                         ('fake_s', self.cat2class(fake_s)),
    #                         ('out_real_s', self.cat2class(out_real_s)), 
    #                         ('out_fake_s', self.cat2class(out_fake_s))
    #                         ])
    def cat2class(self, m):
        y = np.zeros((np.size(m,0),np.size(m,1)),dtype='float64')
        for i in range(np.size(m,2)):
            y = y + m[:,:,i]*i
        y = y / float(np.max(y)) * 255.0 
        y = y.astype(np.uint8)
        y = np.reshape(y,(np.size(m,0),np.size(m,1),1))
        # print(np.shape(y))
        return np.repeat(y, 3, 2)
            

