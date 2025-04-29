import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
import numpy as np
import os
from tqdm import tqdm
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from PIL import Image
from scipy import linalg
import pyiqa

class Trainer:
    def __init__(self, model, discriminator, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.num_epochs = config.num_epochs
        self.lambda_gp = config.lambda_gp
        self.lambda_tv = config.lambda_tv
        self.lambda_adv = config.lambda_adv
        self.log_dir = config.log_dir
        self.generated_dir = config.generated_dir

        # Optimizers
        self.model_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        self.disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )

        # Learning rate schedulers
        self.model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr
        )
        self.disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.disc_optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr
        )

        # Loss functions
        self.content_criterion = CombinedLoss().to(device)
        self.criterion_pixel = nn.L1Loss().to(device)
        self.criterion_tv = TotalVariationLoss().to(device)
        
        # Gradient scaler for mixed precision training
        self.model_scaler = torch.amp.GradScaler()
        self.disc_scaler = torch.amp.GradScaler()

        # Initialize pyiqa metrics
        self.iqa_metrics = {
            'psnr': pyiqa.create_metric('psnr', device=device),
            'ssim': pyiqa.create_metric('ssim', device=device),
            'lpips': pyiqa.create_metric('lpips', device=device),
            'niqe': pyiqa.create_metric('niqe', device=device)
        }

        # Best model tracking
        self.best_psnr = 0
        self.best_ssim = 0

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.generated_dir, exist_ok=True)

        self.path = f"{self.log_dir}/checkpoint_global.pt"

        print("Model's Total Num Model Parameters: {}".format(sum([param.nelement() for param in self.model.parameters()])))
        model_size = get_model_size(self.model)
        print(f"The model size is {model_size:.2f} MB")



    def compute_metrics(self, deblurred, hr_imgs):
        """Compute PSNR, SSIM, LPIPS, and NIQE scores using pyiqa"""
        with torch.no_grad():
            psnr_score = self.iqa_metrics['psnr'](deblurred, hr_imgs)
            ssim_score = self.iqa_metrics['ssim'](deblurred, hr_imgs)
            lpips_score = self.iqa_metrics['lpips'](deblurred, hr_imgs)
            niqe_score = self.iqa_metrics['niqe'](deblurred)
            
        return (
            psnr_score.mean().item(),
            ssim_score.mean().item(),
            lpips_score.mean().item(),
            niqe_score.mean().item()
        )

    def save_checkpoint(self, path, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'model_optimizer': self.model_optimizer.state_dict(),
            'disc_optimizer': self.disc_optimizer.state_dict(),
            'model_scheduler': self.model_scheduler.state_dict(),
            'disc_scheduler': self.disc_scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim
        }
        torch.save(checkpoint, path)
        if is_best:
            best_path = f"{self.log_dir}/best_model.pth"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path):
        epoch = 0
        if not self.config.resume: return epoch
        if os.path.exists(path): #
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.model_optimizer.load_state_dict(checkpoint['model_optimizer'])
            self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
            self.model_scheduler.load_state_dict(checkpoint['model_scheduler'])
            self.disc_scheduler.load_state_dict(checkpoint['disc_scheduler'])
            self.best_psnr = checkpoint['best_psnr']
            self.best_ssim = checkpoint['best_ssim']
            epoch = checkpoint['epoch']
        return epoch

    def train_step(self, lr_imgs, hr_imgs):
        self.model.train()
        self.discriminator.train()

        with torch.amp.autocast("cuda"):
            sr_imgs = self.model(lr_imgs)
            adv_loss = self.lambda_adv * -torch.mean(self.discriminator(sr_imgs))
            content_loss = self.content_criterion(sr_imgs, hr_imgs)
            total_loss = content_loss + adv_loss

            # Backpropagate and update generator
            self.model_optimizer.zero_grad()
            self.model_scaler.scale(total_loss).backward()
            self.model_scaler.step(self.model_optimizer)
            self.model_scaler.update()

            # Forward propagate through generator
            critic_real = self.discriminator(hr_imgs)
            critic_fake = self.discriminator(sr_imgs.detach())
            tv_loss = self.lambda_tv * self.criterion_tv(sr_imgs.detach()) # hr_imgs -
            gp_loss = self.lambda_gp * gradient_penalty(self.discriminator, hr_imgs, sr_imgs.detach(), device=self.device)
            d_loss = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + gp_loss + tv_loss)
            del critic_real, critic_fake
            torch.cuda.empty_cache()

            # Backpropagate and update discriminator
            self.disc_optimizer.zero_grad()
            self.disc_scaler.scale(d_loss).backward()
            self.disc_scaler.step(self.disc_optimizer)
            self.disc_scaler.update()

        # Compute metrics using pyiqa
        psnr, ssim, lpips, niqe = self.compute_metrics(sr_imgs.detach(), hr_imgs)

        return {
            'gen_loss': total_loss.cpu().item(),
            'content_loss': content_loss.cpu().item(),
            'adv_loss': adv_loss.cpu().item(),
            'd_loss': d_loss.cpu().item(),
            'gp_loss': gp_loss.cpu().item(),
            'tv_loss': tv_loss.cpu().item(),
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips,
            'niqe': niqe
        }

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        self.discriminator.eval()

        val_metrics = []
        saved_images = []

        for i, (lr_imgs, hr_imgs) in enumerate(self.val_loader):
            lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)

            sr_imgs = self.model(blurred)
            psnr, ssim, lpips, niqe = self.compute_metrics(sr_imgs, hr_imgs)
            val_metrics.append((psnr, ssim, lpips, niqe))

            # Save some validation images
            if i < 5:  # Save first 5 validation images
                saved_images.append({
                    'original': hr_imgs[0],
                    'originalx4': sr_imgs[0]
                })

        # Average metrics
        avg_metrics = {
            'val_psnr': np.mean([m[0] for m in val_metrics]),
            'val_ssim': np.mean([m[1] for m in val_metrics]),
            'val_lpips': np.mean([m[2] for m in val_metrics]),
            'val_niqe': np.mean([m[3] for m in val_metrics])
        }

                        # Save validation comparison images to disk
        for idx, images in enumerate(saved_images):
            comparison = torch.cat([
                images['original'],
                images['originalx4']
            ], dim=2).cpu()
            
            # Save to disk
            output_path = f"{self.log_dir}/val_comparison_{epoch}_{idx}.png"
            torchvision.utils.save_image(comparison, output_path)

        # Update best models
        if avg_metrics['val_psnr'] > self.best_psnr:
            self.best_psnr = avg_metrics['val_psnr']
            path = f"{self.log_dir}/checkpoint_valid.pth"
            self.save_checkpoint(path, epoch, is_best=True)

        return avg_metrics

    def train(self, resume_from: str = None):
        # Resume if checkpoint provided
        if self.config.resume and resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
        else:
            start_epoch = self.load_checkpoint(self.path)

        print(f"Resuming from step {self.path} with : (epoch {start_epoch})")

        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                # Training loop
                train_metrics = []
                pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')

                for i, (lr_imgs, hr_imgs) in enumerate(pbar):
                    lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
                    metrics = self.train_step(lr_imgs, hr_imgs)
                    train_metrics.append(metrics)

                    # Update progress bar with current metrics
                    pbar.set_postfix({
                        'psnr': f"{metrics['psnr']:.2f}",
                        'ssim': f"{metrics['ssim']:.4f}",
                        'lpips': f"{metrics['lpips']:.4f}",
                        'loss': f"{metrics['gen_loss']:.4f}"
                    })

                    if (i + 1) % self.config.save_freq == 0:
                        checkpoint_path = f"{self.log_dir}/checkpoint_batch.pt"
                        self.save_checkpoint(checkpoint_path, epoch)
                        print(f"Saved epoch checkpoint to {checkpoint_path}")

                # Average training metrics
                avg_train_metrics = {
                    k: np.mean([m[k] for m in train_metrics])
                    for k in train_metrics[0].keys()
                }

                # Validation
                val_metrics = self.validate(epoch)

                # Update learning rate schedulers once per epoch
                self.model_scheduler.step()
                self.disc_scheduler.step()

                # Log metrics to a CSV file
                log_file = f"{self.log_dir}/training_log.csv"
                is_new_file = not os.path.exists(log_file)
                
                with open(log_file, 'a') as f:
                    # Write header if it's a new file
                    if is_new_file:
                        header = ['epoch', 'lr']
                        header.extend(avg_train_metrics.keys())
                        header.extend(val_metrics.keys())
                        f.write(','.join(header) + '\n')
                    
                    # Write metrics
                    values = [str(epoch), str(self.model_scheduler.get_last_lr()[0])]
                    values.extend([str(v) for v in avg_train_metrics.values()])
                    values.extend([str(v) for v in val_metrics.values()])
                    f.write(','.join(values) + '\n')

                # Print epoch summary with more metrics
                print(f"Epoch {epoch+1}/{self.config.num_epochs} Summary:")
                print(f"  Train - PSNR: {avg_train_metrics['psnr']:.2f} | SSIM: {avg_train_metrics['ssim']:.4f} | LPIPS: {avg_train_metrics['lpips']:.4f}")
                print(f"         Gen Loss: {avg_train_metrics['gen_loss']:.4f} | Disc Loss: {avg_train_metrics['d_loss']:.4f}")
                print(f"  Val   - PSNR: {val_metrics['val_psnr']:.2f} | SSIM: {val_metrics['val_ssim']:.4f} | LPIPS: {val_metrics['val_lpips']:.4f}")
                print(f"  Learning Rate: {self.model_scheduler.get_last_lr()[0]:.6f}")

                # Save epoch checkpoint
                checkpoint_path = f"{self.log_dir}/checkpoint_global.pt"
                self.save_checkpoint(checkpoint_path, epoch)
                print(f"Saved epoch checkpoint to {checkpoint_path}")

                # Save sample images every 250 epochs
                if (epoch+1) % 250 == 0:
                    # Get a batch of validation images
                    lr_imgs, hr_imgs = next(iter(self.val_loader))
                    lr_imgs = lr_imgs.to(self.device)
                    
                    # Generate deblurred images
                    with torch.no_grad():
                        sr_imgs = self.model(lr_imgs)
                    
                    # Save the samples
                    save_samples(lr_imgs, hr_imgs, epoch+1, sample_dir=self.generated_dir)
                    
                    # Free memory
                    del sr_imgs, hr_imgs
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("Training interrupted by user")
            # Save interrupted checkpoint
            checkpoint_path = f"{self.log_dir}/checkpoint_interrupted.pt"
            self.save_checkpoint(checkpoint_path, epoch)
            print(f"Saved interrupt checkpoint to {checkpoint_path}")

        finally:
            # Save final checkpoint
            checkpoint_path = f"{self.log_dir}/checkpoint_final.pt"
            self.save_checkpoint(checkpoint_path, self.num_epochs)
            print(f"Saved final checkpoint to {checkpoint_path}")