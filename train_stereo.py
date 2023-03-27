import torch
from torch.utils.data import DataLoader
from models import RGBDepthDataset, ConcatenateRGBDepth, Pix2PixHDStereoModel
from options.train_options import TrainOptions

opt = TrainOptions().parse()

# Set up some training parameters
num_epochs = 100
batch_size = 4
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and dataloader
rgb_dir = "path/to/rgb_images"
depth_dir = "path/to/depth_images"
dataset = RGBDepthDataset(rgb_dir, depth_dir, transform=ConcatenateRGBDepth())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize your model and optimizer
model = Pix2PixHDStereoModel(opt).to(device)
optimizer_G = torch.optim.Adam(model.netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(model.netD.parameters(), lr=lr, betas=(0.5, 0.999))

# Start training loop
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # Load data and move it to the appropriate device
        input_images = batch['concatenated'].to(device)
        
        # Forward pass
        model.zero_grad()
        losses, fake_images = model(input_images)
        loss_G, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake = losses

        # Backward pass for generator
        loss_G_total = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG
        optimizer_G.zero_grad()
        loss_G_total.backward()
        optimizer_G.step()

        # Backward pass for discriminator
        loss_D_total = (loss_D_real + loss_D_fake) * 0.5
        optimizer_D.zero_grad()
        loss_D_total.backward()
        optimizer_D.step()

        # Print training statistics
        if i % 10 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss_G: {loss_G_total.item()}, Loss_D: {loss_D_total.item()}")

    # Save model checkpoint
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"checkpoint_{epoch}.pth")
