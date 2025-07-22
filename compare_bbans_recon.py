import torch
import numpy as np
import util
import rans
from torch_vae.tvae_beta_binomial import BetaBinomialVAE
from torch_vae import tvae_utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- Load model ---
latent_dim = 50
latent_shape = (1, latent_dim)
model = BetaBinomialVAE(hidden_dim=200, latent_dim=latent_dim)
model.load_state_dict(
    torch.load('torch_vae/saved_params/torch_vae_beta_binomial_params', map_location='cpu'))
model.eval()

rec_net = tvae_utils.torch_fun_to_numpy_fun(model.encode)
gen_net = tvae_utils.torch_fun_to_numpy_fun(model.decode)

prior_precision = 8
obs_precision = 14
q_precision = 14

obs_append = tvae_utils.beta_binomial_obs_append(255, obs_precision)
obs_pop = tvae_utils.beta_binomial_obs_pop(255, obs_precision)

vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append, prior_precision, q_precision)
vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop, prior_precision, q_precision)

# --- Load MNIST images ---
num_images = 8
mnist = datasets.MNIST('data/mnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
images = mnist.test_data[:num_images].float().view(num_images, 1, -1)

# --- Encode images ---
rng = np.random.RandomState(0)
other_bits = rng.randint(low=1 << 16, high=1 << 31, size=20, dtype=np.uint32)
state = rans.unflatten(other_bits)

for image in images:
    state = vae_append(state, image)

compressed_message = rans.flatten(state)

# --- Decode images ---
state = rans.unflatten(compressed_message)
reconstructed_images = []
for n in range(num_images):
    state, image_ = vae_pop(state)
    reconstructed_images.append(np.array(image_).reshape(28, 28))

# --- Plot original vs reconstructed ---
fig, axes = plt.subplots(num_images, 2, figsize=(4, num_images*2))
for i in range(num_images):
    axes[i, 0].imshow(images[i].numpy().reshape(28, 28), cmap='gray')
    axes[i, 0].set_title('Original')
    axes[i, 0].axis('off')
    axes[i, 1].imshow(reconstructed_images[num_images-i-1], cmap='gray')
    axes[i, 1].set_title('Reconstructed')
    axes[i, 1].axis('off')
plt.tight_layout()
plt.show()