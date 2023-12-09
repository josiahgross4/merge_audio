import torch
import numpy as np
from torch import nn
import scipy.io.wavfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE(nn.Module):
    def __init__(self, hidden_dim=512, latent_dim=128):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(1323000, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mu and log_var
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1323000),
            nn.Sigmoid()
        )


model = VAE().to(device)

model.load_state_dict(torch.load('VAE_model.pth'))


def generate_audio(model, num_samples, latent_dim):
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)

        generated_samples = model.decoder(z)
        generated_samples = generated_samples.view(num_samples, -1)

        return generated_samples

generated_samples = generate_audio(model, 1, 128)

generated_samples = generated_samples.cpu().numpy()
# Normalize the data to the range -1 to 1
max_val = np.max(np.abs(generated_samples))
generated_samples = generated_samples / max_val

# Scale the data to 32-bit range, scipy.io.wavfile.write can handle this format as well
generated_samples = np.int32(generated_samples * 2147483647)

scipy.io.wavfile.write('aisample2.wav', 44100, generated_samples)

