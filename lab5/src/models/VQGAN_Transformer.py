import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#[x] TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##[x] TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        #z_q: quantized latent, codebook_indices: the index of the codebook vector that each latent vector is closest to
        z_q, z_indices, _ = self.vqgan.encode(x) 
        z_indices = z_indices.reshape(z_q.shape[0], -1)
        
        return z_q, z_indices
    
## [x] TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda ratio: 1 - ratio
        elif mode == "cosine":
            return (lambda ratio: np.cos(ratio * np.pi / 2))
        elif mode == "square":
            return lambda ratio: 1 - ratio ** 2
        else:
            raise NotImplementedError

##[x] TODO2 step1-3:            
    def forward(self, x):
        # encode the input image to latent and quantized latent
        _, z_indices = self.encode_to_z(x)
        
        # In training, the mask ratio is randomly sampled
        mask_ratio = np.random.uniform(0, 1)
        mask = torch.rand_like(z_indices.float()) < mask_ratio
        
        # Mask the tokens
        masked_indices = z_indices.clone()
        masked_indices[mask] = self.mask_token_id
        
        # Pass the masked tokens to the transformer
        logits = self.transformer(masked_indices)

        z_indices = z_indices #ground truth
        logits = logits #transformer predict the probability of tokens
        
        return logits, z_indices
    
##[x] TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_bc, ratio, mask_func):
        z_indices_masked = z_indices.clone()
        z_indices_masked[mask_bc] = self.mask_token_id
        
        #Pass the masked tokens to the transformer, and get the logits ( b, 16*16, num_codebook_vectors+1 ) -> (10, 16*16, 1025)
        logits = self.transformer(z_indices_masked)
        
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = nn.functional.softmax(logits, dim=-1) # (10, 16*16, 1025)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1) # (10, 16*16)
 
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        z_indices_predict_prob[~mask_bc] = float('inf')
        
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.distributions.gumbel.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices_predict_prob.device) # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #sort the confidence for the rank 
        sorted_confidence = torch.sort(confidence, dim=-1)
        
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        z_indices_predict = z_indices_predict * mask_bc + z_indices * ~mask_bc

        mask_ratio = self.gamma_func(mask_func)(ratio)
        mask_len = math.floor(mask_ratio * mask_bc.sum())
        bound = sorted_confidence.values[:, mask_len].unsqueeze(-1)
        mask_bc = confidence < bound

        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
