import os
import torch
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


from models.multimodal_encoder.dinov2_encoder import DinoV2VisionTower
from models.multimodal_encoder.dinov3_encoder import DinoV3VisionTower
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.multimodal_encoder.qwen25_encoder import Qwen2Embedder

from models.fusion.qformer import qformer_base#, qformerv2_base




class primitiveSkillNet(nn.Module):
    """A wrapper for the RDT model, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
    """
    def __init__(
        self, args, 
        device='cuda',
        dtype=torch.bfloat16,
        image_size=None,
        
        #pretrained_vision_encoder_name_or_path="facebook/dinov2-base",#google/siglip-base-patch16-25;siglip-so400m-patch14-384///
        #pretrained_text_encoder_name_or_path="Qwen/Qwen2.5-0.5B",#t5-v1_1-small
        #text_encoder='Qwen2.5-0.5B',#t5-v1_1-small/
        #vision_encoder='dinov2'   #dinov2/Siglip
    ):
        super().__init__() 
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device

        # 't5-v1_1-xxl', 'google/t5-v1_1-xxl';'t5-v1_1-base',"google/t5-v1_1-base"; 'Qwen2.5-7B', "Qwen/Qwen2.5-7B"; 'Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B'
        self.text_encoder_name = self.args["common"]["text_encoder"]
        # 'DinoV2','facebook/dinov2-giant'; 'Siglip','google/siglip-base-patch16-256-i18n'/'google/siglip-so400m-patch14-384';
        self.vision_encoder_name = self.args["common"]["vision_encoder"]
        # We do not use the text encoder due to limited GPU memory
        self.text_tokenizer, self.text_model = self.get_text_encoder(self.args["common"]["pretrained_text_encoder_name_or_path"])
        self.image_processor, self.vision_model = self.get_vision_encoder(self.args["common"]["pretrained_vision_encoder_name_or_path"])

        text_dim = getattr(self.text_model.config, "hidden_size", 512)
        self.vision_dim = getattr(self.vision_model, "hidden_size", 1024)
        
        self.num_hist_state = self.args['common']['num_hist_state']
        self.state_dim = self.args['dataset']['state_dim']
        
        if self.num_hist_state>=1:
            self.state_embed_dim = self.args['common']['state_embed_dim']
            self.fusioner,self.image_proj = self.get_fusioner()
            self.state_encoder = torch.nn.Sequential(*[torch.nn.Linear(self.num_hist_state*self.state_dim, 128),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(128, 256),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(256, self.state_embed_dim)])
            self.state_encoder = self.state_encoder.to(self.device, dtype=self.dtype)
            self.feature_dim = text_dim + self.vision_dim+self.state_embed_dim
        else:
            self.fusioner,self.image_proj = self.get_fusioner()
            self.feature_dim = text_dim + self.vision_dim
        
        self.classifer = self.get_classifer(num_class=self.args["dataset"].get("num_class",8))
        
        self.reset()
        
        print('********* self.feature_dim',self.feature_dim)

    def get_classifer(self, activation='relu',num_class=8):
        """Initialize the model."""
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU
        # Initialize model with arguments
        intermediate_state_dim = 256
        _model = torch.nn.Sequential(*[torch.nn.Linear(self.feature_dim, intermediate_state_dim),
                                                       activate_fn(),
                                                       torch.nn.Linear(intermediate_state_dim, intermediate_state_dim),
                                                       activate_fn(),
                                                       torch.nn.Linear(intermediate_state_dim, num_class)])
        return _model
        
    def get_fusioner(self):
        if self.num_hist_state>=1:
            fusioner = qformer_base(**self.args["qformer"])
            proj = nn.Linear(2*self.vision_dim, self.vision_dim)
        else:
            fusioner = qformer_base(**self.args["qformer"])
            proj = None
        
        return fusioner,proj
    

    def get_text_encoder(self, pretrained_text_encoder_name_or_path):
        if self.text_encoder_name=='Qwen2.5-7B' or self.text_encoder_name=='Qwen2.5-0.5B':
            text_embedder = Qwen2Embedder(from_pretrained=pretrained_text_encoder_name_or_path, 
                            model_max_length=self.args["dataset"]["tokenizer_max_length"],
                            device=self.device)
        #elif self.text_encoder_name=='t5-v1_1-xxl' or self.text_encoder_name=='t5-v1_1-base':
        else:
            text_embedder = T5Embedder(from_pretrained=pretrained_text_encoder_name_or_path, 
                                    model_max_length=self.args["dataset"]["tokenizer_max_length"], 
                                    device=self.device)
        
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        if self.vision_encoder_name=='Siglip':
            vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
            image_processor = vision_encoder.image_processor
        elif self.vision_encoder_name=='DinoV3':
            #vision_encoder = DinoV3VisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
            vision_encoder = DinoV3VisionTower(vision_tower="facebook/dinov3-convnext-tiny-pretrain-lvd1689m", args=None)
            image_processor = vision_encoder.image_processor
        else:
            #'facebookresearch/dinov2/dinov2_vitg14_lc'
            vision_encoder = DinoV2VisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
            image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.device
        weight_dtype = self.dtype
        self.text_model.eval()
        self.vision_model.eval()

        self.classifer = self.classifer.to(device, dtype=weight_dtype)
        self.text_model = self.text_model.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)
        if self.image_proj is not None:
            self.image_proj = self.image_proj.to(device, dtype=weight_dtype)
        
        self.fusioner = self.fusioner.to(device, dtype=weight_dtype)#.eval()
        

    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return 
        print(f'Loading weights from {pretrained}')
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt'):
            checkpoint =  torch.load(pretrained)
            #self.classifer.load_state_dict(checkpoint["module"])
            classifier_dict = {k.replace("classifer.", "classifier."): v 
                       for k, v in checkpoint.items() if "classifer" in k}
            self.classifer.load_state_dict(classifier_dict, strict=False)
        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.classifer, pretrained)
        else:
            #raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")
            checkpoint =  torch.load(pretrained)
            classifier_dict = {k.replace("classifer.", "classifier."): v 
                       for k, v in checkpoint.items() if "classifer" in k}
            self.classifer.load_state_dict(classifier_dict, strict=False)

    def encode_instruction(self, instruction, device="cuda"):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device
        
        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        tokens = self.text_tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            #pred = self.text_model(tokens).last_hidden_state.detach()
            pred = self.text_model.encoder(input_ids=tokens).last_hidden_state.detach()

        return pred

    def features_fusion(self, static_images, wrist_images, instruction, hist_states):
    
        #image_embeds = torch.cat((self.vision_model(static_images),self.vision_model(wrist_images)),dim=-1)   # (B, N, D)
        static_embeds = self.vision_model(static_images)
        wrist_embeds = self.vision_model(wrist_images)   # (B, N, D)
    
        image_embeds = self.image_proj(torch.cat((static_embeds,wrist_embeds),dim=-1))
        text_embeds = self.encode_instruction(instruction).to(dtype=self.dtype)
        
        feats = self.fusioner(image_embeds, text_embeds).mean(dim=1)
        if hist_states is not None:
            state_embeds = self.state_encoder(hist_states.to(dtype=self.dtype))
            feats = torch.cat((feats,state_embeds),dim=-1)
            
        return feats

    #@torch.no_grad()
    def forward(self, images, instruction,hist_states):
        """
        Predict the next action chunk given the 
        proprioceptive states, images, and instruction embeddings.

        Args:
            proprio: proprioceptive states
            images: RGB images, the order should be
                [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1}, 
                ext_{t}, right_wrist_{t}, left_wrist_{t}]
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype
        
        # The background image used for padding
        background_color = np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        '''
        # only for siglip
        background_image = np.ones((
            self.image_processor.size["height"], 
            self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        '''
        # only for dinov2
        background_image = np.ones((224, 224, 3), dtype=np.uint8) * background_color
        
        # Preprocess the images by order and encode them
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)
            
            if self.image_size is not None:
                image = transforms.Resize(self.image_size)(image)
            
            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
       
            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            # 
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)
        half = len(image_tensor_list)//2
        static_images = torch.stack(image_tensor_list[:half], dim=0).to(device, dtype=dtype)
        wrist_images = torch.stack(image_tensor_list[half:], dim=0).to(device, dtype=dtype)
        #image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)
        
        feat = self.features_fusion(static_images, wrist_images, instruction, hist_states)

        #print('********* feat shape',feat.size())
        out = self.classifer(feat)

        return out, feat
