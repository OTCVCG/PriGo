import os
import h5py
import yaml
import random
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from models.primitiveSkillNet import primitiveSkillNet



def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Calculates the total memory size of all parameters in MB.
    Assumes parameters are stored in the default dtype (usually float32/4 bytes).
    """
    total_bytes = 0
    
    # Iterate over all parameters, including buffers
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
        
    # 1 Megabyte (MB) = 1024 * 1024 bytes
    total_megabytes = total_bytes / (1024 * 1024)
    
    return total_megabytes


def train_one_epoch(args,psnet,optimizer,lr_scheduler):
    exp_dir = os.path.join(args.data_root, args.dataset_name)
    dataset_name_list = [str(f) for f in Path(exp_dir).glob("*.hdf5")]
    stIdx = len(exp_dir)+1
    num_hist= args.num_hist_state-1
    psnet.classifer.train()
    for data_name in dataset_name_list:
        instruction = data_name[stIdx:-10].replace('_', ' ')        
        with h5py.File(data_name, "r") as f: 
            demos = list(f['data'].keys()) 
            for demo_key in demos:
                labels = torch.from_numpy(np.array(f['data'][demo_key]['primitives'])).long().to('cuda')
                obs = np.array(f['data'][demo_key]['obs']['agentview_rgb'])
                obs_imgs = [Image.fromarray(ob) for ob in obs]
                
                obs = np.array(f['data'][demo_key]['obs']['eye_in_hand_rgb'])
                obs_imgs = obs_imgs + [Image.fromarray(ob) for ob in obs]
                
                if num_hist>=1:
                    curr_states = torch.from_numpy(np.array(f['data'][demo_key]['obs']['ee_states'])).float()
                    #print("curr_states.shape = ",curr_states.shape)
                    T, dims = curr_states.shape[0],curr_states.shape[1]
                    hist_states = curr_states.repeat((1,args.num_hist_state)).to('cuda')
                    for i in range(num_hist-1):
                        st = i*dims
                        hist_states[num_hist-1-i:,st:st+dims]=curr_states[num_hist-1-i:,:]
                else:
                    hist_states = None
                
                optimizer.zero_grad()
                pred_logits,_ = psnet(obs_imgs, instruction,hist_states)
                loss = F.cross_entropy(pred_logits, labels)
                print(f"[LOSS] {loss.item():.4f}", flush=True)
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()



def test(args,psnet):
    exp_dir = os.path.join(args.data_root, args.dataset_name)
    dataset_name_list = [str(f) for f in Path(exp_dir).glob("*.hdf5")]
    stIdx = len(exp_dir)+1
    num_hist= args.num_hist_state-1
    psnet.classifer.eval()
    correct = 0
    total = 0
    for data_name in dataset_name_list:
        instruction = data_name[stIdx:-10].replace('_', ' ')
        with h5py.File(data_name, "r") as f: 
            demos = list(f['data'].keys()) 
            for demo_key in demos:
                labels = torch.from_numpy(np.array(f['data'][demo_key]['primitives'])).long().to('cuda')
                obs = np.array(f['data'][demo_key]['obs']['agentview_rgb'])
                #print("obs.shape = ",obs.shape)
                obs_imgs = [Image.fromarray(ob) for ob in obs]
                
                obs = np.array(f['data'][demo_key]['obs']['eye_in_hand_rgb'])
                #print("obs.shape = ",obs.shape)
                obs_imgs = obs_imgs + [Image.fromarray(ob) for ob in obs]
                
                if num_hist>=1:
                    curr_states = torch.from_numpy(np.array(f['data'][demo_key]['obs']['ee_states'])).float()
                    
                    T, dims = curr_states.shape[0],curr_states.shape[1]
                    hist_states = curr_states.repeat((1,args.num_hist_state)).to('cuda')
                    for i in range(num_hist-1):
                        st = i*dims
                        hist_states[num_hist-1-i:,st:st+dims]=curr_states[num_hist-1-i:,:]
                else:
                    hist_states = None
                
                with torch.no_grad():
                    logits,_ = psnet(obs_imgs, instruction,hist_states)
                    pred_label = torch.argmax(logits, dim=-1)
                    correct +=(pred_label == labels).sum().item()
                    total += labels.shape[0]
                    

    accuracy = correct / total if total > 0 else 0
    print(f"Dataset name: {args.dataset_name}, Primitive prediction accuracy: {accuracy*100:.2f}%", flush=True)
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./LIBERO/libero/datasets/', help='root path of libero dataset')
    parser.add_argument('--dataset_name', type=str, default='libero_object', help="'libero_object','libero_spactial','libero_goal', 'libero_10', 'libero_90'")
    parser.add_argument('--config_path', type=str, default='./configs/base.yaml', help='path to configuration file')
    #qformer5e-4cosine/dinov2-base5e-5/dinov2-class6
    parser.add_argument('--output_dir', type=str, default='./results/', help='Output path for checkpoints')
    parser.add_argument("--learning_rate",type=float,default=5e-5,help="Initial learning rate (after the potential warmup period)")#5e-6/5e-4/1e-3
    parser.add_argument("--lr_scheduler",type=str,default="constant",
        help=('["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"]')#constant is better than linear and cosine
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=300, help="Number of steps for the warmup in the lr scheduler")
    parser.add_argument("--max_train_steps",type=int,default=600,help="Total number of training steps to perform")
    parser.add_argument("--use_sampling",type=int,default=1,help="sampling primitive actions from the predicted probability")
    parser.add_argument("--num_hist_state",type=int,default=4,help="The number of history states used, 1 means just using the current state")
    parser.add_argument("--seed",type=int,default=2025,help="Total number of training steps to perform")
    args = parser.parse_args()

    
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    config['common']['num_hist_state']=args.num_hist_state
    print(f"[INFO] Data root directory is : {args.data_root}", flush=True)
    print(f"[INFO] The text encoder is : ", config['common']['text_encoder'], flush=True)
    print(f"[INFO] The image encoder is : ", config['common']['vision_encoder'], flush=True)
    args.output_dir = os.path.join(args.output_dir, config['common']['vision_encoder']+'_'+config['common']['text_encoder']+'_hist')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] The output directory is : {args.output_dir}", flush=True)
    
    psnet = primitiveSkillNet(config)
    psnet.load_state_dict(torch.load(args.output_dir+'/checkpoint-103.pt'))
    print(f"Total Parameter Size: {get_model_size_mb(psnet):.2f} MB")
    print(f"Total Parameter Size of text_model: {get_model_size_mb(psnet.text_model):.2f} MB")
    print(f"Total Parameter Size of vision_model: {get_model_size_mb(psnet.vision_model):.2f} MB")
    print(f"Total Parameter Size of fusioner: {get_model_size_mb(psnet.fusioner):.2f} MB")
    print(f"Total Parameter Size of classifer: {get_model_size_mb(psnet.classifer):.2f} MB")
    optimizer = torch.optim.AdamW(psnet.parameters(),lr=args.learning_rate,betas=(0.9, 0.999),weight_decay=1e-2,eps=1e-08)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=1,
        power=1.0,
    )
    
    
    dataset_names = ['libero_object','libero_spatial','libero_goal', 'libero_10', 'libero_90']
    random.seed(args.seed)
    for i in range(1,1+args.max_train_steps):
        idx = random.randint(0,4)
        args.dataset_name = dataset_names[idx]
        print(f"Training dataset name: {args.dataset_name}, Step: {i}", flush=True)
        train_one_epoch(args,psnet,optimizer,lr_scheduler)
        if i % 50 == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{i}.pt")
            print(f"[INFO] Save checkpoint to: {save_path}", flush=True)
            torch.save(psnet.state_dict(), save_path)
            accuracy = test(args,psnet)
    
    for name in dataset_names:
        args.dataset_name = name
        accuracy = test(args,psnet)


if __name__ == "__main__":
    main()
