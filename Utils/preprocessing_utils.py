import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data

def get_four_momenta(jet_tuple):
    # Set all values to zero
    energies = torch.tensor([getattr(jet_tuple, f'E_{i}') for i in range(200)])
    x_values = torch.tensor([getattr(jet_tuple, f'PX_{i}') for i in range(200)])
    y_values = torch.tensor([getattr(jet_tuple, f'PY_{i}') for i in range(200)])
    z_values = torch.tensor([getattr(jet_tuple, f'PZ_{i}') for i in range(200)])
    all_values = torch.stack([energies, x_values, y_values, z_values], dim=1)

    # Only keep existing jets
    existing_jet_mask = energies > 0
    return all_values[existing_jet_mask]

def calc_kinematics(x, y, z):
    pt = np.sqrt(x**2 + y**2)
    theta = np.arctan2(pt, z)
    eta = -1. * np.log(np.tan(theta / 2.))
    phi = np.arctan2(y, x)
    
    return pt, eta, phi

def get_higher_features(p):
    
    E, x, y, z = p.T
    pt, eta, phi = calc_kinematics(x,y,z)
    
    jet_p4 = p.sum(0)        
    jet_pt, jet_eta, jet_phi = calc_kinematics(jet_p4[1], jet_p4[2], jet_p4[3])
    
    delta_eta = eta - jet_eta
    delta_phi = phi - jet_phi
    delta_phi[delta_phi > np.pi] -= 2 * np.pi
    delta_phi[delta_phi < -np.pi] += 2 * np.pi
    
    return pt, eta, phi, delta_eta, delta_phi, jet_p4, jet_pt, jet_eta, jet_phi

def build_all_features(jet):
    p = get_four_momenta(jet)
    y = torch.tensor(jet.is_signal_new)

    pt, eta, phi, delta_eta, delta_phi, jet_p4, jet_pt, jet_eta, jet_phi = get_higher_features(p)
    delta_pt = pt / jet_pt
    log_delta_pt = torch.log(delta_pt)
    delta_E = p[:, 0] / jet_p4[0]
    log_delta_E = torch.log(delta_E)
    delta_R = torch.sqrt( delta_eta**2 + delta_phi**2 )
    jet_mass = torch.sqrt(jet_p4[0]**2 - jet_p4[1]**2 - jet_p4[2]**2 - jet_p4[3]**2)

    pyg_jet = Data(pE=p[:, 0], px=p[:, 1], py=p[:, 2], pz=p[:, 3], 
                        y=y,
                        log_pt = torch.log(pt), 
                        log_E = torch.log(p[:, 0]),
                        delta_pt = delta_pt,
                        log_delta_pt = log_delta_pt,
                        delta_E = delta_E,
                        log_delta_E = log_delta_E,
                        delta_R = delta_R,
                        delta_eta = delta_eta,
                        delta_phi = delta_phi,
                        jet_pt = jet_pt,
                        jet_pE = jet_p4[0],
                        jet_px = jet_p4[1],
                        jet_py = jet_p4[2],
                        jet_pz = jet_p4[3],
                        jet_mass = jet_mass,
                        jet_eta = jet_eta,
                        jet_phi = jet_phi)

    # Convert all to float
    for key in pyg_jet.keys:
        pyg_jet[key] = pyg_jet[key].float()

    return pyg_jet

def process_jet_to_pyg(jet, feature_scales = None):
    pyg_jet = build_all_features(jet)
    if feature_scales is not None:
        # Normalize the features
        for feature, scale in feature_scales.items():
            pyg_jet[feature] /= scale
    
    return pyg_jet

def hdf5_to_pyg_event(jet_entry, feature_scales):
    jet_tuple, _ = jet_entry
    return process_jet_to_pyg(jet_tuple, feature_scales)
    

def hdf5_to_pyg_events(input_file, output_dir, feature_scales, start=0, stop=None):
    os.makedirs(output_dir, exist_ok=True)
    print("Processing files")
    batch_size = 100000

    data_iter = pd.read_hdf(input_file, key='table', iterator=True, chunksize=batch_size)
    i=0
    for chunk in data_iter:
        if stop is not None and (i*batch_size >= stop):
            break

        jet_batch = []
        for jet_id, jet_tuple in enumerate(tqdm(chunk.itertuples(), total=len(chunk))):
            final_id = jet_id + i * batch_size
            if final_id >= start and (stop is None or final_id < stop):
                jet_batch.append(hdf5_to_pyg_event((jet_tuple, final_id), feature_scales))
        print(f"Saving batch to file {output_dir}/{i}.pt")
        torch.save(jet_batch, f"{output_dir}/{i}.pt")
        i+=1
