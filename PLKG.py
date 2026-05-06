import numpy as np
import plkg.greycode_quantization as quantization
# import plkg.ecc as ecc
import plkg.sha256 as sha256
import torch
import torch.nn as nn


# PLKG
class PLKG:
    def __init__(self):
        pass
            
    def run_quantization(self, csi_sequence):
        # print("CSI Sequence:", self.csi_sequence)
        quantizatized_csi = quantization.quantization_1(csi_sequence, 2, 13, 0)
        # print(f"{num} Quantized CSI: {quantizatized_csi}")
        return quantizatized_csi
    
    def run_model(self, type, input_array):
        # Load the model
        cnn_quan_model = torch.load("model/cnn_basic_quan/model_final.pth", map_location=torch.device('cpu'), weights_only=False)
        cnn_model = torch.load("model/cnn_basic/model_final_test.pth", map_location=torch.device('cpu'), weights_only=False)
        cnn_quan_model = cnn_quan_model.to(torch.device('cpu'))
        cnn_model = cnn_model.to(torch.device('cpu'))
        cnn_quan_model.eval()
        cnn_model.eval()
        # print(cnn_quan_model)
        if type == 'csi':
            csi_array = np.array(input_array, dtype=np.float32)
            # print(f"CSI Array: {csi_array.shape}")
            csi_array = torch.tensor(csi_array, dtype=torch.float32).unsqueeze(0)
            # print(f"CSI Array: {csi_array.shape}")
            with torch.no_grad():
                modified_csi = cnn_model(csi_array).detach().numpy()
                
            return modified_csi
        
        elif type == 'key':
            key_array = np.array(input_array, dtype=np.float32)
            # print(f"Key Array: {key_array.shape}")
            key_array = torch.tensor(key_array, dtype=torch.float32).unsqueeze(0)
            # Pass the key array through the model
            with torch.no_grad():
                modified_key = cnn_quan_model(key_array).detach().numpy()
            for i in range(len(modified_key)):
                modified_key[i] = modified_key[i].round()
            # convert into string
            modified_key = ''.join([str(int(i)) for i in modified_key])
            
            return modified_key
        
        
    
    def run_ecc(self, key):
        # Run ECC
        ecc_code = self.bch.encode(key)
        return ecc_code
        
    def run_sha256(self, csi_array):
        # Run SHA256
        hashed_key = sha256.sha_byte(csi_array)
        return hashed_key
        
if __name__ == "__main__":
    # Example CSI sequence
    csi_sequence = np.random.rand(51)
    csi_sequence_2 = np.random.rand(51)
    # print("CSI Sequence:", csi_sequence)

    plkg_test = PLKG()

    # Run the PLKG test
    key1 = plkg_test.run_quantization(csi_sequence)
    key2 = plkg_test.run_quantization(csi_sequence_2)
    print(f"Key 1: {key1}")
    print(f"Key 2: {key2}")
        
    # Compare the keys
    mismatch = 0
    for i in range(len(key1)):
        if key1[i] != key2[i]:
            mismatch += 1
    print(f"KDR: {mismatch/len(key1)}")
    
    key_array = [list(key1), list(key2)]
    modified_key = plkg_test.run_model(key_array)
    print(f"Modified Key: {modified_key}")
    mismatch = 0
    for i in range(len(modified_key)):
        if modified_key[i] != key1[i]:
            mismatch += 1
    print(f"Modified KDR: {mismatch/len(modified_key)}")
    
    hashed_key1 = plkg_test.run_sha256(key1)
    hashed_key2 = plkg_test.run_sha256(key2)
    print(f"Hashed Key 1: {hashed_key1}")
    print(f"Hashed Key 2: {hashed_key2}")
    print(f"Hashed Key 1 Length: {len(hashed_key1)}")
    print(f"Hashed Key 2 Length: {len(hashed_key2)}")
    
        
        
        