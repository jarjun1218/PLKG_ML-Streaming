#!/usr/bin/env python3
"""
金鑰生成與 AES 加密示例（CSI-only 版，含視覺化）
"""
import os
import torch
import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import sha256
from greycode_quantization import quantization_1

# 使用 CSI-only 模型
from models.cnn_basic import cnn_basic
torch.serialization.add_safe_globals([cnn_basic])

SHOW_PLOTS = True  # 想直接跳視窗看圖就維持 True；若是純終端/遠端可改 False（只存檔）

def _load_model_any(path: str) -> torch.nn.Module:
    """同時相容整顆模型或 state_dict 的 .pth"""
    obj = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(obj, torch.nn.Module):         # 直接存整顆模型
        model = obj
    else:                                        # 存的是 state_dict
        state = obj.get('state_dict', obj)
        model = cnn_basic()
        model.load_state_dict(state, strict=False)
    model.eval()
    return model

def _encrypted_bytes_to_image(iv: bytes, ct: bytes, size, mode="RGB") -> Image.Image:
    """把 IV+Ciphertext 轉成同尺寸的雜訊圖（僅視覺化用）"""
    W, H = size
    if mode != "RGB":
        # 統一用 RGB 視覺化
        mode = "RGB"
    total = W * H * 3
    buf = iv + ct
    arr = np.frombuffer(buf, dtype=np.uint8, count=min(total, len(buf)))
    if arr.size < total:
        arr = np.pad(arr, (0, total - arr.size), mode="constant")
    arr = arr.reshape(H, W, 3)
    return Image.fromarray(arr, mode)

def generate_keys_and_encrypt():
    # 1) 載入 CSI-only 模型
    model = _load_model_any('model_reserved/cnn_basic/model_final.pth')

    # 2) 載入測試數據
    test_data = np.load("normalized_testing_set.npy", allow_pickle=True)

    # 統一成 (2, 1, F)
    if test_data.ndim == 2:            # (2, F) -> (2, 1, F)
        test_data = test_data[:, None, :]
    assert test_data.shape[0] == 2, f"期望第 0 維為 2（UAV/GCS），實際 {test_data.shape}"

    sample_data = test_data[:, 0:1, :]           # (2,1,F)

    # 3) 準備 CSI 輸入：自動適配 51 或 102 維
    feat = sample_data[:, :, 1:]                 # 只取特徵欄
    nfeat = feat.shape[-1]
    if nfeat == 102:                             # 兩通道 (2*51)
        csi_np = feat.reshape(2, 1, 2, 51)
    elif nfeat == 51:                            # 一通道 → 第二通道補 0
        c1 = feat.reshape(2, 1, 1, 51)
        c2 = np.zeros_like(c1)
        csi_np = np.concatenate([c1, c2], axis=2)
    else:
        raise ValueError(f"目前只支援 51 或 102 維 CSI，偵測到 nfeat={nfeat}")
    csi_data = torch.from_numpy(csi_np).float()  # (2,1,2,51)

    # 4) 前向推論（cnn_basic 輸出 Tensor 形狀 (2,51)）
    with torch.no_grad():
        out = model(csi_data)
        if not (isinstance(out, torch.Tensor) and out.shape[0] == 2):
            raise RuntimeError(f"模型輸出異常：{type(out)} / shape={getattr(out,'shape',None)}")
        uav_features, iot_features = out[0], out[1]   # 各為 (51,)

    # 5) 量化生成金鑰
    uav_key = quantization_1(uav_features.cpu().numpy().ravel(), Nbits=4, inbits=16, guard=2)
    iot_key = quantization_1(iot_features.cpu().numpy().ravel(), Nbits=4, inbits=16, guard=2)

    agree = (sum(int(a == b) for a, b in zip(uav_key, iot_key)) / min(len(uav_key), len(iot_key))) if len(uav_key) and len(iot_key) else 0.0
    print(f"UAV Key: {uav_key[:32]}...")
    print(f"IoT Key: {iot_key[:32]}...")
    print(f"Key Agreement: {agree*100:.2f}%")

    # 6) AES 加解密 + 影像視覺化
    os.makedirs("out", exist_ok=True)

    # 原圖
    image = Image.open("image/image.png").convert("RGB")
    image.save("out/01_original.png")

    image_bytes = image.tobytes()

    # 加密
    key_bytes = sha256.sha_byte(uav_key)     # 應回傳 16/24/32 bytes
    if len(key_bytes) not in (16, 24, 32):
        raise ValueError(f"AES 金鑰長度錯誤：{len(key_bytes)}")

    cipher = AES.new(key_bytes, AES.MODE_CBC)
    iv = cipher.iv
    encrypted_data = cipher.encrypt(pad(image_bytes, AES.block_size))

    # 視覺化加密後（雜訊圖）
    enc_img = _encrypted_bytes_to_image(iv, encrypted_data, image.size, "RGB")
    enc_img.save("out/02_encrypted_visual.png")

    # 解密
    decipher = AES.new(key_bytes, AES.MODE_CBC, iv)
    decrypted_data = unpad(decipher.decrypt(encrypted_data), AES.block_size)
    dec_img = Image.frombytes("RGB", image.size, decrypted_data)
    dec_img.save("out/03_decrypted.png")

    print(f"Original bytes: {len(image_bytes)}, Encrypted: {len(encrypted_data)}")
    print(f"Decrypt OK: {image_bytes == decrypted_data}")
    print("已輸出：out/01_original.png, out/02_encrypted_visual.png, out/03_decrypted.png")

    # 需要彈視窗就顯示
    if SHOW_PLOTS:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(image);    axes[0].set_title("Original");  axes[0].axis("off")
            axes[1].imshow(enc_img);  axes[1].set_title("Encrypted (visual)"); axes[1].axis("off")
            axes[2].imshow(dec_img);  axes[2].set_title("Decrypted"); axes[2].axis("off")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[INFO] 視窗顯示失敗（可能是遠端/無 GUI 環境）：{e}，已存檔在 out/ 資料夾。")

if __name__ == "__main__":
    generate_keys_and_encrypt()
