import json
import os

def update_notebook(path, is_genomics=False):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            
            # Thay đổi Dataset definition để add collate_fn
            if 'class MultimodalDataset(Dataset):' in source or 'class GenomicsDataset(Dataset):' in source:
                new_source = source + "\n\n" + \
                "import torch.nn.functional as F\n" + \
                "def pad_collate(batch):\n" + \
                "    imgs = [item[0] for item in batch]\n" + \
                "    tabs = [item[1] for item in batch]\n" + \
                "    labels = [item[2] for item in batch]\n" + \
                "    pids = [item[3] for item in batch]\n" + \
                "    \n" + \
                "    max_len = max([img.size(0) for img in imgs])\n" + \
                "    padded_imgs = []\n" + \
                "    for img in imgs:\n" + \
                "        pad_size = max_len - img.size(0)\n" + \
                "        if pad_size > 0:\n" + \
                "            padded = F.pad(img, (0, 0, 0, pad_size), \"constant\", 0)\n" + \
                "        else:\n" + \
                "            padded = img\n" + \
                "        padded_imgs.append(padded)\n" + \
                "        \n" + \
                "    imgs_tensor = torch.stack(padded_imgs, dim=0)\n" + \
                "    tabs_tensor = torch.stack(tabs, dim=0)\n" + \
                "    labels_tensor = torch.tensor(labels)\n" + \
                "    return imgs_tensor, tabs_tensor, labels_tensor, pids\n"
                cell['source'] = [new_source]
                
            # Cập nhật Training Loop
            if 'if len(valid_pids) > 0:' in source and 'skf = StratifiedKFold' in source:
                new_source = []
                lines = source.split('\n')
                for line in lines:
                    if 'epochs, batch_size, accumulation_steps = 30, 1, 16' in line:
                        new_source.append("    # [DUAL-GPU MODE] Tăng Batch Size lên 2, giảm Accumulation xuống 8")
                        new_source.append("    epochs, batch_size, accumulation_steps = 30, 2, 8")
                    elif 'train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)' in line:
                        new_source.append("        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)")
                    elif 'test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)' in line:
                        new_source.append("        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)")
                    elif 'model = Multimodal_' in line:
                        new_source.append(line)
                        new_source.append("        if torch.cuda.device_count() > 1:")
                        new_source.append("            print(f\"🔥 Kích hoạt DUAL-GPU: Đang chạy trên {torch.cuda.device_count()} GPUs (T4x2)!\")")
                        new_source.append("            model = nn.DataParallel(model)")
                    else:
                        new_source.append(line)
                        
                cell['source'] = '\n'.join(new_source)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

update_notebook(r'C:\Users\huynh\Desktop\breast cancer\06.1_Multimodal_WSI_Clinical.ipynb')
update_notebook(r'C:\Users\huynh\Desktop\breast cancer\06.2_Multimodal_WSI_Genomics.ipynb')
print("Dual-GPU support injected successfully.")
