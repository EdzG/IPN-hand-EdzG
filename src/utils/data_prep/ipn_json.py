import sys
import json
from pathlib import Path

def convert_csv_to_dict(csv_path, subset, labels):
    keys = []
    key_labels = []
    key_start_frame = []
    key_end_frame = []
    
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
                
            basename = parts[0]
            label_idx = int(parts[1]) - 1
            class_name = labels[label_idx]
            start_frame = parts[2]
            end_frame = parts[3]
            
            keys.append(basename)
            key_labels.append(class_name)
            key_start_frame.append(start_frame)
            key_end_frame.append(end_frame)
        
    database = {}
    for i in range(len(keys)):
        key = keys[i]  
        if key in database: # need this because there might be multiple clips per folder
            key = key + '^' + str(i) 
        
        database[key] = {
            'subset': subset,
            'annotations': {
                'label': key_labels[i],
                'start_frame': key_start_frame[i],
                'end_frame': key_end_frame[i]
            }
        }
    
    return database

def load_labels(label_csv_path):
    labels = []
    with open(label_csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels.append(parts[1])
    return labels

def convert_ipn_csv_to_activitynet_json(label_csv_path, train_csv_path, val_csv_path, dst_json_path):
    labels = load_labels(label_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training', labels)
    val_database = convert_csv_to_dict(val_csv_path, 'validation', labels)
    
    dst_data = {
        'labels': labels,
        'database': {**train_database, **val_database}
    }
    
    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file, indent=4)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python utils/ipn_json.py <csv_dir_path>")
        sys.exit(1)
        
    csv_dir_path = Path(sys.argv[1])
    sens_suffix = ''
        
    for class_type in ['all', 'all_but_None', 'binary']:
        if class_type == 'all':
            class_ind_file = 'classIndAll.txt'
        elif class_type == 'all_but_None':
            class_ind_file = 'classIndAllbutNone.txt'
        elif class_type == 'binary':
            class_ind_file = 'classIndBinary.txt'
        else:
            continue

        label_csv_path = csv_dir_path / class_ind_file
        train_csv_path = csv_dir_path / f"trainlist{class_type}{sens_suffix}.txt"
        val_csv_path = csv_dir_path / f"vallist{class_type}{sens_suffix}.txt"
        dst_json_path = csv_dir_path / f"ipn{class_type}{sens_suffix}.json"

        if not label_csv_path.exists() or not train_csv_path.exists() or not val_csv_path.exists():
            print(f"Skipping {class_type} because some input files are missing in {csv_dir_path}.")
            continue
            
        convert_ipn_csv_to_activitynet_json(label_csv_path, train_csv_path, val_csv_path, dst_json_path)
        print('Successfully wrote to json:', dst_json_path)

# HOW TO RUN:
# python utils/ipn_json.py ./annotation_ipnGesture
