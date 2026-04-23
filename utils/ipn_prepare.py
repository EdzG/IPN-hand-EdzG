import sys
from pathlib import Path

# Default dataset path relative to the repository root
DATASET_PATH = Path("datasets/HandGestures/IPN_dataset")

def load_split_ipn(file_with_split='Annot_TestList.txt'):
    file_with_split = DATASET_PATH / file_with_split
    list_split = []
    
    with open(file_with_split, 'r') as f:
        # e.g., Annot_TestList -> Annot
        dict_name = file_with_split.name.split('_')[0]

        for line in f:
            params = line.strip().split(',')
            if len(params) < 5:
                continue
            
            params_dictionary = {}
            params_dictionary['dataset'] = dict_name

            for sens in ['frames']:
                # The paths will just be strings like 'frames/video_name'
                path = f"./{sens}/{params[0]}"
                key = sens

                label = int(params[2]) - 1 
                params_dictionary['label'] = label

                # first store path
                params_dictionary[key] = path
                # store start frame
                params_dictionary[key+'_start'] = int(params[3])
                params_dictionary[key+'_end'] = int(params[4])

            list_split.append(params_dictionary)
 
    return list_split

def create_list(example_config, sensor, class_types='all', new_lines=None):
    if new_lines is None:
        new_lines = []
        
    folder_path = example_config[sensor]
    label = example_config['label'] + 1
    start_frame = example_config[sensor + '_start']
    end_frame = example_config[sensor + '_end']
    
    line = [start_frame, end_frame]
    if class_types == 'all':
        new_lines.append(f"{folder_path} {label} {line[0]} {line[1]}")
    elif class_types == 'all_but_None':
        if label != 1:
            new_lines.append(f"{folder_path} {label-1} {line[0]} {line[1]}")
    elif class_types == 'binary':
        if label == 1:
            new_lines.append(f"{folder_path} {label} {line[0]} {line[1]}")
        else:
            new_lines.append(f"{folder_path} 2 {line[0]} {line[1]}")
    elif class_types == 'group':
        if label < 4:
            new_lines.append(f"{folder_path} {label} {line[0]} {line[1]}")
        else:
            new_lines.append(f"{folder_path} 4 {line[0]} {line[1]}")
    elif class_types == 'gests_only':
        if label > 3:
            new_lines.append(f"{folder_path} {label-3} {line[0]} {line[1]}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python utils/ipn_prepare.py <subset> <file_name> <class_types>")
        sys.exit(1)
        
    subset = sys.argv[1]          # [training, validation] 
    file_name = sys.argv[2]       # [trainlistall.txt, trainlistall_but_None.txt, trainlistbinary.txt, vallistall.txt, ...]
    class_types = sys.argv[3]     # [all, all_but_None, binary, group]

    sensors = ['frames', 'segment', 'flow']

    if subset == 'training':
        file_list = "Annot_TrainList.txt"
    elif subset == 'validation':
        file_list = "Annot_TestList.txt"
    else:
        print(f"Unknown subset: {subset}")
        sys.exit(1)
    
    subset_list = load_split_ipn(file_with_split=file_list)
    output_dir = Path('annotation_ipnGesture')
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, sensor in enumerate(sensors):
        new_lines = [] 
        print(f"Processing {subset} List for {sensor}...")
        for sample_name in subset_list:
            create_list(example_config=sample_name, sensor=sensor, class_types=class_types, new_lines=new_lines)

        print("Writing to the file ...")
        if idx > 0:
            f_name_parts = file_name.split('.')
            f_name = f"{f_name_parts[0]}_{sensor[0:3]}.{f_name_parts[1]}"
        else:
            f_name = file_name
            
        file_path = output_dir / f_name
        with open(file_path, 'w') as myfile:
            for new_line in new_lines:
                myfile.write(new_line + '\n')
        print("Successfully wrote file to:", file_path)
