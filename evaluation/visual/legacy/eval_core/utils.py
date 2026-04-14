import json
import csv
import os
from collections import defaultdict
import argparse
import tianhuieval.const as const

def load_json_result(file_name):
    
    file_json = open(file_name)
    json_obj =  json.load(file_json)
    
    try:
        info_json = json_obj[const.JSON_INFO]
        result_task = info_json[const.JSON_TASK]
        result_model = info_json[const.JSON_MODEL]
        result_dataset = info_json[const.JSON_DATASET]
    except:
        info_json = {}
        print (json_obj[const.JSON_INFO])
        result_task = json_obj[const.JSON_TASK]
        result_model = json_obj[const.JSON_MODEL]
        result_dataset = json_obj[const.JSON_DATASET]
    data_json = json_obj[const.JSON_DATA]

    return json_obj, info_json, result_task, result_model, result_dataset, data_json


def flatten_dict(d, parent_key='', sep='_'):
    """Flatten a nested dictionary structure"""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def process_json_files(input_dir, output_dir):
    task_data = defaultdict(list)
    fieldnames = defaultdict(dict)  # Now stores info and results fields separately

    print ('input:',input_dir)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    info = data.get('info', {})
                    results = data.get('results', {})
                    
                    if not info or not results:
                        continue
                    
                    task = info.get('task').upper()
                    
                    if not task:
                        continue
                    
                    flat_results = flatten_dict(results)
                    combined = {**info, **flat_results}
                    
                    task_data[task].append(combined)
                    
                    # Track info and results fields separately
                    if task not in fieldnames:
                        fieldnames[task] = {
                            'info_fields': set(info.keys()),
                            'results_fields': set(flat_results.keys())
                        }
                    else:
                        fieldnames[task]['info_fields'].update(info.keys())
                        fieldnames[task]['results_fields'].update(flat_results.keys())
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    os.makedirs(output_dir, exist_ok=True)
    for task, data in task_data.items():
        if not data:
            continue

        
        
        safe_task = ''.join(c if c.isalnum() else '_' for c in task)
        output_file = os.path.join(output_dir,f"{safe_task}_results.csv")
        
        # Get ordered fieldnames (info fields first, then results fields)
        ordered_fields = (
            sorted(fieldnames[task]['info_fields']) + 
            sorted(fieldnames[task]['results_fields'])
        )
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fields)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Created {output_file} with {len(data)} entries (columns: {len(ordered_fields)})")

def json_to_csv(json_file_path, output_dir, unique_info_fields=['task', 'model', 'dataset']):

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        info = data.get('info', {})
        results = data.get('results', {})
        
        if not info or not results:
            print(f"Skipping {json_file_path}: missing info or results")
            return
        
        task = info.get('task')
        if not task:
            print(f"Skipping {json_file_path}: no task specified in info")
            return
        
        # Flatten the results and combine with info
        flat_results = flatten_dict(results)
        combined = {**info, **flat_results}
        
        # Create safe task name for filename
        safe_task = ''.join(c if c.isalnum() else '_' for c in task)
        csv_file_path = os.path.join(output_dir, f"{safe_task}_results.csv")
        
        # Generate unique key for this record
        unique_key = tuple(str(info.get(field, '')) for field in unique_info_fields)
        
        # Check if CSV exists and read existing data
        existing_data = []
        file_exists = os.path.exists(csv_file_path)
        
        if file_exists:
            with open(csv_file_path, 'r') as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames
                existing_data = list(reader)
        else:
            existing_fields = []
        
        # Find if this record already exists
        record_exists = False
        updated_data = []
        
        for record in existing_data:
            # Generate unique key for existing record
            existing_key = tuple(str(record.get(field, '')) for field in unique_info_fields)
            
            if existing_key == unique_key:
                # Update existing record with new data
                record.update(combined)
                record_exists = True
            updated_data.append(record)
        
        # Combine all fields (existing + new)
        all_fields = set(existing_fields) | set(combined.keys())
        ordered_fields = sorted(all_fields)
        
        # Write the data
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fields)
            writer.writeheader()
            
            # Write all existing (possibly updated) records
            writer.writerows(updated_data)
            
            # If new record, append it
            if not record_exists:
                # Ensure all fields are present in the data (fill missing with None)
                row = {field: combined.get(field) for field in ordered_fields}
                writer.writerow(row)
        
        action = "Updated" if record_exists else "Appended"
        print(f"{action} data from {json_file_path} to {csv_file_path}")
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {str(e)}")


def process_all_type(group_dir, output_dir):
    
    # Specify the directory path
    directory_path = group_dir
    
    # List all entries in the directory
    entries = os.listdir(directory_path)
    
    # Filter out only the directories (first-layer folders)
    first_layer_folders = [
        entry for entry in entries 
        if os.path.isdir(os.path.join(directory_path, entry))
    ]
    
    # Print the list of first-layer folders
    print("First-layer folders:")
    for folder in first_layer_folders:
        
        process_json_dir =os.path.join(group_dir,folder)
        group_output_dir = os.path.join(output_dir,folder)
        process_json_files(process_json_dir, group_output_dir)