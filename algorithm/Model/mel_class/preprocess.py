import os
import pandas as pd
import shutil

# Define paths
audio_dir = r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/Perceptual Voice Qualities Database (PVQD)/Audio_Files'
output_dir = r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/mel_class/renamed_audio_files'
os.makedirs(output_dir, exist_ok=True)

# Load Excel file with participant details
input_excel = r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/Perceptual Voice Qualities Database (PVQD)/Org_Dataset.xlsx'
data = pd.read_excel(input_excel)

# Prepare lists for old and new file names
old_names = []
new_names = []
ages = []
genders = []
diagnoses = []

# Initialize counters for normal and disordered files
normal_count = 1
disorder_count = 1
processed_count = 0

# Get list of actual audio files in audio_dir
audio_files = os.listdir(audio_dir)

# Process each row in the Excel file
for idx, row in data.iterrows():
    participant_id = str(row['Participant ID'])  # Convert to string for matching
    age = row['Age']
    gender = row['Gender']
    diagnosis = row['Diagnosis']

    # Skip rows where Diagnosis is empty or has 'y' or 'Y'
    if pd.isna(diagnosis) or diagnosis.lower() == 'y':
        print(f"Skipping row {idx} due to invalid diagnosis: {diagnosis}")
        continue

    # Find the audio file that contains the Participant ID as a substring
    matching_files = [file for file in audio_files if participant_id in file]
    
    if matching_files:
        # If there's a match, use the first match found (assuming one file per participant ID)
        closest_match = matching_files[0]
        print(f"Found match for Participant ID {participant_id}: {closest_match}")

        # Determine the new file name based on the diagnosis
        if diagnosis == 'N':
            new_name = f'N_{normal_count}.wav'
            normal_count += 1
        else:
            new_name = f'D_{disorder_count}.wav'
            disorder_count += 1

        # Original file path
        original_file = os.path.join(audio_dir, closest_match)
        
        # New file path
        new_file = os.path.join(output_dir, new_name)
        
        # Copy and rename the file
        shutil.copyfile(original_file, new_file)
        
        # Append details to lists
        old_names.append(closest_match)
        new_names.append(new_name)
        ages.append(age)
        genders.append(gender)
        diagnoses.append(diagnosis)
        processed_count += 1
    else:
        print(f"No file found for Participant ID {participant_id} in {audio_dir}")

# Create a DataFrame with the mapping and save it to Excel if any files were processed
if processed_count > 0:
    output_df = pd.DataFrame({
        'Old Name': old_names,
        'New Name': new_names,
        'Age': ages,
        'Gender': genders,
        'Diagnosis': diagnoses
    })
    # Save to a new Excel file
    output_df.to_excel('renamed_audio_mapping.xlsx', index=False)
    print(f"Renaming complete. Processed {processed_count} files, and mapping saved to renamed_audio_mapping.xlsx.")
else:
    print("No files were processed. Check the input data and file paths.")
