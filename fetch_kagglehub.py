import kagglehub
import shutil
import os

print('Downloading Home Credit dataset via kagglehub...')
try:
    path = kagglehub.competition_download('home-credit-default-risk')
    print('Downloaded to:', path)
    
    # We need application_train.csv
    src_file = os.path.join(path, 'application_train.csv')
    dst_file = 'data/application_train.csv'
    
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        print(f'Successfully copied {src_file} to {dst_file}')
    else:
        print(f'Error: {src_file} not found inside downloaded contents.')
except Exception as e:
    print('Failed to download via kagglehub:', e)
